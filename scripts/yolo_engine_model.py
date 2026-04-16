#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Puente para encontrar rospy en Melodic
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)

import rospy
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sensor_msgs.msg import Image

class YOLOv8TRT:
    def __init__(self, engine_path, conf_threshold=0.5, iou_threshold=0.4):
        rospy.init_node('nodo_vision_trt', anonymous=True)
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold


        rospy.on_shutdown(self.cleanup)
        # 1. Gestión explícita del contexto de CUDA
        self.cfx = cuda.Device(0).make_context()
        
        # Configuración de TensorRT
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # Preparar memoria en la GPU
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        # Publisher y Subscriber
        self.image_pub = rospy.Publisher("/deteccion_conos", Image, queue_size=1)
        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        
        rospy.loginfo("✅ Motor cargado con Bounding Boxes activas.")

    def cleanup(self):
        """Limpia el contexto de CUDA al cerrar el nodo"""
        rospy.loginfo("🧹 Limpiando contexto de CUDA...")
        try:
            # Vaciamos el stack de contextos
            self.cfx.pop()
            rospy.loginfo("✅ Contexto cerrado limpiamente.")
        except Exception as e:
            rospy.logerr(f"Error al cerrar contexto: {e}")

            
    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def postprocess(self, outputs, orig_wh):
        # 1. Reformatear salida (YOLOv8 suele ser [1, clases+4, 8400])
        output = outputs[0]['host'].reshape(self.engine.get_binding_shape(1))
        predictions = np.squeeze(output).T  # Shape: (8400, 5) si es 1 clase

        # 2. FILTRADO VECTORIZADO (Sustituye al bucle FOR)
        # Suponiendo que el índice 4 es la confianza del cono
        scores = predictions[:, 4]
        mask = scores > self.conf_threshold
        
        # Solo procesamos lo que ha pasado el filtro
        valid_predictions = predictions[mask]
        valid_scores = scores[mask]

        if len(valid_scores) == 0:
            return [], [], [], []

        # 3. ESCALADO VECTORIZADO
        image_w, image_h = orig_wh
        x_factor = image_w / 640
        y_factor = image_h / 640

        # Convertir centro x,y,w,h a top, left, w, h de una sola vez
        # Evitamos calcular uno por uno
        boxes = np.zeros_like(valid_predictions[:, :4])
        boxes[:, 0] = (valid_predictions[:, 0] - valid_predictions[:, 2]/2) * x_factor # x_min
        boxes[:, 1] = (valid_predictions[:, 1] - valid_predictions[:, 3]/2) * y_factor # y_min
        boxes[:, 2] = valid_predictions[:, 2] * x_factor # width
        boxes[:, 3] = valid_predictions[:, 3] * y_factor # height

        # 4. NMS (Esto es rápido porque ya hay pocas cajas)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), valid_scores.tolist(), self.conf_threshold, self.iou_threshold)
        
        return boxes.astype(int), valid_scores, [0]*len(valid_scores), indices

    def callback(self, msg):
        self.cfx.push()
        try:
            # Conversión ROS -> OpenCV
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Pre-procesado
            input_image = cv2.resize(frame, (640, 640))
            input_image = input_image.transpose((2, 0, 1)).astype(np.float32) / 255.0
            input_image = np.ascontiguousarray(input_image)

            # Inferencia
            self.inputs[0]['host'] = input_image.ravel()
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # --- DIBUJAR CAJAS ---
            boxes, confs, ids, indices = self.postprocess(self.outputs, (msg.width, msg.height))
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    conf = confs[i]
                    # Dibujar rectángulo
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Añadir texto con confianza
                    label = f"Cono: {conf:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Publicar imagen con cajas
            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = frame.shape[0]
            out_msg.width = frame.shape[1]
            out_msg.encoding = "bgr8"
            out_msg.step = frame.shape[1] * 3
            out_msg.data = frame.tobytes()
            self.image_pub.publish(out_msg)

        finally:
            self.cfx.pop()

if __name__ == '__main__':
    PATH_ENGINE = "/home/tx2/Development/racecar-ws/src/tfg_ariel/models/best_fp16.engine"
    try:
        # Puedes ajustar conf_threshold (sensibilidad) y iou_threshold (solapamiento)
        YOLOv8TRT(PATH_ENGINE, conf_threshold=0.45, iou_threshold=0.4)
        rospy.spin()
    except Exception as e:
        rospy.logerr(f"❌ Error: {e}")