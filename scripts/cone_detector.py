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
from sensor_msgs.msg import Image, CameraInfo

from geometry_msgs.msg import PointStamped
import image_geometry


class ConeDetector:
    def __init__(self, engine_path, conf_threshold=0.5, iou_threshold=0.3):
        rospy.init_node("cones_position_node")

        #Inference
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        rospy.on_shutdown(self.cleanup)

        self.cfx = cuda.Device(0).make_context()
        """
        Crea un entorno de ejecucion en la GPU 0.
        Se asocia al hilo actual y ahi es cuando se puede reservar memoria
        (cuda.mem_alloc), copiar datos mem_alloc y ejecutar kernels

        """

        self.logger = trt.Logger(trt.Logger.INFO)
        """
        Este es el debug para el modelo
        """


        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            """Se abre el archivo en binario y creamos el runtime de tensorRT que
            es el encargado de interpretar y ejecutar el modelo"""
            
            self.engine = runtime.deserialize_cuda_engine(f.read())
            """Esto de encima es como convertir el modelo .engine en ejecutable"""
        
        self.context = self.engine.create_execution_context()
        """Esto es para crear un contexto de ejecucion, parecido al make_context"""

        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        """allocate_buffers reserva memoria en CPU y GPU y prepara la ejecucion.
        inputs = datos de entrada
        outputs = se guardan los resultados
        bindings = lista de punteros a memoria GPU
        self.stream = permite ejecutar operaciones en paralelo. cpy cpu-><-gpu, ejecutar modelo"""


        #Camera
        self.camera_model = image_geometry.PinholeCameraModel()
        self.current_boxes = []
        self.intrinsic_params = None


        #Publicadores y suscriptores de ROS
        self.image_pub = rospy.Publisher("/yolo/deteccion_conos", Image, queue_size=1)
        self.relative_poses_pub = rospy.Publisher("cones/relative_poses", PointStamped, queue_size=10)

        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback_yolo, queue_size=1, buff_size=2**24)

        self.mask_yolo_depth = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.callback_depth, queue_size=1, buff_size=2**24)

        self.camera_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.retrieve_camera_info)



        rospy.loginfo("Motor cargado con Bounding Boxes activas.")


    def cleanup(self):
        """Al usar make context hay que vaciar la memoria reservada por las posibles fugas de memoria y demas"""
        rospy.loginfo("Limpiando contexto")
        try:
            self.cfx.pop()
            rospy.loginfo("Memoria limpiada correctamente")
        except Exception as e:
            rospy.logerr(f"Error al limpiar la memoria: {e}")

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            """Reservar memoria en CPU para mejorar la velocidad en transacciones"""

            device_mem = cuda.mem_alloc(host_mem.nbytes)
            """Reservar memoria en GPU"""
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
                """Se indica donde esta la entrada (imagen) en CPU y en
                donde se copiará a la GPU"""
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                """Se indica donde se va a escribir el resultado dentro de la GPU
                y donde se va a copiar en CPU"""
        return inputs, outputs, bindings, stream

    def postprocess(self, outputs, orig_wh):
        output = outputs[0]['host'].reshape(self.engine.get_binding_shape(1))
        """Leemos los datos de salida que se copian a la CPU desde la GPU
        La GPU nos da una ristra 1D pero nosotros lo pasamos a tensores.
        En nuestro caso (N, C, H, W)-->(1, 3, 640, 640)
        N = Batch size
        C = Canales (RGB)
        H = Height
        W = width"""

        predictions = np.squeeze(output).T
        """Esto lo que hace es eliminar las dimensiones que valen 1
        En mi caso, (1,1,8400) ya que tengo solo 1 clase.
        Entonces esto me dejará una matriz con 8400 filas y 5 columnas,
        estas 5 columnas son x_center, y_center, width, height y score"""
        scores = predictions[:, 4] #Coger de todas las filas, la quinta columna
        mask = scores > self.conf_threshold #Elimino las que tengan poca confianza

        valid_predictions = predictions[mask]
        valid_scores = scores[mask]
        """Aplico la máscara de antes a las predicciones y a los scores de confianza"""

        if len(valid_scores) == 0:
            return [], [], [], []

        image_w, image_h = orig_wh
        x_factor = image_w/640
        y_factor = image_h/640
        """La camara trabaja en 640x480, pero YOLO lo hace en 640x640,
        entonces para poder pasar bien las coordenadas en la imagen de la camara,
        hay que multiplicarlo por el factor de la altura y el ancho.
        En el ancho no hay cambio pero en la altura si"""

        boxes = np.zeros_like(valid_predictions[:, :4])
        boxes[:, 0] = (valid_predictions[:, 0] - valid_predictions[:, 2]/2)*x_factor
        boxes[:, 1] = (valid_predictions[:, 1] - valid_predictions[:, 3]/2)*y_factor
        boxes[:, 2] = valid_predictions[:, 2] * x_factor
        boxes[:, 3] = valid_predictions[:, 3] * y_factor

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), valid_scores.tolist(), self.conf_threshold, self.iou_threshold)
        """Supresion de no maximos"""

        """Aqui se devuelve 1. coordenadas finales, 2. La confianza, 3. El id de la clase, 4. Indice de las cajas que sobrevivieron a NMS"""
        return boxes.astype(int), valid_scores, [0]*len(valid_scores), indices

    
    def callback_yolo(self, msg):
        self.cfx.push()
        """Tomar el control de la GPU"""
        try:
            #ROS->OPENCV
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg. width, -1)
            """Esto es para procesar los datos que nos llegan por el topic ya que no hay cv_bridge
            en ros melodic.
            Primero cogemos y lo pasamos a uint8, basicamente le decimos que cada numero es un entero
            de 8 bits (0, 255), que es lo estandar en imagenes.
            Con reshape le doy a los datos del mensaje la forma de 480x640 de la camara y el -1 es para
            que numpy coja los canales en funcion de lo que sobre.
            Basicamente es obtener la imagen a partir de la ristra de numeros"""

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            """OpenCV trabaja en formato BGR, por lo que hay que hacer la conversion desde RGB"""

            #PREPROCESADO
            input_image = cv2.resize(frame, (640,640))
            """Como el modelo trabaja en 640x640, pues hay que escalar la imagen"""
            
            input_image = input_image.transpose((2, 0, 1)).astype(np.float32)/255.0
            """Las redes neuronales primero leer primero toda la capa roja, luego toda la verde
            y al final toda la azul.
            Ir pixel a pixel seria ineficiente, por eso leen cada canal por separado.
            Bueno por la arquitectura SIMD(Single Instruction, Multiple Data). Basicamente que le
            pueden aplicar esa misma operacion a miles de numeros a la vez siempre y cuando esten pegados
            en la memoria"""

            input_image = np.ascontiguousarray(input_image)
            """Hace que los datos esten pegados unos a otros en la memoria para evitar que despues del
            transpose estén desordenados"""

            #INFERENCIA
            self.inputs[0]['host'] = input_image.ravel()
            """ravel hace que la imagen pase de (3, 640, 640) a una lista seguida de numeros."""

            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            """Pasar los datos de CPU a GPU"""

            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            """Este comando es el que aprieta el trigger para que la GPU empiece a calcular"""

            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            """Cuando la GPU acaba de calcular, se le devuelven los resultados a la CPU"""

            self.stream.synchronize()
            """La CPU espera aqui las respuestas de la GPU"""

            """Conocimiento del curso de CUDA de nvidia dado por Manuel Ujaldon:
                Aunque esto genera un "pipeline stall" (cuello de botella), en este sistema
                es aceptable ya que el tiempo de procesamiento total es menor al periodo 
                de muestreo de la cámara (33ms para 30 FPS)."""

            boxes, confs, ids, indices = self.postprocess(self.outputs, (msg.width, msg.height))

            if len(indices)>0:
                self.current_boxes = boxes[indices.flatten()] # Guardar para el callback de depth
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    conf = confs[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    label = f"Cono: {conf:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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


    def retrieve_camera_info(self, camera_info):
        self.camera_model.fromCameraInfo(camera_info)
        self.intrinsic_params = True

        self.camera_info_sub.unregister()

    def transform_into_relative_coordinates(self, u1, v1, u2, v2, distance_to_cone):
        u = (u1 + u2) / 2
        v = (v1 + v2) / 2
        cone_center_transformation_vector = self.camera_model.projectPixelTo3dRay((u, v))
        x_real = cone_center_transformation_vector[0]*distance_to_cone
        y_real = cone_center_transformation_vector[1]*distance_to_cone
        z_real = distance_to_cone
        #Falta comprobar este programa, esto da las coordenadas relativas al robot
        return x_real, y_real, z_real


    def callback_depth(self, depth_msg):
        if len(self.current_boxes) == 0:
            return

        depth_data = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)

        depth_data = depth_data.astype(float)/1000.0

        for bbox in self.current_boxes:
            x1, y1, w, h = map(int, bbox)
            x2, y2 = x1 + w, y1 + h
            roi = depth_data[y1:y2, x1:x2]

            mask = np.isfinite(roi) & (roi > 0.5) & (roi < 3.0)

            valid_points = roi[mask]
            
            if valid_points.size > 0:
                distance_to_cone = np.median(valid_points)

                x_c, y_c, z_c = self.transform_into_relative_coordinates(x1, y1, x2, y2, distance_to_cone)
                cono_msg = PointStamped()
                cono_msg.header.stamp = depth_msg.header.stamp

                cono_msg.header.frame_id = "camera_color_optical_frame"
        
                # 4. LAS COORDENADAS
                cono_msg.point.x = x_c
                cono_msg.point.y = y_c
                cono_msg.point.z = z_c
                
                # 5. ¡A VOLAR!
                self.relative_poses_pub.publish(cono_msg)

            else:
                pass


if __name__=='__main__':
    PATH_ENGINE = "/home/tx2/Development/racecar-ws/src/tfg_ariel/models/best.engine"
    try:
        ConeDetector(PATH_ENGINE, conf_threshold=0.25, iou_threshold = 0.3)
        rospy.spin()
    except Exception as e:
        rospy.logerr(f"Error: {e}")


