#!/usr/bin/env python3
from ultralytics import YOLO
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
from sensor_msgs.msg import CameraInfo
import image_geometry
from geometry_msgs.msg import PointStamped
import tf
#from map_creation import MapCones


class ConeDetector:
    def __init__(self):
        rospy.init_node("detection_node")
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.detection_model = YOLO("/home/ariel/catkin_racecar_ws/src/fs_perception_pkg/models/best.pt")
        self.camera_model = image_geometry.PinholeCameraModel()
        self.publish_detection = rospy.Publisher("/yolo/detection/image", Image)
        self.detect = rospy.Subscriber("/camera/color/image_raw", Image, self.detect_cones, queue_size=1, buff_size=2**24)
        self.mask_yolo_depth = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_operation, queue_size=1, buff_size=2**24)
        self.current_boxes = []
        self.cones_map = []
        self.intrinsic_params = None
        self.camera_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.retrieve_camera_info)

    def retrieve_camera_info(self, camera_info):
        self.camera_model.fromCameraInfo(camera_info)
        self.intrinsic_params = True

        self.camera_info_sub.unregister()


    def detect_cones(self, image_to_predict):
        cv_image = self.bridge.imgmsg_to_cv2(image_to_predict, desired_encoding="bgr8")
        detection_result = self.detection_model(cv_image)
        self.current_boxes = detection_result[0].boxes.xyxy.cpu().numpy().astype(int)
        detection_annotated = detection_result[0].plot(show=False) #Es una lista porque YOLO podria procesar mas de una imagen a la vez
        ros_msg = self.bridge.cv2_to_imgmsg(detection_annotated, encoding="bgr8")
        self.publish_detection.publish(ros_msg)


    def transform_into_global_coordinates(self, relative_cone_position):
        p = PointStamped()
        p.header.frame_id = "camera_link_optical"
        p.header.stamp = rospy.Time(0)
        p.point.x = relative_cone_position[0]
        p.point.y = relative_cone_position[1]
        p.point.z = relative_cone_position[2]

        #try:
        p_global = self.listener.transformPoint("map", p)
        print("Coordenada global del cono: ", p_global)
        return p_global.point.x, p_global.point.y
        #except:
            #rospy.logwarn("Esta el coche en el mapa?")
            #return None


    def transform_into_relative_coordinates(self, u1, v1, u2, v2, distance_to_cone):
        u = (u1 + u2) / 2
        v = (v1 + v2) / 2
        cone_center_transformation_vector = self.camera_model.projectPixelTo3dRay((u, v))
        x_real = cone_center_transformation_vector[0]*distance_to_cone
        y_real = cone_center_transformation_vector[1]*distance_to_cone
        z_real = distance_to_cone
        #Falta comprobar este programa, esto da las coordenadas relativas al robot
        return x_real, y_real, z_real



    #Funcion para devolver la distancia del cono con la mediana de la medida dentro del bounding box completo
    def depth_operation(self, depth_data):
        if len(self.current_boxes) == 0:
            return
        

        depth_data = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")
        for bbox in self.current_boxes:
            x1, y1, x2, y2 = bbox
            roi = depth_data[y1:y2, x1:x2]

            #Explicar que se usa una mascara que no tiene base en redes neuronales porque es mas ligero
            mask = np.isfinite(roi) & (roi > 0.5) & (roi < 3.0)

            valid_points = roi[mask]
            
            if valid_points.size > 0:
                distance_to_cone = np.median(valid_points)

                relative_cone_position = self.transform_into_relative_coordinates(x1, y1, x2, y2, distance_to_cone)
                gobal_cone_position = self.transform_into_global_coordinates(relative_cone_position)
                
            else:
                pass
            #Aqui debo poner que haga algo

        
        



def main():
    node = ConeDetector()
    rospy.spin()


if __name__ == '__main__':
    main()