#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

import rospy
from geometry_msgs.msg import PoseStamped, PoseArray, TransformStamped
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from tf.transformations import quaternion_from_euler

class VisualMapper:
    def __init__(self):
        self.Posicion_Coche_Global = [0.0, 0.0, 0.0]
        self.Mapa_Global_Conos = MarkerArray()      # Lista de diccionarios: {'id', 'x', 'y', 'color'}
        self.Contador_IDs = 0                            # Para asignar identificadores únicos
        self.Umbral_Asociacion = 0.25                    # Distancia máxima en metros para considerar el mismo cono
        self.initial_ground_truth_obtained = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.ultima_timestamp_tf = rospy.Time(0)
        #Queremos la posicion desde el mapa hasta la camara
        self.target_frame = "base_link"
        self.source_frame = "camera_link_optical"

        #Publishers y subscribers
        rospy.Subscriber("/cones/relative_poses", PoseArray, self.visual_slam)
        self.map_pub = rospy.Publisher(
            "/slam/map_cones", MarkerArray, queue_size=10
        )
        self.pose_pub = rospy.Publisher('/coche/pose_global', PoseStamped, queue_size=10)

    def get_transformation(self, Lista_Conos_Camara):
        try:
            # SOLUCIÓN EXTRAPOLACIÓN: Cambiamos el timestamp del mensaje por rospy.Time(0).
            # Como la cámara no se mueve de su sitio respecto al coche, nos vale con la última
            # transformada conocida en el buffer, ignorando desfases temporales del simulador.
            self.transformation = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame, 
                rospy.Time(0), # <--- Antes: Lista_Conos_Camara.header.stamp
                rospy.Duration(0.1)
            )
            #rospy.logerr('Success finding the transformation from %s to %s'
            #            % (self.source_frame, self.target_frame))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            
            # SOLUCIÓN CRASH: Envolvemos las variables en una tupla () para corregir el formateo
            rospy.logerr('Unable to find the transformation from %s to %s'
                        % (self.source_frame, self.target_frame))
            self.transformation = None
            
        

    def compute_initial_cones_pose(self, Lista_Conos_Camara):
        for pose_msg in Lista_Conos_Camara.poses:
            
            # 2. CORRECCIÓN: Convertimos la Pose en PoseStamped para que tf2 no explote
            cono_cam = PoseStamped()
            cono_cam.header = Lista_Conos_Camara.header # Conserva el timestamp y el frame de la cámara
            cono_cam.pose = pose_msg

            marker_cono = Marker()
            marker_cono.header.frame_id = "map"
            marker_cono.header.stamp = rospy.Time.now()
            
            # Usamos self. porque es un atributo de clase
            marker_cono.id = self.Contador_IDs
            self.Contador_IDs += 1
            
            marker_cono.type = Marker.CYLINDER
            marker_cono.action = Marker.ADD
            
            # Ahora tf2_geometry_msgs recibirá el PoseStamped correcto
            coords_globales = self.Transformar_a_Base_Link(cono_cam)

            marker_cono.pose.position.x = coords_globales.pose.position.x
            marker_cono.pose.position.y = coords_globales.pose.position.y
            marker_cono.pose.position.z = 0.0
            marker_cono.pose.orientation.w = 1.0  

            # Dimensiones y opacidad básicas para RViz
            marker_cono.scale.x = 0.2
            marker_cono.scale.y = 0.2
            marker_cono.scale.z = 0.3
            marker_cono.color.r = 1.0
            marker_cono.color.a = 1.0

            self.Mapa_Global_Conos.markers.append(marker_cono)
            
        self.initial_ground_truth_obtained = True
        #Hay q mirar como pasar los posearray a markerarray

    def Transformar_a_Base_Link(self, cono_cam):
        #Esto me hace la transformacion a las coordenadas globales
        return tf2_geometry_msgs.do_transform_pose(cono_cam, self.transformation)

    def Calcular_Distancia_Euclidea(self, x1, y1, x2, y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    def Encontrar_Cono_Mas_Cercano_En_Mapa(self, cono_global_est, Mapa_Global_Conos):
        # CORRECCIÓN ROS: El acceso correcto a PoseStamped es a través de .position
        x_est = cono_global_est.pose.position.x
        y_est = cono_global_est.pose.position.y
        
        cono_mas_cercano = None
        distancia_minima = float('inf') # Inicializamos con "infinito"
        
        for cono in Mapa_Global_Conos.markers:
            x_global = cono.pose.position.x
            y_global = cono.pose.position.y

            # Calculamos la distancia al cono de la iteración actual
            distancia_actual = self.Calcular_Distancia_Euclidea(x_est, y_est, x_global, y_global)
            
            # Si este cono está más cerca que el más cercano que habíamos visto antes:
            if distancia_actual < distancia_minima:
                distancia_minima = distancia_actual # Actualizamos el récord de distancia corta
                cono_mas_cercano = cono             # Nos guardamos el objeto Marker entero
                
        # Devolvemos el objeto del mapa que ha ganado (o None si el mapa estaba vacío)
        return cono_mas_cercano
    

    def Estimar_Pose_Por_SVD(self, Conos_Viejos_Emparejados):
        """Usa TODOS los conos emparejados para encontrar la posición óptima

        del coche minimizando el error (Algoritmo de Kabsch).
        """
        # 1. Extraer matrices de puntos locals (coche) y globales (mapa)
        puntos_locales = np.array([
            [pareja[0].pose.position.x, pareja[0].pose.position.y] 
            for pareja in Conos_Viejos_Emparejados
        ])
        puntos_globales = np.array([
            [pareja[1].pose.position.x, pareja[1].pose.position.y] 
            for pareja in Conos_Viejos_Emparejados
        ])

        # 2. Calcular centroides
        c_local = np.mean(puntos_locales, axis=0)
        c_global = np.mean(puntos_globales, axis=0)

        # 3. Centrar las nubes de puntos
        pl_centrado = puntos_locales - c_local
        pg_centrado = puntos_globales - c_global

        # 4. Matriz de covarianza y SVD
        H = np.dot(pl_centrado.T, pg_centrado)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Corrección geométrica por si hay reflexión
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # 5. Calcular traslación y ángulo óptimos
        T = c_global - np.dot(R, c_local)
        X_global = T[0]
        Y_global = T[1]
        Theta_global = np.arctan2(R[1, 0], R[0, 0])

        return [X_global, Y_global, Theta_global]
    
    def publicar_pose(self):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map" # Es importante que esté en "map"
        
        # Posición
        msg.pose.position.x = self.Posicion_Coche_Global[0]
        msg.pose.position.y = self.Posicion_Coche_Global[1]
        msg.pose.position.z = 0.0
        
        # Orientación (Conversión de Euler a Cuaternión)
        # theta es el tercer valor de tu lista [x, y, theta]
        q = quaternion_from_euler(0, 0, self.Posicion_Coche_Global[2])
        
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        
        self.pose_pub.publish(msg)

    def Trasladar_a_Global(self, cono_local, coche):
        """Transforma las coordenadas de un cono del sistema local del coche

        al sistema de referencia global del mapa.
        coche = [X_global, Y_global, Theta_global]
        """
        # 1. Desempaquetar la pose actual del coche
        x_coche, y_coche, theta_coche = coche

        # 2. Extraer las coordenadas locales del cono (sintaxis correcta de ROS)
        x_local = cono_local.pose.position.x
        y_local = cono_local.pose.position.y

        # 3. Aplicar rotación + traslación (Geometría epipolar estándar)
        x_global = x_coche + (
            x_local * np.cos(theta_coche) - y_local * np.sin(theta_coche)
        )
        y_global = y_coche + (
            x_local * np.sin(theta_coche) + y_local * np.cos(theta_coche)
        )

        # 4. Construir el nuevo mensaje PoseStamped global
        cono_global = PoseStamped()
        cono_global.header.frame_id = "map"  # El cono ahora vive en el mapa global
        cono_global.header.stamp = cono_local.header.stamp

        cono_global.pose.position.x = x_global
        cono_global.pose.position.y = y_global
        cono_global.pose.position.z = 0.0
        cono_global.pose.orientation.w = 1.0  # Orientación neutra en el mapa

        return cono_global
    
    def publicar_transformada_coche(self, Lista_Conos_Camara):
        # 1. Sincronizamos el tiempo con el simulador para que no proteste
        stamp_actual = Lista_Conos_Camara.header.stamp if Lista_Conos_Camara.header.stamp.to_sec() > 0 else rospy.Time.now()
        
        # Filtro rápido para que no te llene la terminal de avisos amarillos
        if stamp_actual <= self.ultima_timestamp_tf:
            return

        t = TransformStamped()
        t.header.stamp = stamp_actual
        t.header.frame_id = "map"          # Tu nuevo suelo fijo para los conos
        t.child_frame_id = "odom"          # Engancha con el simulador

        # CERO MATEMÁTICAS: Ponemos todo a cero absoluto. El mapa no se mueve.
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # Cuaternión neutro (sin rotación)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        # Publicamos el frame en el árbol de ROS
        self.tf_broadcaster.sendTransform(t)
        self.ultima_timestamp_tf = stamp_actual

    def visual_slam(self, Lista_Conos_Camara):
        self.publicar_transformada_coche(Lista_Conos_Camara)
        self.get_transformation(Lista_Conos_Camara)
        if not self.transformation:
            return
        
        Conos_Viejos_Emparejados = []
        Conos_Nuevos_Detectados = []
        if not self.initial_ground_truth_obtained:
            self.compute_initial_cones_pose(Lista_Conos_Camara)
            return

        for cono_cam_pose in Lista_Conos_Camara.poses:
            cono_cam = PoseStamped()
            cono_cam.header = Lista_Conos_Camara.header # Conserva el timestamp y el frame de la cámara
            cono_cam.pose = cono_cam_pose
            cono_base_link = self.Transformar_a_Base_Link(cono_cam)
            cono_global_est = self.Trasladar_a_Global(cono_base_link, self.Posicion_Coche_Global)

            Cono_Asociado = self.Encontrar_Cono_Mas_Cercano_En_Mapa(cono_global_est, self.Mapa_Global_Conos)
            Distancia = self.Calcular_Distancia_Euclidea(cono_global_est.pose.position.x,
                                                         cono_global_est.pose.position.y,
                                                         Cono_Asociado.pose.position.x,
                                                         Cono_Asociado.pose.position.y)
            
            if Distancia < self.Umbral_Asociacion:
                Conos_Viejos_Emparejados.append([cono_base_link, Cono_Asociado])
            else:
                Conos_Nuevos_Detectados.append(cono_base_link)
        
        if len(Conos_Viejos_Emparejados) >= 2:
            #Nva_Posicion_Coche_Global = self.Estimar_Pose_Por_Trigonometria(Conos_Viejos_Emparejados)
            Nva_Posicion_Coche_Global = self.Estimar_Pose_Por_SVD(Conos_Viejos_Emparejados)
            self.Posicion_Coche_Global = Nva_Posicion_Coche_Global
        else:
            self.Posicion_Coche_Global = self.Posicion_Coche_Global
            print "No hay suficientes conos conocidos para determinar la pose del coche"

        for cono_nuevo in Conos_Nuevos_Detectados:
            coords_globales_reales = self.Trasladar_a_Global(cono_nuevo, self.Posicion_Coche_Global)


            marker_cono = Marker()
            marker_cono.header.frame_id = "map"
            marker_cono.header.stamp = rospy.Time.now()
            
            # 2. ASIGNACIÓN DE ID: Usamos el contador global y lo sumamos para el siguiente
            marker_cono.id = self.Contador_IDs
            self.Contador_IDs += 1
            
            # Configuración geométrica del Marker
            marker_cono.type = Marker.CYLINDER
            marker_cono.action = Marker.ADD

            marker_cono.pose.position.x = coords_globales_reales.pose.position.x
            marker_cono.pose.position.y = coords_globales_reales.pose.position.y
            marker_cono.pose.position.z = 0.0
            marker_cono.pose.orientation.w = 1.0  # Evita que RViz se queje en rojo

            #Estilo en rviz
            marker_cono.scale.x = 0.2
            marker_cono.scale.y = 0.2
            marker_cono.scale.z = 0.3
            marker_cono.color.r = 1.0
            marker_cono.color.g = 1.0  # Blanco por defecto
            marker_cono.color.b = 1.0
            marker_cono.color.a = 1.0
            self.Mapa_Global_Conos.markers.append(marker_cono)

        x, y, theta = self.Posicion_Coche_Global
        print "Posicion coche -> X: %.1f, Y: %.1f, Theta: %.1f" % (x, y, theta)
        self.map_pub.publish(self.Mapa_Global_Conos)



def main():
    rospy.init_node('visual_mapper_node')
    VisualMapper()
    rospy.spin()

if __name__=='__main__':
    main()
