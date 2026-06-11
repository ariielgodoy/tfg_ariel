#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from tf.transformations import euler_from_quaternion

class PurePursuit:
    def __init__(self):
        self.L   = 0.325 
        self.Lt  = 0.3 
        self.MAX_STEER = 1.0 
        
        self.waypoints = []
        self.waypoint_idx = 0
        
        # Variables para almacenar la pose actual
        self.current_x = None
        self.current_y = None
        self.current_theta = None

        # Suscriptor al mapa dinámico del VisualMapper
        rospy.Subscriber("/slam/map_cones", MarkerArray, self.map_callback)
        
        # Suscriptor a la pose global publicada por el VisualMapper
        rospy.Subscriber("/coche/pose_global", PoseStamped, self.pose_callback)
        
        self.mov_publisher = rospy.Publisher("/vesc/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)

    def pose_callback(self, msg):
        # Actualizamos la pose interna con la información del tópico global
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        
        # Convertimos el cuaternión a Euler (yaw)
        rot = msg.pose.orientation
        _, _, self.current_theta = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])

    def map_callback(self, marker_array):
        puntos = np.array([[m.pose.position.x, m.pose.position.y] for m in marker_array.markers])
        if len(puntos) > 2:
            self.waypoints = self.compute_waypoints(puntos)
            self.waypoint_idx = 0

    def compute_waypoints(self, points, ancho_pista=1.10, separacion_conos=0.5):
        waypoints = []
        if len(points) == 0: return np.array(waypoints)
        
        inicio_derecho = np.array([0.0, -ancho_pista / 2])
        idx_actual = np.argmin(np.linalg.norm(points - inicio_derecho, axis=1))
        
        cadena_derecha = [idx_actual]
        visitados_derecha = {idx_actual}
        direccion_marcha = np.array([1.0, 0.0])
        
        for _ in range(len(points)):
            p_actual = points[idx_actual]
            best_next_idx = None
            min_error_dist = float('inf')
            
            for idx, p_cand in enumerate(points):
                if idx in visitados_derecha: continue
                v_cand = p_cand - p_actual
                dist = np.linalg.norm(v_cand)
                
                if (separacion_conos * 0.5) < dist < (separacion_conos * 1.8):
                    if np.dot(v_cand, direccion_marcha) > 0:
                        error_dist = abs(dist - separacion_conos)
                        if error_dist < min_error_dist:
                            min_error_dist = error_dist
                            best_next_idx = idx
                            
            if best_next_idx is not None:
                v_progreso = points[best_next_idx] - points[idx_actual]
                direccion_marcha = v_progreso / np.linalg.norm(v_progreso)
                idx_actual = best_next_idx
                cadena_derecha.append(idx_actual)
                visitados_derecha.add(idx_actual)
            else: break
                
        for idx_der in cadena_derecha:
            p_der = points[idx_der]
            best_izq = None
            min_dist_transversal = float('inf')
            for idx, p_cand in enumerate(points):
                if idx in visitados_derecha: continue
                dist = np.linalg.norm(p_cand - p_der)
                if dist < min_dist_transversal:
                    min_dist_transversal = dist
                    best_izq = p_cand
            if best_izq is not None and min_dist_transversal < (ancho_pista * 1.5):
                waypoints.append((p_der + best_izq) / 2.0)
        return np.array(waypoints)

    def publish_movement(self, speed, angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.speed = speed
        msg.drive.steering_angle = angle
        self.mov_publisher.publish(msg)

    def pure_pursuit(self):
        # Usamos las variables globales actualizadas por el callback
        if self.current_x is None or len(self.waypoints) == 0:
            self.publish_movement(0.0, 0.0)
            return

        x, y, theta = self.current_x, self.current_y, self.current_theta

        # Buscar el punto más cercano para actualizar waypoint_idx
        dists = np.linalg.norm(self.waypoints - np.array([x, y]), axis=1)
        self.waypoint_idx = np.argmin(dists)

        # Buscar punto de lookahead
        target_point = None
        for i in range(self.waypoint_idx, len(self.waypoints)):
            if np.linalg.norm(self.waypoints[i] - np.array([x, y])) >= self.Lt:
                target_point = self.waypoints[i]
                break
        
        if target_point is None: target_point = self.waypoints[-1]

        # Convertir a coordenadas locales del coche
        dx = target_point[0] - x
        dy = target_point[1] - y
        x_local = dx * np.cos(theta) + dy * np.sin(theta)
        y_local = -dx * np.sin(theta) + dy * np.cos(theta)

        # Ley de control
        Lt_sq = x_local**2 + y_local**2
        if Lt_sq < 1e-6: return
        delta = -np.arctan2(2 * self.L * y_local, Lt_sq)
        
        self.publish_movement(0.45, np.clip(delta, -self.MAX_STEER, self.MAX_STEER))

def main():
    rospy.init_node("Autonomous_Driving")
    pp = PurePursuit()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if len(pp.waypoints) > 0:
            pp.pure_pursuit()
        rate.sleep()

if __name__ == '__main__':
    main()