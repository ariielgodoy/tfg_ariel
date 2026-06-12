#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, PoseArray
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np




class PurePursuitReactivo:
    def __init__(self):
        self.L   = 0.325 #m Distancia entre ejes (Batalla del coche)
        self.Lt  = 0.5 #m Distancia de mirada (Lookahead distance)
        self.MAX_STEER = 3.14/2 # (En radianes) Ángulo máximo de giro físico de las ruedas

        self.current_speed = 0.0
        self.current_angle = 0.0
        
        # Ajusta estos valores según cuánto "tiron" quieras.
        # Más bajo = más suave (tarda más en acelerar).
        self.MAX_ACCEL = 0.02      # Aumento de velocidad por ciclo
        self.MAX_STEER_RATE = 0.05 # Aumento de ángulo por ciclo (radianes)

        self.conos_memorizados = []
        self.cone_id_counter = 0

        #Publishers y subscribers
        self.relative_pose_subscriber = rospy.Subscriber("/cones/relative_poses", PoseArray, self.calcular_conduccion_central)
        self.mov_publisher = rospy.Publisher("/vesc/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)

    """
    El proceso que debo hacer es:
    Calcular los conos mas cercanos a izquierda y derecha del coche.
    Con eso calculo el punto medio más proximo.
    Después, segun el primer cono en la derecha detectado, continuo mirando a derecha si
    hay otro cono entre 20 y 100 centimetros del cono que he detectado antes en derecha.
    Como ya he triado los conos de la derecha, miro ahora en la izquierda, buscando el cono más cercano
    de la izquierda al cono que tengo ahora seleccionado en la derecha, y asi voy generando la trayectoria.
    """

    def publish_movement(self, target_speed, target_angle):

        # 3. Publicar los valores suavizados
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.speed = target_speed
        msg.drive.steering_angle = target_angle
        self.mov_publisher.publish(msg)



    def calcular_conduccion_central(self, poses_msg):
        # 1. Extraer coordenadas locales (X: lateral, Z: profundidad)
        conos_detectados = [{'x': p.position.x, 'z': p.position.z} for p in poses_msg.poses]

        if len(conos_detectados) < 1:
            self.publish_movement(0.0, 0.0)
            return

        puntos_medios = []
        HALF_TRACK = 0.5  # La mitad del pasillo de 1m

        # 2. TU HEURÍSTICA: Clasificación directa por plano de simetría del coche
        for cono in conos_detectados:
            if cono['x'] > 0:
                # Cono Izquierdo: el centro de la pista está a su derecha
                x_centro = cono['x'] - HALF_TRACK
                puntos_medios.append({'x': x_centro, 'z': cono['z']})
            else:
                # Cono Derecho: el centro de la pista está a su izquierda
                x_centro = cono['x'] + HALF_TRACK
                puntos_medios.append({'x': x_centro, 'z': cono['z']})

        # 3. Ordenar la trayectoria central generada por profundidad (Z)
        trayectoria = sorted(puntos_medios, key=lambda p: p['z'])
        
        # 4. Buscar el punto de Lookahead (self.Lt)
        target_point = trayectoria[0]
        for punto in trayectoria:
            dist_coche = np.sqrt(punto['x']**2 + punto['z']**2)
            if dist_coche >= self.Lt:
                target_point = punto
                break

        # 5. Ley de control de Pure Pursuit
        # ====================================================
        # ⚠️ ¡ALERTA DE SIGNO!: Si el coche sigue apuntando a los conos, 
        # cambia 'x_target = target_point['x']' por 'x_target = -target_point['x']'
        # Un signo invertido aquí hace que el coche gire HACIA el peligro en vez de huir.
        # ====================================================
        x_target = -target_point['x'] 
        Lt_squared = target_point['x']**2 + target_point['z']**2
        
        delta = -np.arctan2(2 * self.L * x_target, Lt_squared)
        
        # Acotamos el giro máximo del coche
        if delta > self.MAX_STEER: delta = self.MAX_STEER
        if delta < -self.MAX_STEER: delta = -self.MAX_STEER

        # Publicamos velocidad constante y el giro calculado
        self.publish_movement(0.4, delta)
    


def main():
    rospy.init_node("Autonomous_Driving")
    PurePursuitReactivo()
    rospy.spin()



if __name__ == '__main__':
    main()