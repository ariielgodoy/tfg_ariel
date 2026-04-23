#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class cone_mapper:
    def __init__(self):
        rospy.init_node('cone_mapper_node')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.sub_cones_relative_poses = rospy.Subscriber('cones/relative_poses', PointStamped, self.transform_into_global_coordinates_callback)
        self.pub_cones_global_poses = rospy.Publisher('cones/global_poses', PointStamped, queue_size=10)

        rospy.loginfo("Mapper node initialized and waiting for data...")

    def transform_into_global_coordinates_callback(self, relative_cone_position):
        try:
            # Buscamos la transformación exacta en el momento en que se tomó la foto (stamp)
            # Esto compensa el retraso del procesado de YOLO
            transform = self.tf_buffer.lookup_transform(
                "base_link", 
                relative_cone_position.header.frame_id, 
                relative_cone_position.header.stamp, 
                rospy.Duration(0.1)
            )

            # Aplicamos la transformación al punto
            p_global = tf2_geometry_msgs.do_transform_point(relative_cone_position, transform)

            # Publicamos el resultado
            self.pub_cones_global_poses.publish(p_global)
            
            rospy.loginfo("Cono detectado en Mapa: X=%.2f, Y=%.2f", 
                          p_global.point.x, p_global.point.y)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Error en la transformación: %s", str(e))


if __name__ == '__main__':
    try:
        mapper = cone_mapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass