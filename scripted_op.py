#!/usr/bin/env python
""" Node that outputs velocities from the keyboard inputs"""

import rospy
from geometry_msgs.msg import Twist
import motor_command_model as mcm
import me416_utilities as mu


def main():

    rospy.init_node('scripted_op', anonymous='True')

    pub = rospy.Publisher('robot_twist', Twist, queue_size=10)

    tmsg = Twist()

    rate = rospy.Rate(1)

    #import the file test.csv

    csv = './test.csv'

    speeds = mu.read_two_columns_csv(csv)

    i = 0
    while not rospy.is_shutdown() and i < len(speeds) - 1:

        i = i + 1

        tmsg.linear.x = speeds[i][0]
        tmsg.angular.z = speeds[i][1]

        pub.publish(tmsg)

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    finally:
        #This is the place to put any "clean up" code that should be executed
        #on shutdown even in case of errors, e.g., closing files or windows
        pass
