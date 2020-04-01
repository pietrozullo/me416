#!/usr/bin/env python
""" Node that outputs velocities from the keyboard inputs"""

import rospy
from geometry_msgs.msg import Twist
import motor_command_model as mcm
import me416_utilities as mu


def main():
    """The node permits to change velocities of the robot through the press of certain keys on your keyboard:
    press:
    w - accelerate/increases linear velocities in front direction
    a - turn left
    s - decelerate/decreases linear velocity in front direction
    d - turn right
    q - quit
    """

    rospy.init_node('key_op', anonymous='True')

    pub = rospy.Publisher('robot_twist', Twist, queue_size=10)

    tmsg = Twist()

    rate = rospy.Rate(50)

    speed = mcm.KeysToVelocities()

    getch = mu._Getch()

    while not rospy.is_shutdown():

        key = getch()

        if key == 'q':

            rospy.loginfo('Shutdown Initiated')
            rospy.signal_shutdown('Shutting down initiated by %s' %
                                  rospy.get_name())

        else:

            speedv = speed.update_speeds(key)

            tmsg.linear.x = speedv[0]
            tmsg.angular.z = speedv[1]
            pub.publish(tmsg)

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    finally:
        #This is the place to put any "clean up" code that should be executed
        #on shutdown even in case of errors, e.g., closing files or windows
        pass
