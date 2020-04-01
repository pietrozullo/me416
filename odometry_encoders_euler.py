#!/usr/bin/env python
"""This node keeps an estimate of the state of the robot.
The estimate is initialized as the zero state where x,y,theta = 0 .
Every time a msg is sent from the motor_speeds topic the nodes update the estimate by using the close form step.
The result will be pubilished on pse_arcs topic"""
import rospy
import motor_command_model as mcm
import me416_utilities as mu
import numpy as np
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from me416_lab.msg import MotorSpeedsStamped

z_zero = np.array([[0.0], [0.0], [0.0]])


def callback(tmsg):

    global pub
    global z_zero

    #get info from the content of the message
    u = np.array([[tmsg.left], [tmsg.right]])
    #calculate time thanks to StampedMsgRegister class
    # T = mcm.StampedMsgRegister().replace_and_compute_delay(tmsg)
    # T = T[0]
    #set the step size
    stepSize = 1.0
    #set the new message's format, and fill it with the pose
    pose = Pose2D()
    pose.x = mcm.euler_step(z_zero, u, stepSize)[0][0]
    pose.y = mcm.euler_step(z_zero, u, stepSize)[1][0]
    pose.theta = mcm.euler_step(z_zero, u, stepSize)[2][0]

    z_zero = z_zero + mcm.euler_step(z_zero, u, stepSize)

    #publish the pose on the topic pose_arcs
    pub.publish(pose)
    rospy.loginfo(u)


def main():
    """"""

    global pub
    global z_zer

    #initialize the node and the Ti object
    rospy.init_node('odometry_wheel_arcs', anonymous='True')

    #define publisher and subscriber
    rospy.Subscriber('encoders', MotorSpeedsStamped, callback)
    pub = rospy.Publisher('pose_euler', Pose2D, queue_size=10)

    #loop
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    finally:
        #This is the place to put any "clean up" code that should be executed
        #on shutdown even in case of errors, e.g., closing files or windows
        pass
