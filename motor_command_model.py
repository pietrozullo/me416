"""Functions for modeling ROSBot"""

import numpy as np
from math import cos, sin


def model_parameters():
    """Returns two constant model parameters"""
    global k
    global d
    k = 3.73
    d = 0.7
    return k, d


def twist_to_speeds(tmsg):
    """Given the desired angular and linear velocity of the robot, returns normalized speeds fort he left and right motor. Speeds need to b    e hresholded to be between -1.0 (backward at maximum speed) and 1.0 (forward at maximum speed) """

    k = model_parameters()[0]
    d = model_parameters()[1]
    #turns right offset lower
    speed_offset = 0.00
    speed_angular = tmsg.angular.z
    speed_linear = tmsg.linear.x

    right = ((speed_linear - d * speed_angular) / k)
    left = ((speed_linear + d * speed_angular) / k)

    left = left
    right = right + speed_offset

    if left > 1:
        left = 1

    if left < -1:
        left = -1
    if right > 1:
        right < -1
    if right < -1:
        right = -1

    return right, left


class KeysToVelocities():
    def __init__(self):

        self.speed_linear = 0.0
        self.speed_angular = 0.0
        self.SPEED_DELTA = 0.2

    def update_speeds(self, key):

        if key == 'w' or key == 'W':
            self.speed_linear = self.speed_linear + self.SPEED_DELTA
            print('Increment Linear Speed')

        if key == 's' or key == 'S':
            self.speed_linear = self.speed_linear - self.SPEED_DELTA
            print('Decrement Linear Speed ')

        if key == 'a' or key == 'A':
            self.speed_angular = self.speed_angular + self.SPEED_DELTA
            print('Turn Right')
        if key == 'd' or key == 'D':
            self.speed_angular = self.speed_angular - self.SPEED_DELTA
            print('Turn Left')
        if key == 'z' or key == 'Z':
            self.speed_linear = 0.0
            print('Set linear speed to zero')
        if key == 'c' or key == 'C':
            self.speed_angular = 0.0
            print('Set angular speed to zero')
        if key == 'x' or key == 'X':
            self.speed_angular = 0.0
            self.speed_linear = 0.0
            print('Stop the robot ')
        if self.speed_angular > 1:
            self.speed_angular = 1
        if self.speed_angular < -1:
            self.speed_angular = -1
        if self.speed_linear > 1:
            self.speed_linear = 1
        if self.speed_linear < -1:
            self.speed_linear = -1

        return self.speed_linear, self.speed_angular


class StampedMsgRegister():
    #define a global variable to initialize the tmsg_previous message
    global tmsg_previous
    tmsg_previous = None

    def replace_and_compute_delay(self, tmsg):
        global tmsg_previous
        
       #main loop to compute delay 
        if tmsg == None or tmsg_previous == None:
            time_delay = None
            msgaus = None
        else:
            time_delay = tmsg.header.stamp.to_sec(
            ) - tmsg_previous.header.stamp.to_sec()
            msgaus = tmsg_previous
        #the just analyzed message will become the next previous_message
        tmsg_previous = tmsg

        return time_delay, msgaus


def system_matrix(theta):
    """Returns a numpy array with the A(theta) matrix for a differential drive robot"""
    global k 
    global d
    #k = model_parameters()[0]
    #d = model_parameters()[1]

    A = 0.5 * k * (np.array([[cos(theta), cos(theta)],
                             [sin(theta), sin(theta)], [-1 / d, 1 / d]]))
    return A


def system_field(z, u):
    """Computes the field at a given state for the dynamical model"""
    return dot_z


def euler_step(z, u, stepSize):
    """Integrates the dynamical model for one time step using Euler's method"""

    theta = z[2][0]
    A = system_matrix(theta)
    dz = A.dot(u)
    zp = dz * stepSize
    #print(z, dz, zp, A, u)
    return zp


def closed_form_parameters(z, u):
    """Computes the values of several parameters of the closed form of the
    solution of the trajectory of the robot given the wheels velocities and the initial state of the robot"""
    global k 
    global d
    #k = model_parameters()[0]
    #d = model_parameters()[1]

    if abs(u[1][0] - u[0][0]) >= 0.0001:
        r = (d * (u[0][0] + u[1][0])) / (u[0][0] - u[1][0])
        omega = (k * (u[0][0] - u[1][0])) / (2 * d)
    else:
        c_x = z[0][0]
        c_y = z[1][0]
        c_theta = z[2][0]
        r = 0
        omega = 0
    c_x = z[0][0]-r*sin(z[2][0])
    c_y = z[1][0]+r*cos(z[2][0])
    c_theta = z[2][0]
    return r, omega, c_x, c_y, c_theta

def closed_form_step(z, u, T):
    """"""
    global k 
    global d
    #k = model_parameters()[0]
    #d = model_parameters()[1]
    cfp = closed_form_parameters(z, u)
    r = cfp[0]
    omega = cfp[1]
    c_x = cfp[2]
    c_y = cfp[3]
    c_theta = cfp[4]
    s_lw = u[0][0]
    s_rw = u[1][0]
 
    if abs(s_lw - s_rw) >= float(0.0001):
        zp = np.array([[r * sin(omega * T + c_theta) + c_x],
                       [- r * cos(omega * T + c_theta) + c_y],
                       [omega * T + c_theta]])
    else:
        zp = np.array([[(k / 2) * cos(c_theta) * (s_lw + s_rw) * T + c_x],
                       [(k / 2) * sin(c_theta) * (s_lw + s_rw) * T + c_y],
                       [c_theta]])
    
    return zp
