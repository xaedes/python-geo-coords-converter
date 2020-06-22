# -*- coding: utf-8 -*-
# Copyright 2019 Martin Huenermund

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import sys
import logging

from geo_coords_converter import __version__

__author__ = "Martin Huenermund"
__copyright__ = "Martin Huenermund"
__license__ = "mit"

_logger = logging.getLogger(__name__)

import numpy as np

def transform3d(mat44, points):
    return np.dot(points,mat44[:3,:3].T)+mat44[:3,3]

def translation3d(x,y,z):
    return np.array([
        [1,0,0,x],
        [0,1,0,y],
        [0,0,1,z],
        [0,0,0,1],
    ])

def rotation3d_x(angle):
    cs,sn = math.cos(angle), math.sin(angle)
    return np.array([
        [1,0,0,0],
        [0,cs,-sn,0],
        [0,+sn,cs,0],
        [0,0,0,1],
    ])
def rotation3d_y(angle):
    cs,sn = math.cos(angle), math.sin(angle)
    return np.array([
        [cs,0,+sn,0],
        [0,1,0,0],
        [-sn,0,cs,0],
        [0,0,0,1],
    ])
def rotation3d_z(angle):
    cs,sn = math.cos(angle), math.sin(angle)
    return np.array([
        [cs,-sn,0,0],
        [+sn,cs,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ])

def rotation3d_rpy(roll, pitch, yaw):
    # euler-1-2-3 scheme 
    # transforms from body to world
    return rotation3d_z(yaw) @ rotation3d_y(pitch) @ rotation3d_x(roll)

def fill_altitude(geo_arr, default_altitude=0):
    return np.array([
        [x,y,default_altitude]
        for x,y in geo_arr
    ])