# -*- coding: utf-8 -*-
# Copyright 2019 Martin Huenermund

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging

from geo_coords_converter import __version__

__author__ = "Martin Huenermund"
__copyright__ = "Martin Huenermund"
__license__ = "mit"

_logger = logging.getLogger(__name__)

import numpy as np
from geo_coords_converter import coords
from geo_coords_converter.utils import transform3d, rotation3d_z, translation3d

def estimate_rotation(pts1, pts2):
    """
    Compute optimal rotation between point correspondences.
    
    :param      pts1:   The points 1
    :param      pts2:   The points 2
    :type       pts1:   numpy.ndarray with shape(N,2)
    :type       pts2:   numpy.ndarray with shape(N,2)
    
    :returns:   Rotation between point correspondences
    :rtype:     number
    """

    assert(pts1.shape == pts2.shape)
    angles = np.array([
        estimate_rotation_pivot_median(pts1, pts2, k)
        for k in range(coords_gps.shape[0])
    ])
    return np.unwrap(angles).mean()

def estimate_rotation_pivot_mean(pts1, pts2, pivot=0):
    """
    Compute optimal rotation between point correspondences by fixing one point correspondence in place.
    
    :param      pts1:   The points 1
    :param      pts2:   The points 2
    :param      pivot:  Index of the pivot element (fixed point correspondence)
    :type       pts1:   numpy.ndarray with shape(N,2)
    :type       pts2:   numpy.ndarray with shape(N,2)
    :type       pivot:  number
    
    :returns:   Mean of estimated rotations
    :rtype:     number
    """

    angles = estimate_rotation_pivot(pts1, pts2, pivot)
    return np.unwrap(angles).mean()

def estimate_rotation_pivot_median(pts1, pts2, pivot=0):
    """
    Compute optimal rotation between point correspondences by fixing one point correspondence in place.
    
    :param      pts1:   The points 1
    :param      pts2:   The points 2
    :param      pivot:  Index of the pivot element (fixed point correspondence)
    :type       pts1:   numpy.ndarray with shape(N,2)
    :type       pts2:   numpy.ndarray with shape(N,2)
    :type       pivot:  number
    
    :returns:   Median of estimated rotations
    :rtype:     number
    """
    angles = estimate_rotation_pivot(pts1, pts2, pivot)
    return np.median(np.unwrap(angles))

def estimate_rotation_pivot(pts1, pts2, pivot=0):
    """
    Compute rotations between point correspondences by fixing one point correspondence in place.
    
    :param      pts1:   The points 1
    :param      pts2:   The points 2
    :param      pivot:  Index of the pivot element (fixed point correspondence)
    :type       pts1:   numpy.ndarray with shape(N,2)
    :type       pts2:   numpy.ndarray with shape(N,2)
    :type       pivot:  number
    
    :returns:   Estimated rotations for each point correspondence except pivot element
    :rtype:     numpy.ndarray with shape(N-1,)
    """
    # Compute optimal rotation by fixing one point correspondence in place 
    assert(pts1.shape == pts2.shape)
    assert(0 <= pivot < pts1.shape[0])
    assert(pts1.shape[0] > 1)
    idcs = np.arange(pts1.shape[0])
    # compute direction differences for all other points
    sel = (idcs == pivot)
    pts1_ = (pts1 - pts1[pivot])[~sel]
    pts2_ = (pts2 - pts2[pivot])[~sel]
    angles1 = np.arctan2(pts1_[:,1], pts1_[:,0])
    angles2 = np.arctan2(pts2_[:,1], pts2_[:,0])
    return angles1-angles2


def plot_pivot(pts1,pts2,pivot):
    angle = estimate_rotation_pivot_mean(pts1,pts2,pivot)
    print(angle)
    pts1_ = rot(pts1-pts1[pivot], -angle)
    pts2_ = rot(pts2-pts2[pivot], angle)
    
    plt.figure()
    plt.title("rotate pts1 to fit pts2")
    plt.plot(
        pts1_[:,0],
        pts1_[:,1],
        "o",
        label="pts1"
    )
    plt.plot(
        pts2[:,0]-pts2[pivot,0],
        pts2[:,1]-pts2[pivot,1],
        "o",
        label="pts2"
    )
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    
    plt.figure()
    plt.title("rotate pts2 to fit pts1")
    plt.plot(
        pts1[:,0]-pts1[pivot,0],
        pts1[:,1]-pts1[pivot,1],
        "o",
        label="pts1"
    )
    plt.plot(
        pts2_[:,0],
        pts2_[:,1],
        "o",
        label="pts2"
    )
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

def tf_world_geo(origin, lat2m, lon2m, world_angle, world_offset):
    """
    Transformation matrix transforming from geo coordinates to metric world coordinates.
    
    :param      origin:        The geo origin
    :param      lat2m:         The latitude to meter conversion factor
    :param      lon2m:         The longitude to meter conversion factor
    :param      world_angle:   The world angle
    :param      world_offset:  The world offset
    :type       origin:        coords.GeodeticCoordinate
    :type       lat2m:         number
    :type       lon2m:         number
    :type       world_angle:   number [radian]
    :type       world_offset:  number [meter]
    
    :returns:   Transformation matrix
    :rtype:     np.ndarray with shape [4,4]
    """
    rotation = rotation3d_z(world_angle)
    translation = translation3d(*world_offset)
    return translation @ rotation @ tf_meter_geo(origin, lat2m, lon2m)

def tf_map_geo(origin, lat2m, lon2m, world_angle, world_offset, map_origin_in_world):
    """
    Transformation matrix transforming from geo coordinates to metric map coordinates.
    
    :param      origin:               The geo origin
    :param      lat2m:                The latitude to meter conversion factor
    :param      lon2m:                The longitude to meter conversion factor
    :param      world_angle:          The world angle
    :param      world_offset:         The world offset
    :param      map_origin_in_world:  The map origin in the world
    :type       origin:               coords.GeodeticCoordinate
    :type       lat2m:                number
    :type       lon2m:                number
    :type       world_angle:          number
    :type       world_offset:         np.ndarray with shape(3,)
    :type       map_origin_in_world:  np.ndarray with shape(3,)
    
    :returns:   Transformation matrix
    :rtype:     np.ndarray with shape [4,4]
    """
    rotation = rotation3d_z(world_angle)
    translation = translation3d(*(world_offset-map_origin_in_world))
    return translation @ rotation @ tf_meter_geo(origin, lat2m, lon2m)

def tf_geo_world(origin, lat2m, lon2m, world_angle, world_offset):
    """
    Transformation matrix transforming from metric world coordinates to geo
    coordinates.
    
    :param      origin:        The geo origin
    :param      lat2m:         The latitude to meter conversion factor
    :param      lon2m:         The longitude to meter conversion factor
    :param      world_angle:   The world angle
    :param      world_offset:  The world offset
    :type       origin:        coords.GeodeticCoordinate
    :type       lat2m:         number
    :type       lon2m:         number
    :type       world_angle:   number
    :type       world_offset:  np.ndarray with shape(3,)
    
    :returns:   Transformation matrix
    :rtype:     np.ndarray with shape [4,4]
    """
    rotation = rotation3d_z(world_angle).T
    translation = -translation3d(*world_offset)
    return tf_geo_meter(origin, lat2m, lon2m) @ rotation @ translation

def tf_geo_map(origin, lat2m, lon2m, world_angle, world_offset, map_origin_in_world):
    """
    Transformation matrix transforming from metric map coordinates to geo
    coordinates.
    
    :param      origin:               The geo origin
    :param      lat2m:                The latitude to meter conversion factor
    :param      lon2m:                The longitude to meter conversion factor
    :param      world_angle:          The world angle
    :param      world_offset:         The world offset
    :param      map_origin_in_world:  The map origin in the world
    :type       origin:               coords.GeodeticCoordinate
    :type       lat2m:                number
    :type       lon2m:                number
    :type       world_angle:          number
    :type       world_offset:         np.ndarray with shape(3,)
    :type       map_origin_in_world:  np.ndarray with shape(3,)
    
    :returns:   Transformation matrix
    :rtype:     np.ndarray with shape [4,4]
    """
    rotation = rotation3d_z(world_angle).T
    translation = -translation3d(*(world_offset-map_origin_in_world))
    return translation @ rotation @ tf_meter_geo(origin, lat2m, lon2m)

def transform_geo_to_world(origin, lat2m, lon2m, world_angle, world_offset, coords_geo_rad):
    """
    { function_description }
    
    :param      origin:          The origin
    :param      lat2m:           The lat 2 m
    :param      lon2m:           The lon 2 m
    :param      world_angle:     The world angle
    :param      world_offset:    The world offset
    :param      coords_geo_rad:  The coordinates geo radians
    :type       origin:          { type_description }
    :type       lat2m:           { type_description }
    :type       lon2m:           { type_description }
    :type       world_angle:     { type_description }
    :type       world_offset:    { type_description }
    :type       coords_geo_rad:  { type_description }
    
    :returns:   { description_of_the_return_value }
    :rtype:     { return_type_description }
    """
    return transform3d(
        tf_world_geo(origin, lat2m, lon2m, world_angle, world_offset),
        geo_rad)

def transform_world_to_geo(origin, lat2m, lon2m, world_angle, world_offset, coords_world_meter):
    return transform3d(
        tf_geo_world(origin, lat2m, lon2m, world_angle, world_offset),
        coords_world_meter)

def transform_geo_to_map(origin, lat2m, lon2m, world_angle, world_offset, map_origin_in_world, coords_geo_rad):
    return transform3d(
        tf_map_geo(origin, lat2m, lon2m, world_angle, world_offset, map_origin_in_world),
        geo_rad)

def transform_map_to_geo(origin, lat2m, lon2m, world_angle, world_offset, map_origin_in_world, coords_map_meter):
    return transform3d(
        tf_geo_map(origin, lat2m, lon2m, world_angle, world_offset, map_origin_in_world),
        coords_map_meter)


def approximate_geo_to_world(coords_geo_rad, coords_world):
    geo_origin = coords.GeodeticCoordinate(*np.mean(coords_geo_rad, axis=0))
    lat2m, lon2m = approximate_geo_to_meter_NED(geo_origin, coords_geo_rad)
    coords_ned = transform_geo_to_meter(geo_origin, lat2m, lon2m, coords_geo_rad)
    world_angle = estimate_rotation(coords_world[:,:2], coords_ned[:,:2])
    
    coords_world_parallel = transform3d(rotation3d_z(world_angle), coords_ned)
    world_offset = np.mean(coords_world - coords_world_parallel,axis=0)
    
    return geo_origin, lat2m, lon2m, world_angle, world_offset

