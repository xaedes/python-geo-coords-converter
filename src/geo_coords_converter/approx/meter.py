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
from geo_coords_converter.utils import transform3d

def approximate_geo_to_meter(geo_origin, geo_coords_arr_rad, carthesian_coords_arr_meter):
    """
    Compute best latitude/longitude to meter factors for linear approximation of
    geo to carthesian from point correspondences.
    
    :param      geo_origin:                   The geo coordinate where to center
                                              the approximation
    :param      geo_coords_arr_rad:           The geo coordinates in radians
    :param      carthesian_coords_arr_meter:  The carthesian coordinates in
                                              meter
    :type       geo_origin:                   coords.GeodeticCoordinate
    :type       geo_coords_arr_rad:           numpy.ndarray with shape(N,K), K
                                              >= 2
    :type       carthesian_coords_arr_meter:  numpy.ndarray with shape(N,K), K
                                              >= 2
    
    :returns:   Factors to get from geo coordinate to meter: latitude_to_meter,
                longitude_to_meter
    :rtype:     number
    """

    dlatlon = np.array(geo_coords_arr_rad[:,:2]) - (geo_origin.latitude, geo_origin.longitude)
    dxy = carthesian_coords_arr_meter[:,:2]
    latitude_to_meter, longitude_to_meter = (dxy / dlatlon).mean(axis=0)
    return latitude_to_meter, longitude_to_meter

def approximate_geo_to_meter_(geo_origin, geo_coords_arr_rad, geo_to_carth_func):
    """
    Compute best latitude/longitude to meter factors for linear approximation of
    geo to carthesian from geo coordinates and geo coordinate conversion
    function.
    
    :param      geo_origin:          The geo coordinate where to center the
                                     approximation
    :param      geo_coords_arr_rad:  The geo coordinates in radians
    :param      geo_to_carth_func:   Function converting geo coordinate to
                                     carthesian coordinate. E.g. `lambda coord,
                                     origin: coord.toENU(origin)`
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       geo_coords_arr_rad:  numpy.ndarray with shape(N,3), K >= 2
    :type       geo_to_carth_func:   (coords.GeodeticCoordinate,coords.GeodeticCoordinate)
                                     -> coords.CarthesianCoordinate
    
    :returns:   Factors to get from geo coordinate to meter: latitude_to_meter,
                longitude_to_meter
    :rtype:     number
    """
    km_to_m = 1000
    carthesian_coords_meter = np.array([
        geo_to_carth_func(coords.GeodeticCoordinate(*coord), geo_origin).position * km_to_m 
        for coord in geo_coords_arr_rad
    ])
    
    return approximate_geo_to_meter(geo_origin, geo_coords_arr_rad, carthesian_coords_meter)

def approximate_geo_to_meter_SEZ(geo_origin, geo_coords_arr_rad):
    """
    Compute best latitude/longitude to meter factors for linear approximation of
    geo to carthesian from geo coordinates and corresponding SEZ coordinates.
    
    :param      geo_origin:          The geo coordinate where to center the
                                     approximation
    :param      geo_coords_arr_rad:  The geo coordinates in radians
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       geo_coords_arr_rad:  numpy.ndarray with shape(N,K), K >= 2
    
    :returns:   Factors to get from geo coordinate to meter: latitude_to_meter,
                longitude_to_meter
    :rtype:     number
    """

    return approximate_geo_to_meter_(
        geo_origin, geo_coords_arr_rad, 
        lambda coord,origin: coord.toSEZ(origin))

def approximate_geo_to_meter_NED(geo_origin, geo_coords_arr_rad):
    """
    Compute best latitude/longitude to meter factors for linear approximation of
    geo to carthesian from geo coordinates and corresponding NED coordinates.
    
    :param      geo_origin:          The geo origin
    :param      geo_coords_arr_rad:  The geo coordinates in radians
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       geo_coords_arr_rad:  numpy.ndarray with shape(N,K), K >= 2
    
    :returns:   Factors to get from geo coordinate to meter: latitude_to_meter,
                longitude_to_meter
    :rtype:     number
    """
    return approximate_geo_to_meter_(
        geo_origin, geo_coords_arr_rad, 
        lambda coord,origin: coord.toNED(origin))

def approximate_geo_to_meter_ENU(geo_origin, geo_coords_arr_rad):
    """
    Compute best latitude/longitude to meter factors for linear approximation of
    geo to carthesian from geo coordinates and corresponding ENU coordinates.
    
    :param      geo_origin:          The geo origin
    :param      geo_coords_arr_rad:  The geo coordinates radians
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       geo_coords_arr_rad:  numpy.ndarray with shape(N,K), K >= 2
    
    :returns:   Factors to get from geo coordinate to meter: latitude_to_meter,
                longitude_to_meter
    :rtype:     number
    """
    return approximate_geo_to_meter_(
        geo_origin, geo_coords_arr_rad, 
        lambda coord,origin: coord.toENU(origin))



def tf_meter_geo(geo_origin, latitude_to_meter, longitude_to_meter):
    """
    Transformation matrix transforming from geo coordinates to meter.
    
    :param      geo_origin:          The geo origin
    :param      latitude_to_meter:   The latitude to meter conversion factor
    :param      longitude_to_meter:  The longitude to meter conversion factor
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       latitude_to_meter:   number
    :type       longitude_to_meter:  number
    
    :returns:   Transformation matrix
    :rtype:     np.ndarray with shape [4,4]
    """
    return np.array([
        [latitude_to_meter,0,0,-geo_origin.latitude*latitude_to_meter],
        [0,longitude_to_meter,0,-geo_origin.longitude*longitude_to_meter],
        [0,0,1,0],
        [0,0,0,1],
    ])

def tf_geo_meter(geo_origin, latitude_to_meter, longitude_to_meter):
    """
    Transformation matrix transforming from meter to geo coordinates.
    
    :param      geo_origin:          The geo origin
    :param      latitude_to_meter:   The latitude to meter conversion factor
    :param      longitude_to_meter:  The longitude to meter conversion factor
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       latitude_to_meter:   number
    :type       longitude_to_meter:  number
    
    :returns:   Transformation matrix
    :rtype:     np.ndarray with shape [4,4]
    """
    return np.array([
        [1/latitude_to_meter,0,0,geo_origin.latitude],
        [0,1/longitude_to_meter,0,geo_origin.longitude],
        [0,0,1,0],
        [0,0,0,1],
    ])

def transform_geo_to_meter(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad):
    """
    Transforms geo coordinates to metric carthesian coordinate system
    
    :param      geo_origin:          The geo origin
    :param      latitude_to_meter:   The latitude to meter conversion factor
    :param      longitude_to_meter:  The longitude to meter conversion factor
    :param      geo_coords_arr_rad:  The geo coordinates in radians
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       latitude_to_meter:   number
    :type       longitude_to_meter:  number
    :type       geo_coords_arr_rad:  numpy.ndarray with shape(N,3)
    
    :returns:   The coordinates in meter
    :rtype:     numpy.ndarray with shape(N,3)
    """
    return transform3d(tf_meter_geo(geo_origin, latitude_to_meter, longitude_to_meter), geo_coords_arr_rad)

def transform_meter_to_geo(geo_origin, latitude_to_meter, longitude_to_meter, meter_arr):
    """
    Transforms metric carthesian coordinates to geodetic coordinate system
    
    :param      geo_origin:          The geo origin
    :param      latitude_to_meter:   The latitude to meter conversion factor
    :param      longitude_to_meter:  The longitude to meter conversion factor
    :param      meter_arr:           The coordinates in meter
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       latitude_to_meter:   number
    :type       longitude_to_meter:  number
    :type       meter_arr:           numpy.ndarray with shape(N,3)
    
    :returns:   The geo coordinates in radians
    :rtype:     numpy.ndarray with shape(N,3)
    """
    return transform3d(tf_geo_meter(geo_origin, latitude_to_meter, longitude_to_meter), geo_coords_arr_rad)

def error_of_geo_to_meter_approx(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad, carthesian_coords_meter):
    """
    Computes error of linear transformation approximation in comparision with
    point correspondences.
    
    :param      geo_origin:               The geo origin
    :param      latitude_to_meter:        The latitude to meter conversion
                                          factor
    :param      longitude_to_meter:       The longitude to meter conversion
                                          factor
    :param      geo_coords_arr_rad:       The geo coordinates in radians
    :param      carthesian_coords_meter:  The carthesian coordinates in meter
    :type       geo_origin:               coords.GeodeticCoordinate
    :type       latitude_to_meter:        number
    :type       longitude_to_meter:       number
    :type       geo_coords_arr_rad:       numpy.ndarray with shape(N,3)
    :type       carthesian_coords_meter:  numpy.ndarray with shape(N,3)
    
    :returns:   errors of linear transformation in comparision with point
                correspondences
    :rtype:     numpy.ndarray with shape(N,3)
    """
    meter = transform_geo_to_meter(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad)
    error = carthesian_coords_meter - meter
    return error

def error_of_geo_to_meter_approx_(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad, geo_to_carth_func):
    """
    Computes error of linear transformation approximation with implicit point
    correspondences.
    
    :param      geo_origin:          The geo origin
    :param      latitude_to_meter:   The latitude to meter conversion factor
    :param      longitude_to_meter:  The longitude to meter conversion factor
    :param      geo_coords_arr_rad:  The geo coordinates in radians
    :param      geo_to_carth_func:   Function converting geo coordinate to
                                     carthesian coordinate. E.g. `lambda coord,
                                     origin: coord.toENU(origin)`
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       latitude_to_meter:   number
    :type       longitude_to_meter:  number
    :type       geo_coords_arr_rad:  numpy.ndarray with shape(N,3)
    :type       geo_to_carth_func:   (coords.GeodeticCoordinate,coords.GeodeticCoordinate)
                                     -> coords.CarthesianCoordinate
    
    :returns:   errors of linear transformation
    :rtype:     numpy.ndarray with shape(N,3)
    """
    km_to_m = 1000
    carthesian_coords_meter = np.array([
        geo_to_carth_func(coords.GeodeticCoordinate(*coord), geo_origin).position*km_to_m 
        for coord in geo_coords_arr_rad
    ])
    return error_of_geo_to_meter_approx(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad, carthesian_coords_meter)

def error_of_geo_to_meter_approx_SEZ(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad):
    """
    Computes error of linear transformation approximation with point
    correspondences to SEZ coordinates.
    
    :param      geo_origin:          The geo origin
    :param      latitude_to_meter:   The latitude to meter
    :param      longitude_to_meter:  The longitude to meter
    :param      geo_coords_arr_rad:  The geo coordinates inradians
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       latitude_to_meter:   number
    :type       longitude_to_meter:  number
    :type       geo_coords_arr_rad:  numpy.ndarray with shape(N,3)
    
    :returns:   errors of linear transformation
    :rtype:     numpy.ndarray with shape(N,3)
    """
    return error_of_geo_to_meter_approx_(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad, 
        lambda coord,origin: coord.toSEZ(origin))

def error_of_geo_to_meter_approx_NED(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad):
    """
    Computes error of linear transformation approximation with point
    correspondences to NED coordinates.
    
    :param      geo_origin:          The geo origin
    :param      latitude_to_meter:   The latitude to meter
    :param      longitude_to_meter:  The longitude to meter
    :param      geo_coords_arr_rad:  The geo coordinates inradians
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       latitude_to_meter:   number
    :type       longitude_to_meter:  number
    :type       geo_coords_arr_rad:  numpy.ndarray with shape(N,3)
    
    :returns:   errors of linear transformation
    :rtype:     numpy.ndarray with shape(N,3)
    """
    return error_of_geo_to_meter_approx_(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad, 
        lambda coord,origin: coord.toNED(origin))

def error_of_geo_to_meter_approx_ENU(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad):
    """
    Computes error of linear transformation approximation with point correspondences to ENU coordinates.
    
    :param      geo_origin:          The geo origin
    :param      latitude_to_meter:   The latitude to meter
    :param      longitude_to_meter:  The longitude to meter
    :param      geo_coords_arr_rad:  The geo coordinates inradians
    :type       geo_origin:          coords.GeodeticCoordinate
    :type       latitude_to_meter:   number
    :type       longitude_to_meter:  number
    :type       geo_coords_arr_rad:  numpy.ndarray with shape(N,3)
                                       
    :returns:   errors of linear transformation
    :rtype:     numpy.ndarray with shape(N,3)
    """
    return error_of_geo_to_meter_approx_(geo_origin, latitude_to_meter, longitude_to_meter, geo_coords_arr_rad, 
        lambda coord,origin: coord.toENU(origin))

def error_stats(error_arr, axis=0):
    """
    Computes statistics for array of errors.
    
    :param      error_arr:  The error arr
    :param      axis:       The axis
    :type       error_arr:  numpy.ndarray
    :type       axis:       number
    
    :returns:   dict containing stats of error array
    :rtype:     dict
    """
    return {
        "mean":   np.mean(error_arr, axis=axis),
        "median": np.median(error_arr, axis=axis),
        "std":    np.std(error_arr, axis=axis),
        "min":    np.min(error_arr, axis=axis),
        "max":    np.max(error_arr, axis=axis),
        "minabs": np.min(np.abs(error_arr), axis=axis),
        "maxabs": np.max(np.abs(error_arr), axis=axis)
    }


