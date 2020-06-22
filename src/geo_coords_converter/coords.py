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

import time
import math
import numpy as np
from geo_coords_converter import vallado

def jdnow():
    return vallado.julianDateFromUnix(time.time())

class GeodeticCoordinate:
    def __init__(self, latitude=0, longitude=0, altitude=0, julianDate=0):
        self.latitude = latitude     # [-pi/2..pi/2] in radians
        self.longitude = longitude   # [-2pi..2pi] in radians 
        self.altitude = altitude     # km
        self.julianDate = julianDate # days

    def copy(self):
        return GeodeticCoordinate(self.latitude, self.longitude, self.altitude, self.julianDate)
        
    def toECEF(self):
        pos, vel = vallado.site(self.latitude, self.longitude, self.altitude)
        return ECEF(pos, vel, self.julianDate)
    
    def toSEZ(self, origin):
        return self.toECEF().toSEZ(origin)
    
    def toENU(self, origin):
        return self.toECEF().toENU(origin)
    
    def toNED(self, origin):
        return self.toECEF().toNED(origin)
    
    def toSpherical(self, origin):
        return self.toECEF().toSEZ(origin).toSpherical()
    
    def __repr__(self):
        return "GeodeticCoordinate(latitude=%.8f, longitude=%.8f, altitude=%f, julianDate=%.3f)" % \
                (self.latitude, self.longitude, self.altitude, self.julianDate)    

class SphericalCoordinate:
    def __init__(self, azimuth=0, elevation=0, range_=0, julianDate=0, origin=None):
        self.azimuth = azimuth        # [-2pi..2pi in radians] angle between north and this, positive in east direction
        self.elevation = elevation    # [0..pi/2 in radians]   angle between ground plane and this, positive upwards
        self.range_ = range_          # in km
        self.julianDate = julianDate  # days
        self.origin = origin
    
    def copy(self):
        pass
    def toENU(self):
        csaz, snaz = math.cos(self.azimuth), math.sin(self.azimuth)
        csel, snel = math.cos(self.elevation), math.sin(self.elevation)
        r = csel * self.range_
        
        return ENU(np.array([
                snaz * r,
                csaz * r,
                snel * self.range_
            ]), np.zeros(3), # ignore velocity
            self.julianDate, self.origin)
        
    def toSEZ(self):
        return self.toENU().toSEZ()
    
    def toNED(self):
        return self.toENU().toNED()
    
    def toECEF(self):
        return self.toENU().toECEF()
    
    def toGeodetic(self):
        return self.toENU().toGeodetic()
    
    def __repr__(self):
        return "SphericalCoordinate(azimuth=%f, elevation=%f, range=%f, julianDate=%.3f, origin=%s)" % \
                (self.azimuth, self.elevation, self.range_, self.julianDate, str(self.origin))    

class CarthesianCoordinate:
    def __init__(self, position=(0,0,0), velocity=(0,0,0), julianDate=0):
        self.position = np.array(position) # km
        self.velocity = np.array(velocity) # km/s
        self.julianDate = julianDate       # days
        
    def __add__(self, other):
        return type(self)(
            self.position+other.position, 
            self.velocity+other.velocity, 
            self.julianDate)
    def __sub__(self, other):
        return type(self)(
            self.position-other.position, 
            self.velocity-other.velocity, 
            self.julianDate)
    
    def __repr__(self):
        return "CarthesianCoordinate(position=%s, velocity=%s, julianDate=%.3f)" % \
                (str(self.position), str(self.velocity), self.julianDate)

class ECEF(CarthesianCoordinate):
    def __init__(self, position=(0,0,0), velocity=(0,0,0), julianDate=0):
        super().__init__(position, velocity, julianDate)
    
    def copy(self):
        return ECEF(self.position, self.velocity, self.julianDate)

    def toENU(self, origin):
        originECEF = origin.toECEF()
        rangeECEF = self.copy() - originECEF
        
        ENU_ECEF = ENU.ENU_ECEF(origin)
        return ENU(
            ENU_ECEF.dot(rangeECEF.position),
            ENU_ECEF.dot(rangeECEF.velocity),
            self.julianDate,
            origin
        )
        
    def toSEZ(self, origin):
        originECEF = origin.toECEF()
        rangeECEF = self.copy() - originECEF
        
        SEZ_ECEF = SEZ.SEZ_ECEF(origin)
        return SEZ(
            SEZ_ECEF.dot(rangeECEF.position),
            SEZ_ECEF.dot(rangeECEF.velocity),
            self.julianDate,
            origin
        )

    def toNED(self, origin):
        return self.toSEZ(origin).toNED()

    def toGeodetic(self):
        lat,lon,alt = vallado.ijk2ll(self.position)
        return GeodeticCoordinate(lat, lon, alt, self.julianDate)
    
    def toSpherical(self, origin):
        return self.toSEZ(origin).toSpherical()

    def __add__(self, other):
        return ECEF(
            self.position+other.position, 
            self.velocity+other.velocity, 
            self.julianDate)
    def __sub__(self, other):
        return ECEF(
            self.position-other.position, 
            self.velocity-other.velocity, 
            self.julianDate)
    
    def __repr__(self):
        return "ECEF(position=%s, velocity=%s, julianDate=%.3f)" % \
                (str(self.position), str(self.velocity), self.julianDate)    


class TEME(CarthesianCoordinate):
    def __init__(self, position=(0,0,0), velocity=(0,0,0), julianDate=0):
        super().__init__(position, velocity, julianDate)
    
    def copy(self):
        return TEME(self.position, self.velocity, self.julianDate)

    def toECEF(self):
        pos, vel = vallado.teme2ecef(self.position, self.velocity, self.julianDate)
        return ECEF(pos, vel, self.julianDate)
        
    def toSpherical(self, origin):
        razel, razelrates = vallado.rv2azel(
            self.position, self.velocity, 
            origin.latitude, origin.longitude, origin.altitude,
            self.julianDate
        )
        return SphericalCoordinate(razel[0], razel[1], razel[2], self.julianDate, origin)

    def __add__(self, other):
        return TEME(
            self.position+other.position, 
            self.velocity+other.velocity, 
            self.julianDate)
    def __sub__(self, other):
        return TEME(
            self.position-other.position, 
            self.velocity-other.velocity, 
            self.julianDate)
    
    def __repr__(self):
        return "TEME(position=%s, velocity=%s, julianDate=%.3f)" % \
                (str(self.position), str(self.velocity), self.julianDate)    

class ENU(CarthesianCoordinate):
    """East North Up"""
    def __init__(self, position=(0,0,0), velocity=(0,0,0), julianDate=0, origin=None):
        super().__init__(position, velocity, julianDate)
        self.origin = origin
    
    def copy(self):
        return ENU(self.position, self.velocity, self.julianDate)

    def toECEF(self):
        originECEF = self.origin.toECEF()
        ECEF_ENU = ENU.ECEF_ENU(origin)
        ecef = ECEF(
            ECEF_ENU.dot(self.position),
            ECEF_ENU.dot(self.velocity),
            self.julianDate
        )
        return ecef + originECEF
    
    def toSEZ(self):
        SEZ_ENU = np.array([
            [0,1,0],
            [-1,0,0],
            [0,0,1]
        ])
        return SEZ(
            SEZ_ENU.dot(self.position),
            SEZ_ENU.dot(self.velocity),
            self.julianDate,
            self.origin
        )
    
    def toNED(self):
        NED_ENU = np.array([
            [0,1,0],
            [1,0,0],
            [0,0,-1]
        ])
        return NED(
            NED_ENU.dot(self.position),
            NED_ENU.dot(self.velocity),
            self.julianDate,
            self.origin
        )
    
    def toSpherical(self):
        mag_xy = vallado.mag(self.position[:2])
        mag_xyz = vallado.mag(self.position[:3])
        north = self.position[1]
        east = self.position[0]
        zenith = self.position[2]
        azimuth = math.atan2(east / mag_xy, north / mag_xy)
        elevation = math.asin(zenith / mag_xyz)
        
        return SphericalCoordinate(azimuth, elevation, mag_xyz, self.julianDate, self.origin)
    
    def toGeodetic(self):
        return self.toECEF().toGeodetic()
    
    def ENU_ECEF(origin):
        """Transformation matrix to transform from ECEF to ENU"""
        cslat, snlat = math.cos(origin.latitude), math.sin(origin.latitude)
        cslon, snlon = math.cos(origin.longitude), math.sin(origin.longitude)
        return np.array([
            [-snlon, cslon, 0],
            [-cslon*snlat, -snlon*snlat, cslat],
            [cslon*cslat, snlon*cslat, snlat]
        ])
        
    def ECEF_ENU(origin):
        """Transformation matrix to transform from ENU to ECEF"""
        return ENU.ENU_ECEF(origin).T
    
    def __add__(self, other):
        return ENU(
            self.position+other.position, 
            self.velocity+other.velocity, 
            self.julianDate)
    def __sub__(self, other):
        return ENU(
            self.position-other.position, 
            self.velocity-other.velocity, 
            self.julianDate)

    def __repr__(self):
        return "ENU(position=%s, velocity=%s, julianDate=%.3f, origin=%s)" % \
                (str(self.position), str(self.velocity), self.julianDate, str(self.origin))    
    
class SEZ(CarthesianCoordinate):
    """South East Zenith"""
    def __init__(self, position=(0,0,0), velocity=(0,0,0), julianDate=0, origin=None):
        super().__init__(position, velocity, julianDate)
        self.origin = origin
    
    def copy(self):
        return SEZ(self.position, self.velocity, self.julianDate)

    def toECEF(self):
        originECEF = self.origin.toECEF()
        ECEF_SEZ = SEZ.ECEF_SEZ(origin)
        ecef = ECEF(
            ECEF_SEZ.dot(self.position),
            ECEF_SEZ.dot(self.velocity),
            self.julianDate
        )
        return ecef + originECEF
    
    def toENU(self):
        ENU_SEZ = np.array([
            [0,-1,0],
            [1,0,0],
            [0,0,1]
        ])
        return ENU(
            ENU_SEZ.dot(self.position),
            ENU_SEZ.dot(self.velocity),
            self.julianDate,
            self.origin
        )
    
    def toNED(self):
        NED_SEZ = np.array([
            [-1,0,0],
            [0,1,0],
            [0,0,-1]
        ])
        return NED(
            NED_SEZ.dot(self.position),
            NED_SEZ.dot(self.velocity),
            self.julianDate,
            self.origin
        )
        
    def toSpherical(self):
        mag_xy = mag(self.position[:2])
        mag_xyz = mag(self.position[:3])
        north = -self.position[0]
        east = self.position[1]
        zenith = self.position[2]
        azimuth = math.atan2(east / mag_xy, north / mag_xy)
        elevation = math.asin(zenith / mag_xyz)
        
        return SphericalCoordinate(azimuth, elevation, mag_xyz, self.julianDate, self.origin)
    
    def toGeodetic(self):
        return self.toECEF().toGeodetic()
    
    def SEZ_ECEF(origin):
        """Transformation matrix to transform from ECEF to SEZ"""
        cslat, snlat = math.cos(origin.latitude), math.sin(origin.latitude)
        cslon, snlon = math.cos(origin.longitude), math.sin(origin.longitude)
        return np.array([
            [cslon*snlat, snlat*snlon, -cslat],
            [-snlon, cslon, 0],
            [cslat*cslon, cslat*snlon, snlat]
        ])
    def ECEF_SEZ(origin):
        """Transformation matrix to transform from SEZ to ECEF"""
        return SEZ.SEZ_ECEF(origin).T
    
    def __add__(self, other):
        return SEZ(
            self.position+other.position, 
            self.velocity+other.velocity, 
            self.julianDate)
    def __sub__(self, other):
        return SEZ(
            self.position-other.position, 
            self.velocity-other.velocity, 
            self.julianDate)

    def __repr__(self):
        return "SEZ(position=%s, velocity=%s, julianDate=%.3f, origin=%s)" % \
                (str(self.position), str(self.velocity), self.julianDate, str(self.origin))    

class NED(CarthesianCoordinate):
    """North East Down"""
    def __init__(self, position=(0,0,0), velocity=(0,0,0), julianDate=0, origin=None):
        super().__init__(position, velocity, julianDate)
        self.origin = origin
    
    def copy(self):
        return SEZ(self.position, self.velocity, self.julianDate)

    def toECEF(self):
        originECEF = self.origin.toECEF()
        ECEF_NED = NED.ECEF_NED(origin)
        ecef = ECEF(
            ECEF_NED.dot(self.position),
            ECEF_NED.dot(self.velocity),
            self.julianDate
        )
        return ecef + originECEF
    
    def toENU(self):
        ENU_NED = np.array([
            [0,1,0],
            [1,0,0],
            [0,0,-1]
        ])
        return ENU(
            ENU_NED.dot(self.position),
            ENU_NED.dot(self.velocity),
            self.julianDate,
            self.origin
        )
    
    def toSEZ(self):
        SEZ_NED = np.array([
            [-1,0,0],
            [0,1,0],
            [0,0,-1]
        ])
        return SEZ(
            SEZ_NED.dot(self.position),
            SEZ_NED.dot(self.velocity),
            self.julianDate,
            self.origin
        )
    
    def toSpherical(self):
        return self.toENU().toSpherical()
    
    def toGeodetic(self):
        return self.toECEF().toGeodetic()
    

    
    def __add__(self, other):
        return NED(
            self.position+other.position, 
            self.velocity+other.velocity, 
            self.julianDate)
    def __sub__(self, other):
        return NED(
            self.position-other.position, 
            self.velocity-other.velocity, 
            self.julianDate)

    def __repr__(self):
        return "NED(position=%s, velocity=%s, julianDate=%.3f, origin=%s)" % \
                (str(self.position), str(self.velocity), self.julianDate, str(self.origin))    

    
EarthCenteredEarthFixed = ECEF
TrueEquatorMeanEquinox = TEME
SouthEastZenith = SEZ
EastNorthUp = ENU
NorthEastDown = NED
