# -*- coding: utf-8 -*-
# This file contains miscellaneous functions required for coordinate transformation.
# Functions were originally written for Matlab as companion code for "Fundamentals of Astrodynamics 
# and Applications" by David Vallado (2007). (w) 719-573-2600, email dvallado@agi.com
# Ported to C++ by Grady Hillhouse with some modifications, July 2015.
# Ported to Python by Martin Huenermund, April 2019.

# https://celestrak.com/publications/AIAA/2006-6753/faq.php
# Are there any Licenses required to use the SGP4 code?
# There is no license associated with the code and you may use it for any purpose—personal or commercial—as you wish. We ask only that you include citations in your documentation and source code to show the source of the code and provide links to the main page, to facilitate communications regarding any questions on the theory or source code. 

# https://celestrak.com/publications/AIAA/2006-6753/
# http://www.celestrak.com/software/vallado-sw.php
# 
import argparse
import sys
import logging

from geo_coords_converter import __version__

__author__ = "Martin Huenermund"
__copyright__ = "Martin Huenermund"
__license__ = "mit"

_logger = logging.getLogger(__name__)


import math
import numpy as np

mag = np.linalg.norm    
sgn = np.sign

def teme2ecef(rteme, vteme, jdut1):
    """
    This function transforms a vector from a true equator mean equinox (TEME)
    frame to an earth-centered, earth-fixed (ECEF) frame.
    Author: David Vallado, 2007
    Ported to C++ by Grady Hillhouse with some modifications, July 2015.
    Ported to Python by Martin Huenermund, April 2019.
    
    INPUTS          DESCRIPTION                     RANGE/UNITS
    rteme           Position vector (TEME)          km
    vteme           Velocity vector (TEME)          km/s
    jdut1           Julian date                     days
    
    OUTPUTS         DESCRIPTION                     RANGE/UNITS
    recef           Position vector (ECEF)          km  
    vecef           Velocity vector (ECEF)          km/s
    """
    # Get Greenwich mean sidereal time
    gmst = gstime(jdut1)
    cs, sn = math.cos(gmst), math.sin(gmst)
    
    # st is the pef - tod matrix
    st = np.array([
        [cs, -sn, 0],
        [sn, cs, 0],
        [0,0,1]
    ])
    
    # Get pseudo earth fixed position vector by multiplying the inverse pef-tod matrix by rteme
    rpef = st.T.dot(rteme)
    
    # Get polar motion vector
    pm = polarm(jdut)
    
    # ECEF postion vector is the inverse of the polar motion vector multiplied by rpef
    recef = pm.T.dot(rpef)
    
    # Earth's angular rotation vector (omega)
    # Note: I don't have a good source for LOD. Historically it has been on the order of 2 ms so I'm just using that as a constant. The effect is very small.
    omegaearth = np.array([0,0,7.29211514670698e-05 * (1.0 - 0.002/86400.0)])
    
    # Pseudo Earth Fixed velocity vector is st'*vteme - omegaearth X rpef
    vpef = st.T.dot(vteme)
    
    # ECEF velocty vector is the inverse of the polar motion vector multiplied by vpef
    vecef = pm.T.dot(vpef)
    
    return recef, vecef

def polarm(jdut1):
    """
    This function calulates the transformation matrix that accounts for polar
    motion. Polar motion coordinates are estimated using IERS Bulletin
    rather than directly input for simplicity.
    Author: David Vallado, 2007
    Ported to C++ by Grady Hillhouse with some modifications, July 2015.
    Ported to Python by Martin Huenermund, April 2019.
    
    INPUTS          DESCRIPTION                     RANGE/UNITS
    jdut1           Julian date                     days
    
    OUTPUTS         DESCRIPTION
    pm              Transformation matrix for ECEF - PEF
    """
    # Predict polar motion coefficients using IERS Bulletin - A (Vol. XXVIII No. 030)
    MJD = jdut1 - 2400000.5
    A = 2 * math.pi * (MJD - 57226) / 365.25
    C = 2 * math.pi * (MJD - 57226) / 435
    cosA = math.cos(A)
    sinA = math.sin(A)
    cosC = math.cos(C)
    sinC = math.sin(C)
    
    xp = (0.1033 + 0.0494*cosA + 0.0482*sinA + 0.0297*cosC + 0.0307*sinC) * 4.84813681e-6
    yp = (0.3498 + 0.0441*cosA - 0.0393*sinA + 0.0307*cosC - 0.0297*sinC) * 4.84813681e-6
    
    # pm = (rot1(yp)*rot2(xp))
    # with rot1(a) = 1 ,     0  ,     0  ;
    #                0 , cos(a) , sin(a) ;
    #                0 , -sin(a) , cos(a) ;
    #  and rot2(a) = cos(a) , 0 , -sin(a) ;
    #                    0  , 1 ,      0  ;
    #                sin(a) , 0 ,  cos(a) ;
    #
    # transponiert:
    # pm_T =  cos(xp) , sin(xp)sin(yp) ,  cos(yp)sin(xp)  ;
    #        0        ,        cos(yp) , -sin(yp)         ;
    #        -sin(xp) , cos(xp)sin(yp) ,  cos(yp)*cos(xp) ;    
    
    # row major
    csX, snX = math.cos(xp), math.sin(xp)
    csY, snY = math.cos(yp), math.sin(yp)
    pm = np.array([
        [csX, 0, -snX],
        [snX*snY, csY, csX*snY],
        [snX*csY, -snY, csX*csY]
    ])
    return pm    



def ijk2ll(r):
    """
    This function calulates the latitude, longitude and altitude
    given the ECEF position matrix.
    Author: David Vallado, 2007
    Ported to C++ by Grady Hillhouse with some modifications, July 2015.
    Ported to Python by Martin Huenermund, April 2019.
    
    INPUTS          DESCRIPTION                     RANGE/UNITS
    r               Position matrix (ECEF)          km
    
    OUTPUTS         DESCRIPTION
    latlongh        Latitude, longitude, and altitude (rad, rad, and km)
    """
    pi2 = math.pi*2
    small = 0.00000001 # small value for tolerances
    re = 6378.137 # radius of earth in km
    eesqrd = 0.006694385000  #eccentricity of earth sqrd
    
    # http://www.epsg.org/Portals/0/373-07-2.pdf
    # section 2.2.1 Geographic/Geocentric conversions
    magr = mag(r)
    temp = mag(r[:2])
    
    if abs(temp) < small:
        rtasc = sgn(r[2])*math.pi*0.5
    else:
        rtasc = math.atan2(r[1], r[0])
    
    latlongh = np.zeros(3)
    latlongh[1] = rtasc
    
    if abs(latlongh[1]) >= math.pi:
        if latlongh[1] < 0:
            latlongh[1] += pi2
        else:
            latlongh[1] -= pi2
            
    latlongh[0] = math.asin(r[2] / magr)
    
    # Iterate to find geodetic latitude
    i = 1
    olddelta = latlongh[0] + 10.0
    sintemp = 0
    c = 0
    
    while (abs(olddelta - latlongh[0]) >= small) and (i < 10):
        olddelta = latlongh[0]
        sintemp = math.sin(latlongh[0])
        c = re / math.sqrt(1 - eesqrd*sintemp*sintemp)
        latlongh[0] = math.atan((r[2]+c*eesqrd*sintemp)/temp)
        i+=1
    
    if math.pi*0.5 - abs(latlongh[0]) > math.pi / 180:
        latlongh[2] = (temp / math.cos(latlongh[0])) - c
    else:
        latlongh[2] = r[2] / math.sin(latlongh[0]) - c*(1.0 - eesqrd)
    
    return latlongh

def site(latgd, lon, alt):
    """
    This function finds the position and velocity vectors for a site. The
    outputs are in the ECEF coordinate system. Note that the velocity vector
    is zero because the coordinate system rotates with the earth.
    Author: David Vallado, 2007
    Ported to C++ by Grady Hillhouse with some modifications, July 2015.
    Ported to Python by Martin Huenermund, April 2019.

    INPUTS          DESCRIPTION                     RANGE/UNITS
    latgd           Site geodetic latitude          -PI/2 to PI/2 in radians
    lon             Longitude                       -2PI to 2PI in radians
    alt             Site altitude                   km

    OUTPUTS         DESCRIPTION
    rs              Site position vector            km
    vs              Site velocity vector            km/s
    """
    re = 6378.137 # radius of earth in km
    eesqrd = 0.006694385000  #eccentricity of earth sqrd
    
    # Find rdel and rk components of site vector
    sinlat = math.sin(latgd)
    cearth = re / math.sqrt( 1.0 - (eesqrd*sinlat*sinlat) )
    rdel = (cearth + alt) * math.cos(latgd)
    rk = ((1.0 - eesqrd) * cearth + alt ) * sinlat
    
    # Find site position vector (ECEF)
    rs = np.array([
        rdel * math.cos( lon ),
        rdel * math.sin( lon ),
        rk
    ])

    # Velocity of site is zero because the coordinate system is rotating with the earth
    vs = np.array([
        0, 0, 0
    ])
    
    return rs, vs

def rv2azel(ro, vo, latgd, lon, alt, jdut1):
    """
    This function calculates the range, elevation, and azimuth (and their rates)
    from the TEME vectors output by the SGP4 function.
    Author: David Vallado, 2007
    Ported to C++ by Grady Hillhouse with some modifications, July 2015.
    Ported to Python by Martin Huenermund, April 2019.

    INPUTS          DESCRIPTION                     RANGE/UNITS
    ro              Sat. position vector (TEME)     km
    vo              Sat. velocity vector (TEME)     km/s
    latgd           Site geodetic latitude          -PI/2 to PI/2 in radians
    lon             Site longitude                  -2PI to 2PI in radians
    alt             Site altitude                   km
    jdut1           Julian date                     days

    OUTPUTS         DESCRIPTION
    razel           Range, azimuth, and elevation matrix
    razelrates      Range rate, azimuth rate, and elevation rate matrix
    """
    halfpi = math.pi * 0.5
    small = 0.00000001 # small value for tolerances
    
    # Get site vector in ECEF coordinate system
    rs, vs = site(latgd, lon, alt)
    
    # Convert TEME vectors to ECEF coordinate system
    recef, vecef = teme2ecef(ro, vo, jdut1)
    
    # Find ECEF range vectors
    rhoecef = recef - rs
    drhoecef = vecef[:]
    rho = mag(rhoecef) # range in km
    
    # Convert to SEZ (topocentric horizon coordinate system)
    tempvec = rot3(rhoecef, lon)
    rhosez = rot2(tempvec, halfpi-latgd)
    
    tempvec = rot3(drhoecef, lon)
    drhosez = rot2(tempvec, halfpi-latgd)
    
    # Calculate azimuth, and elevation
    temp = mag(rhosez[:2])
    if temp < small:
        el = sgn(rhosez[2]) * halfpi
        az = math.atan2(drhosez[1], -drhosez[0])
    else:
        magrhosez = mag(rhosez)
        el = math.asin(rhosez[2] / magrhosez)
        az = math.atan2(rhosez[1]/temp, -rhosez[0]/temp)
        
    # Calculate rates for range, azimuth, and elevation
    drho = np.dot(rhosez,drhosez) / rho
    
    if abs(temp*temp) > small:
        del_ = (drhosez[2] - drho*sin(el)) / temp
    else:
        del_ = 0.0
        
    # move values to output vectors
    razel = np.array([
        rho,  # Range (km)
        az,   # Azimuth (radians)
        el    # Elevation (radians)
    ])
    
    razelrates = np.array([
        drho,   # Range (km/s)
        daz,    # Azimuth (radians/s)
        del_    # Elevation (radians/s)
    ])
    return razel, razelrates
        
def rot3(invec, xval):
    """
    Rotate vector around z-axis in clockwise direction.
    
    I.e. apply transformation matrix:
       cs  sn  0
       -sn cs  0
        0   0  1
    
    Author: David Vallado, 2007
    Ported to C++ by Grady Hillhouse with some modifications, July 2015.
    Ported to Python by Martin Huenermund, April 2019.
    
    INPUTS          DESCRIPTION                     RANGE/UNITS
    invec           Vector
    xval            Rotation angle                  radians
    
    OUTPUTS         DESCRIPTION
    rotated         rotated vector
    """
    temp = invec[2]
    cs, sn = math.cos(xval), math.sin(xval)
    return np.array([
        cs*invec[0] + sn*invec[1],
        cs*invec[1] - sn*invec[0],
        invec[2]
    ])
    
def rot2(invec, xval):
    """
    Rotate vector around y-axis in clockwise direction.
    
    I.e. apply transformation matrix:
       cs  0 -sn
       0   1   0
       sn  0  cs
    
    Author: David Vallado, 2007
    Ported to C++ by Grady Hillhouse with some modifications, July 2015.
    Ported to Python by Martin Huenermund, April 2019.
    
    INPUTS          DESCRIPTION                     RANGE/UNITS
    invec           Vector
    xval            Rotation angle                  radians
    
    OUTPUTS         DESCRIPTION
    rotated         rotated vector
    """
    temp = invec[2]
    cs, sn = math.cos(xval), math.sin(xval)
    return np.array([
        cs*invec[0] - sn*invec[2],
        invec[1],
        cs*invec[2] + s*invec[0]
    ])

def julianDateFromUnix(unixSecs):
    return unixSecs / 86400.0  + 2440587.5

#https://github.com/xaedes/GNSS-Shadowing/blob/master/sgp4_vallado/src/sgp4unit.cpp
# gstime

def gstime(jdut1):
    """
    This function finds the greenwich sidereal time.

    Author: David Vallado, 2001
    Ported to Python by Martin Huenermund, April 2019.

    INPUTS          DESCRIPTION                     RANGE/UNITS
    jdut1           julian date in ut1              days from 4713 bc


    OUTPUTS         DESCRIPTION                     RANGE/UNITS
    gstime          greenwich sidereal time         radians [0 to 2pi] 

    References    :      Vallado       2004, 191, eq 3-45
    """
    twopi = 2*math.pi
    d2r = math.pi/180
    # julian centuries from the jan 1, 2000 12 h epoch (ut1)
    tut1 = jdut1 - 2451545.0 / 36525.0
    
    # sec
    temp = -6.2e-6* tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + (876600.0*3600 + 8640184.812866) * tut1 + 67310.54841
    # 360/8640 = 1/240, to deg, to rad
    temp = math.fmod(temp * d2r / 240.0, twopi) 
    
    # check quadrants 
    if temp < 0:
        temp += twopi
    
    return temp 
