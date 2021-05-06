
import math
import numpy as np


def ecef2enu(ECEF, origin_ECEF):

    # origin_ECEF = L2E.llh2ecef(origin_LLH)  # origin (at zero height)
    origin_LLH = ecef2llh(origin_ECEF)
    s_lon0 = np.sin(origin_LLH[1])
    c_lon0 = np.cos(origin_LLH[1])
    s_lat0 = np.sin(origin_LLH[0])
    c_lat0 = np.cos(origin_LLH[0])

    # Direction cosine matrix from delECEF to Local Level
    C_G_E = np.array([[-s_lon0, c_lon0, 0],
                      [-s_lat0 * c_lon0, -s_lat0 * s_lon0, c_lat0],
                      [c_lat0 * c_lon0, c_lat0 * s_lon0, s_lat0]])


    delECEF = ECEF - origin_ECEF
    p_E2L = np.dot(C_G_E, delECEF)  # using dot product to return row vector

    return p_E2L


def ecef2llh(pe):
    """
    Converts Earth-Centered, Earth-Fixed coordinates into the equivalent
    WGS-84 representation.

    :param pe 3x1 numpy array of ECEF coordinates in x, y, z order. The x-axis
         passes through the intersection of the equator and the Greenwich
         meridian, the z axis through the ITRF North pole, and the y-axis
         completes the right-handed frame. (meters)

    :return llh 3x1 numpy array of geodetic latitude, longitude and altitude coordinates
         with respect to the WGS-84 ellipsoid. (radians, radians, meters)

    """
    # WGS-84 Constants
    a = 6378137.0          # Semi-major radius (m)
    e2 = 0.00669437999013  # Eccentricity squared (.)

    # Calculate measured position in meridian plane
    pm = np.array([np.sqrt(pow(pe[0], 2) + pow(pe[1], 2)), pe[2]])
    h0 = 0
    phi0 = np.arctan2(pm[1], pm[0])

    dp = np.array([a, a])

    # Loop 20 times (20 was picked mostly randomly to be "good enough")
    for ii in range(20):

        s_lat = np.sin(phi0)
        c_lat = np.cos(phi0)
        s2lat = s_lat ** 2

        n_den = 1 - e2 * s2lat

        n = a / np.sqrt(n_den)

        # Calculate initial position in meridian plane
        p0 = np.array([(n + h0) * c_lat, (n * (1 - e2) + h0) * s_lat])

        # Calculate residual
        dp = pm - p0  # meters

        # Calculate inverse Jacobian (transformation from residual to lat and alt)
        k1 = 1 - e2 * s2lat
        k2 = np.sqrt(k1)

        a11 = s_lat * (e2 * a * c_lat * c_lat / k1 / k2 - a / k2 - h0)
        a12 = c_lat
        a21 = c_lat * (a * (1 - e2) / k2 + h0 + a * e2 * (1 - e2) * s2lat / k1 / k2)
        a22 = s_lat
        a_det = a11 * a22 - a21 * a12
        a_inv1 = np.array([a22 / a_det, -a12 / a_det])
        a_inv2 = np.array([-a21 / a_det, a11 / a_det])

        dha = a_inv1[0] * dp[0] + a_inv1[1] * dp[1]
        dhb = a_inv2[0] * dp[0] + a_inv2[1] * dp[1]

        phi0 = phi0 + dha
        h0 = h0 + dhb

    lam = np.arctan2(pe[1], pe[0])
    return np.array([phi0, lam, h0])


def computeClosest(number, data_struct):
    """
    Find nearest value from list of values

    :param number: value to get closest to (time, etc)
    :param data_struct: list of values
    :return closest_index: index of closest value in list
    """

    closest_index = 0
    diff_value = 10000000  # really big number
    for ii in range(len(data_struct)):
        if diff_value > abs(data_struct[ii] - number):
            diff_value = abs(data_struct[ii] - number)
            closest_index = ii
    return closest_index


def datetime2tow(datetime_string):
    """
    Converts GeoRinex time.data[idx].astype('str') string (e.g. '2020-12-15T00:00:00.000000000')
    to GPS time of week

    :param datetime_string: single time.data value from a GeoRinex object converted to a string
    :return tow: time of week [sec]
    """

    SEC_PER_WEEK = 604800

    year = int(datetime_string[2:4])
    month = int(datetime_string[5:7])
    day = int(datetime_string[8:10])

    hour = int(datetime_string[11:13])
    minute = int(datetime_string[14:16])
    second = int(datetime_string[17:19])

    # convert two digit year to four digits (assume range of 1980-2079)
    if 80 <= year <= 99:
        year = 1900 + year

    if 0 <= year <= 79:
        year = 2000 + year

    # calculate "m" and "y" terms used below from the calendar month
    if month <= 2:
        y = year - 1
        m = month + 12

    if month > 2:
        y = year
        m = month

    # compute Julian date corresponding to given calendar date
    JD = math.floor(365.25 * y) + math.floor(30.6001 * (m+1)) + \
         day + (hour + minute / 60 + second / 3600) / 24 + 1720981.5

    gps_week = math.floor((JD - 2444244.5) / 7)

    tow = round(((((JD - 2444244.5) / 7) - gps_week) * SEC_PER_WEEK) / 0.5) * 0.5

    return tow




