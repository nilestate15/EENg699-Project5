
import math
import numpy as np


def calc_sv_pos(eph, tow, rcvr_ecef):
    """
    Satellite position and satellite clock correction calculation based on IS-GPS-200L (Table 20-IV and pp. 95-96).
    The satellite position is for a particular GPS time, expressed in the ECEF coordinate frame
    valid at the time the signal was received by a GPS receiver.

    :param eph:              ephemeris object (from georinex) for a single satellite and time epoch
    :param tow:              time of transmission [GPS week sec]
    :param rcvr_ecef:        1x3 numpy array of ECEF receiver position [x y z in meters]

    :return sv_ecef:         1x3 numpy array of ECEF satellite position [x y z in meters] at time of transmission,
                               expressed in ECEF frame at the time of reception
    :return sv_clock_error:  satellite clock error [sec] at time of transmission
    """

    # verify element counts on inputs
    rcvr_length = len(rcvr_ecef)

    if rcvr_length != 3:
        print("invalid rcvr_ecef input into calc_sv_pos function. Exiting...")
        return None

    # verify input is of type numpy.ndarray
    if type(rcvr_ecef) is not np.ndarray:
        rcvr_ecef = np.array(rcvr_ecef)

    # ephemeris calculations based on Table 20-IV, pp. 102-103, IS-GPS-200L

    # constant values
    C = 299792458.0  # speed of light [m / sec]
    MU = 3.986005e14  # Earth gravitational constant [m^3 / sec^2]
    OMEGA_DOT_E = 7.2921151467e-5  # Earth rotation rate [rad / sec]

    # semi-major axis [m]
    A = eph.sqrtA ** 2

    # corrected mean motion [rad / sec]
    n_0 = math.sqrt(MU / (A ** 3))
    n = n_0 + eph.delta_n

    # time since ephemeris reference epoch, corrected for week rollovers [sec]
    t_k = tow - eph.t_oe
    if t_k > 302400:
        t_k = t_k - 604800
    elif t_k < -302400:
        t_k = t_k + 604800

    # mean anomaly [rad]
    M_k = eph.M_0 + (n * t_k)

    # eccentric anomaly (solve Kepler's equation by RCVR-3A iteration) [rad]
    E_k = M_k + eph.e * math.sin(M_k)
    E_k = (eph.e * (math.sin(E_k) - E_k * math.cos(E_k)) + M_k) / (1 - eph.e * math.cos(E_k))
    E_k = (eph.e * (math.sin(E_k) - E_k * math.cos(E_k)) + M_k) / (1 - eph.e * math.cos(E_k))

    # true anomaly [rad]
    sin_nu = (math.sqrt(1 - eph.e**2) * math.sin(E_k)) / (1 - eph.e * math.cos(E_k))
    cos_nu = (math.cos(E_k) - eph.e) / (1 - eph.e * math.cos(E_k))
    nu_k = math.atan2(sin_nu, cos_nu)

    # uncorrected argument of latitude [rad]
    Phi_k = nu_k + eph.omega

    # second harmonic corrections
    del_u_k = eph.C_us * math.sin(2 * Phi_k) + eph.C_uc * math.cos(2 * Phi_k)
    del_r_k = eph.C_rs * math.sin(2 * Phi_k) + eph.C_rc * math.cos(2 * Phi_k)
    del_i_k = eph.C_is * math.sin(2 * Phi_k) + eph.C_ic * math.cos(2 * Phi_k)

    # corrected argument of latitude [rad]
    u_k = Phi_k + del_u_k

    # corrected radius [m]
    r_k = A * (1 - eph.e * math.cos(E_k)) + del_r_k

    # corrected inclination [rad]
    i_k = eph.i_0 + del_i_k + eph.i_dot * t_k

    # position in orbital plane [m]
    x_k_p = r_k * math.cos(u_k)
    y_k_p = r_k * math.sin(u_k)

    # corrected longitude of ascending node
    Omega_k = eph.Omega_0 + (eph.Omega_dot - OMEGA_DOT_E) * t_k - OMEGA_DOT_E * eph.t_oe

    # position at transmit time relative to the ECEF frame at TRANSMIT time [m]
    x_k = x_k_p * math.cos(Omega_k) - y_k_p*math.cos(i_k) * math.sin(Omega_k)
    y_k = x_k_p * math.sin(Omega_k) + y_k_p * math.cos(i_k) * math.cos(Omega_k)
    z_k = y_k_p * math.sin(i_k)

    # approximate propagation time and ECEF frame rotation angle
    sat_pos_tx = np.array([x_k, y_k, z_k])
    t_prop = np.linalg.norm(sat_pos_tx - rcvr_ecef) / C
    gamma = OMEGA_DOT_E * t_prop

    C_t_r = np.array([[math.cos(gamma), math.sin(gamma), 0],
                     [-math.sin(gamma), math.cos(gamma), 0],
                     [0, 0, 1]])

    # position at transmit time relative to the ECEF frame at RECEIVE time [m]
    sv_ecef = np.dot(C_t_r, sat_pos_tx)

    # time since clock reference epoch corrected for week rollovers (p. 96 in IS-GPS-200L)
    delta_t_c = tow - eph.t_oc
    if delta_t_c > 302400:
        delta_t_c = delta_t_c - 604800
    if delta_t_c < -302400:
        delta_t_c = delta_t_c + 604800

    # satellite clock error calculation (p. 95, eq. 2, IS-GPS-200L)
    delta_t_r = (-2 * math.sqrt(MU)) / (C ** 2) * eph.e * eph.sqrtA * math.sin(E_k)

    sv_clock_error = eph.a_f0 + eph.a_f1 * delta_t_c + eph.a_f2 * (delta_t_c ** 2) + delta_t_r

    return sv_ecef, sv_clock_error
