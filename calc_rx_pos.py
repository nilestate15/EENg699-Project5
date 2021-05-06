
import numpy as np
import calc_sv_pos as csp


C = 299792458.0  # [m/s]


def calc_rx_pos(prn_array, rho_hat_array, t_rx_hat, ephem_list, pos_rx, dt_rx):
    """
    Satellite position and satellite clock correction calculation based on IS-GPS-200L (Table 20-IV and pp. 95-96).
    The satellite position is for a particular GPS time, expressed in the ECEF coordinate frame
    valid at the time the signal was received by a GPS receiver.

    :param prn_array:        array of PRNs utilized in this epoch (e.g. 'G01', 'G11')
    :param rho_hat_array:    array of pseudoranges for this epoch [meters]
    :param t_rx_hat:         approximate receiver time [sec of week]
    :param ephem_list:       list of ephemeris objects for all PRNs [see ephemeris.py]
    :param pos_rx:           approximate receiver position [x y z in meters]
    :param dt_rx:            approximate receiver clock correction [sec]

    :return pos_rx:          iterated receiver position [x y z in meters]
    :return dt_rx:           iterated receiver clock correction [sec]
    :return v:               residuals [meters]
    """

    t_sv_hat = np.zeros(len(prn_array))
    dt_sv_prime = np.zeros(len(prn_array))
    t_sv_prime = np.zeros(len(prn_array))
    pos_sv = np.zeros((len(prn_array), 3))
    dt_sv = np.zeros(len(prn_array))
    rho_prime = np.zeros(len(prn_array))
    rho_tilde = np.zeros(len(prn_array))
    geo_range = np.zeros(len(prn_array))
    drho = np.zeros(len(prn_array))
    H = np.zeros((len(prn_array), 4))

    norm_dx = 100  # large initial value

    for ii in range(len(prn_array)):
        t_sv_hat[ii] = t_rx_hat - rho_hat_array[ii]/C
        dt_sv_prime[ii] = ephem_list[ii].a_f0 + ephem_list[ii].a_f1 * (t_sv_hat[ii] - ephem_list[ii].t_oc) \
            + ephem_list[ii].a_f2 * (t_sv_hat[ii] - ephem_list[ii].t_oc)**2
        t_sv_prime[ii] = t_sv_hat[ii] - dt_sv_prime[ii]
        pos_sv[ii, :], dt_sv[ii] = csp.calc_sv_pos(ephem_list[ii], t_sv_prime[ii], pos_rx)
        dt_sv[ii] = dt_sv[ii] - ephem_list[ii].T_GD
        rho_prime[ii] = rho_hat_array[ii] + C*dt_sv[ii]

    x_vect = np.array([pos_rx[0], pos_rx[1], pos_rx[2], C*dt_rx])

    while norm_dx > 10:

        for ii in range(len(prn_array)):
            geo_range[ii] = np.sqrt((pos_sv[ii, 0] - x_vect[0])**2 + (pos_sv[ii, 1] - x_vect[1])**2
                                    + (pos_sv[ii, 2] - x_vect[2])**2)
            rho_tilde[ii] = geo_range[ii] + x_vect[3]
            drho[ii] = rho_prime[ii] - rho_tilde[ii]
            H[ii, :] = [-(pos_sv[ii, 0] - x_vect[0])/geo_range[ii], -(pos_sv[ii, 1] - x_vect[1])/geo_range[ii],
                        -(pos_sv[ii, 2] - x_vect[2])/geo_range[ii], 1]

        # "@" vs np.dot vs np.matmult
        dx = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(H), H)), np.transpose(H)), drho)
        x_vect = x_vect + dx
        norm_dx = np.linalg.norm(dx)

    pos_rx = x_vect[0:3]
    dt_rx = x_vect[3]/C
    v = drho - (H @ dx)

    return pos_rx, dt_rx, v
