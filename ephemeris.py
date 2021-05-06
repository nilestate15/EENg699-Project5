
import helper


# define ephemeris class (Table 20-II (p. 100) in IS-GPS-200L)
class Ephemeris:

    def __init__(self, nav_values, idx):

        # Set all the appropriate ephemeris params from the parsed RINEX input at the current index
        self.M_0 = nav_values.M0.data[idx]
        self.delta_n = nav_values.DeltaN.data[idx]
        self.e = nav_values.Eccentricity.data[idx]
        self.sqrtA = nav_values.sqrtA.data[idx]
        self.Omega_0 = nav_values.Omega0.data[idx]
        self.i_0 = nav_values.Io.data[idx]
        self.omega = nav_values.omega.data[idx]
        self.Omega_dot = nav_values.OmegaDot.data[idx]
        self.i_dot = nav_values.IDOT.data[idx]
        self.C_uc = nav_values.Cuc.data[idx]
        self.C_us = nav_values.Cus.data[idx]
        self.C_rc = nav_values.Crc.data[idx]
        self.C_rs = nav_values.Crs.data[idx]
        self.C_ic = nav_values.Cic.data[idx]
        self.C_is = nav_values.Cis.data[idx]
        self.t_oe = nav_values.Toe.data[idx]
        self.IODE = nav_values.IODE.data[idx]
        self.t_oc = helper.datetime2tow(nav_values.time.data[idx].astype('str'))
        self.a_f0 = nav_values.SVclockBias.data[idx]
        self.a_f1 = nav_values.SVclockDrift.data[idx]
        self.a_f2 = nav_values.SVclockDriftRate.data[idx]
        self.health = nav_values.health.data[idx]
        self.T_GD = nav_values.TGD.data[idx]
