
import helper as hp
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
import georinex as gr   # "pip install georinex" from command prompt launched from Anaconda to add to environment
import ephemeris as eph
import calc_rx_pos as crp

rinex_obs_filename = 'ohdt0320.21o'
rinex_nav_filename = 'ohdt0320.21n'
TRUE_POS_ECEF = np.array([497796.507, -4884306.583, 4058066.619])  # XYZ [meters]

SEC_PER_DAY = 86400

# load RINEX files
obs = gr.load(rinex_obs_filename)
nav = gr.load(rinex_nav_filename)

# conversion from RINEX observation calendar date and time to TOW (seconds)
time_obs_tow = [hp.datetime2tow(x.isoformat()) for x in obs.time.data]

# initial unbiased estimate of receiver position and receiver clock correction
pos_rx = np.array([0.0, 0.0, 0.0])
dt_rx = 0.0

# initialize storage
num_epochs = len(time_obs_tow)
pos_rx_array = np.empty((num_epochs, 3))
pos_rx_array[:] = np.nan
dt_rx_array = np.empty(num_epochs)
dt_rx_array[:] = np.nan
v_array = np.empty((num_epochs, 32))
v_array[:] = np.nan
enu_array = np.empty((num_epochs, 3))
enu_array[:] = np.nan

for epoch_idx in range(num_epochs):
    ephem_list = []
    t_rx_hat = time_obs_tow[epoch_idx]
    current_obs = obs.sel(time=obs.time.data[epoch_idx]).dropna(dim='sv',how='all')
    prn_array = current_obs.sv.data
    rho_hat_array = current_obs['C1'].data

    prn_idx = 0
    while prn_idx < len(prn_array):
        nav_prn = nav.sel(sv=prn_array[prn_idx]).dropna(dim='time', how='all')
        eph_idx = hp.computeClosest(time_obs_tow[epoch_idx], nav_prn.Toe.data)
        ephem_list.append(eph.Ephemeris(nav_prn, eph_idx))
        if ephem_list[prn_idx].health != 0.0:   # unhealthy!
            prn_array = np.delete(prn_array, prn_idx)
            rho_hat_array = np.delete(rho_hat_array, prn_idx)
            ephem_list.pop(prn_idx)
        else:
            prn_idx += 1

    if len(prn_array) < 4:  # too few satellites to solve for 3-D position and clock correction
        continue  # skip to next epoch

    # this is the function to write
    [pos_rx, dt_rx, v] = crp.calc_rx_pos(prn_array, rho_hat_array, t_rx_hat, ephem_list, pos_rx, dt_rx)

    # save results
    pos_rx_array[epoch_idx, :] = pos_rx
    dt_rx_array[epoch_idx] = dt_rx
    v_idx = 0
    for v_prn in prn_array:
        v_prn_int = int(v_prn[1:]) - 1
        v_array[epoch_idx, v_prn_int] = v[v_idx]
        v_idx += 1


# print true position and 2nd epoch position result and range residuals
np.set_printoptions(precision=2)
print('\nTrue Position = ')
print(TRUE_POS_ECEF)
print('pos_rx_array[1, :] = ')
print(pos_rx_array[1, :])
print('\nv_array[1, :] = ')
print(v_array[1, :])


# ENU conversion
for ii in range(num_epochs):
    enu_array[ii, :] = hp.ecef2enu(pos_rx_array[ii, :], TRUE_POS_ECEF)


# plotting code here

sec_of_day = np.mod(time_obs_tow, SEC_PER_DAY)

# figure 1: sat viz vs time (seconds of day)
# plt.figure()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for ii in range(32):
    # non-NaN residuals represent utilized (healthy) measurements
    plt.plot(sec_of_day, v_array[:, ii] / v_array[:, ii] * (ii + 1), '.')
ax.set_xlim(0, 3600)
ax.set_ylim(0, 33)

ax.xaxis.set_major_locator(MultipleLocator(500))  # major (labeled tick) every 500
ax.yaxis.set_major_locator(MultipleLocator(5))    # major (labeled tick) every 5
ax.xaxis.set_minor_locator(AutoMinorLocator(5))   # 5 minor per major
ax.yaxis.set_minor_locator(AutoMinorLocator(5))   # 5 minor per major

ax.grid(which='major', alpha=0.5)
ax.grid(which='minor', alpha=0.2)

plt.xlabel('Time of Day [sec]')
plt.ylabel('PRN [ND]')
plt.title('PRNs Utilized in Solution vs Time')

# figure 2: ENU errors vs time
plt.figure()
plt.grid()
plt.plot(sec_of_day, enu_array[:, 0], label='East')
plt.plot(sec_of_day, enu_array[:, 1], label='North')
plt.plot(sec_of_day, enu_array[:, 2], label='Up')
plt.legend()
plt.xlim(0, 3600)
plt.xlabel('Time of Day [sec]')
plt.ylabel('ENU Errors [m]')
plt.title('ENU Errors vs Time')

# figure 3: East and North Errors Cross Plot
plt.figure()
plt.grid()
plt.plot(enu_array[:, 0], enu_array[:, 1], '.')
plt.axis('equal')
plt.xlabel('East Error [m]')
plt.ylabel('North Error [m]')
plt.title('North vs East Error')

# figure 4:  Receiver Clock Error vs Time
plt.figure()
plt.grid()
plt.plot(sec_of_day, dt_rx_array*1e9)
plt.xlim(0, 3600)
plt.xlabel('Time of Day [sec]')
plt.ylabel('Receiver Clock Error [nsec]')
plt.title('Receiver Clock Error vs Time')

# figure 5: Range Residuals vs Time
plt.figure()
plt.grid()
for ii in range(32):
    plt.plot(sec_of_day, v_array[:, ii], '.')
plt.xlim(0, 3600)
plt.xlabel('Time of Day [sec]')
plt.ylabel('Range Residual [m]')
plt.title('Range Residuals vs Time')

plt.show()

