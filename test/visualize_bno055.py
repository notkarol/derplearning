#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import Adafruit_BNO055.BNO055
import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt

bno = Adafruit_BNO055.BNO055.BNO055(busnum=0)
bno.begin()
calibration_status = bno.get_calibration_status()

print("BNO055 status: %s self_test: %s error: %s" % bno.get_system_status() )
print("BNO055 sw: %s bl: %s accel: %s mag: %s gyro: %s" % bno.get_revision())

funcs = {'quaternion': bno.read_quaternion,
         'euler': bno.read_euler,
         'gravity': bno.read_gravity,
         'magneto': bno.read_magnetometer,
         'gyro': bno.read_gyroscope,
         'accel': bno.read_linear_acceleration,
         }
stores = {name: [] for name in funcs}
timestamps = []
names = [x for x in sorted(funcs)]

# Pre-read some values
for i in range(100):
    for name in names:
        vals = funcs[name]()

# Start reading and storing
time_start = time()
t = time()
for i in range(100 * 60 * 10):
    # Make sure we wait as close to 100hz as possible
    t_elapsed = time() - t
    while t_elapsed < 0.00999:
        t_elapsed = time() - t
    t = time()
    timestamps.append(t)
    
    # Get all the datapoints
    for name in names:
        vals = funcs[name]()
        stores[name].append(vals)
time_end = time()

# Plot statistics and save them
print("Elapsed: %7.3f" % (time_end - time_start))
timestamps = np.array(timestamps, dtype=np.float64) - timestamps[0]
for name in names:
    store = np.array(stores[name], dtype=np.float32)
    means = " ".join(["%9.5f" % x for x in store.mean(axis=0)])
    stds = " ".join(["%9.5f" % x for x in store.std(axis=0)])
    print("%-10s %40s %40s" % (name, means, stds))

    for col in range(store.shape[1]):
        plt.plot(timestamps, store[:, col], ',')
    plt.title(name)
    plt.savefig('%s.png' % name, dpi=400, bbox_inches='tight')
    plt.close()

