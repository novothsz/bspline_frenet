import numpy as np
import matplotlib.pyplot as plt
import os

my_data = np.load('/Users/szilard/Dropbox/Sztaki/code/code_examples/bspline_static/src/trajs.npy')
# states = my_data['states']
# allstates = my_data['allstates']
y = my_data['y']
z = my_data['z']
q0 = my_data['q0']
q1 = my_data['q1']
time = my_data['time']
inp = my_data['input']
# allstates[0,3,155:] *= -1
# allstates[0,6,155:] *= -1
# plt.plot(time[0,:], allstates[0,1,:], time[0,:], allstates[0,2,:], time[0,:], allstates[0,6,:],
#          time[0,:], allstates[0,3,:], time[0,:], allstates[0,7,:])
# plt.legend(['y', 'z', 'q0', 'q1', 'roll (Euler)'])
# plt.show()

# time = time[0, :]
# y = allstates[0, 1, :]
# z = allstates[0, 2, :]
# q0 = allstates[0, 6, :]
# q1 = allstates[0, 3, :]


# with open(os.path.dirname(os.path.abspath(__file__)) + "/../files/logs/trajs.npy", 'wb') as out_file:
#     np.savez(out_file, time=time, y=y, z=z, q0=q0, q1=q1)

# plt.close('all')
idx_len = len(q0)
idx_new = np.linspace(0, idx_len-1, 100, endpoint=True)
idx_new = np.floor(idx_new)
downsampled_q0 = [q0[int(i)] for i in idx_new]

plt.plot(q0)
# plt.plot(downsampled_q0)
plt.plot(q1)

plt.figure()
plt.plot(inp[0, :])
plt.plot(inp[1, :])
plt.plot(inp[2, :])
plt.plot(inp[3, :])


for k in my_data.files:
    print(k)