plt.close('all')
plt.figure()
for veh in group.vehicles:
    plt.plot(veh.DvX.y[37], veh.DvX.y[75], 'ro')
    print(veh.DvX.y[-3], veh.DvX.y[-2], veh.DvX.y[-1])
plt.show()


plt.close('all')
plt.figure()
for veh in group.vehicles:
    plt.plot(veh.PvX.z_i[37], veh.PvX.z_i[75], 'ro')
    print(veh.PvX.xf[0], veh.PvX.xf[1], veh.PvX.xf[2], veh.PvX.z_i[-1])
plt.show()