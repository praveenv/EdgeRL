import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

data = np.genfromtxt("colormap_data.csv",delimiter=",",skip_header=1)
data = np.array(data.astype(int))
print data

data1 = np.genfromtxt("colormap_truth.csv",delimiter=",",skip_header=1)
data1 = np.array(data1.astype(int))
print data1

# fig, (ax0, ax1) = plt.subplots(1, 2)
# plt.xticks([])
# plt.yticks([])
# c = ax0.pcolormesh(data1,cmap='Greys',edgecolors='black',linewidth=2)
# ax0.axis([0, 7, 0, 24])
# fig.colorbar(c, ax=ax0)
# ax0.set_title('Ground Truth')

# c = ax1.pcolormesh(data,cmap='Greys',edgecolors='black',linewidth=2)
# ax1.axis([0, 7, 0, 24])
# fig.colorbar(c, ax=ax1)
# plt.xticks([])
# plt.yticks([])
# ax1.set_title('Action Plan Chosen') 

fig, ax = plt.subplots(1,1)
plt.xticks([])
plt.yticks([])
c = ax.pcolormesh(data1,cmap='Blues',edgecolors='black',linewidth=2)
ax.axis([0, 7, 0, 24])
fig.colorbar(c, ax=ax)
# ax.set_title('Ground Truth')

plt.savefig('video_colormap_ground_truth.eps', format='eps')

fig, ax = plt.subplots(1,1)
plt.xticks([])
plt.yticks([])
c = ax.pcolormesh(data,cmap='Blues',edgecolors='black',linewidth=2)
ax.axis([0, 7, 0, 24])
# colorbar = ax.collections[0].colorbar
# colorbar.set_ticks([0,1,2])
# colorbar.set_ticklabels(['0', '1','2'])
# ax.set_title('Action Plan Chosen')
plt.savefig('video_colormap_action_plan.eps', format='eps')

plt.show()