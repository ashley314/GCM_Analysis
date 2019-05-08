## Basic example to plot a variable from model output ##
## Ashley Dicks ##

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# set path for location of data
path = "/Users/ash/Documents/GitHub/GCM_Analysis/Data/"

# open data files
file1 = nc.Dataset(path+"E.2000_C5.TS.nc")
for v in file1.variables:
    print(v)
ts_mod1 = file1.variables['TS']
print(ts_mod1)

# topography information
topo = nc.Dataset(path+"USGS-gtopo30_1.9x2.5_remap_c050602.nc")
lat_topo = topo.variables['lat'][:]
lon_topo = topo.variables['lon'][:]

# quick and diry plot (Q&D)
var1d = ts_mod1[1,:,:]
# mask for only land - might use later
#var1d_masked = np.ma.MaskedArray(var1d, topo.variables['LANDFRAC'][:] == 0. )

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
cax1 = ax1.pcolormesh(lon_topo, lat_topo, var1d)
cbar1 = plt.colorbar(cax1)
ax1.set_title('Q&D Surface Temperature Plot (K)')
ax1.axis([0, 360, -90, 90])
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.contour(lon_topo, lat_topo, topo.variables['LANDFRAC'][:], [0.5,0.51], colors='k')
plt.show()

# close files
file1.close()
topo.close()

