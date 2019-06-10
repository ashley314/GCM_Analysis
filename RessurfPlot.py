## RESSURF Line Plots ##
## Ashley Dicks ##

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# set path for location of data
path = "/Users/ash/Documents/Gradschool/Research/Miocene/Data/"
f = nc.Dataset(path+"E.2000_C5.00-49.global_av_ressurf3.nc")
f2 = nc.Dataset(path+"E.2000_C5.00-49.global_av_ressurf_LND.nc")
f3 = nc.Dataset(path+"E.2000_C5.00-49.global_av_ressurf_OCN.nc")

# read in variable
ressurf = f.variables['RESSURF']
ressurf_lnd = f2.variables['RESSURF']
ressurf_ocn = f3.variables['RESSURF']
ts = f.variables['TS']
year = [i for i in range(1,50,1)]

# Q&D plot
fig = plt.figure()
plt.plot(year,ressurf, color='r',linewidth=2)
#plt.plot(year,ressurf_lnd, color='g',linewidth=2)
#plt.plot(year,ressurf_ocn, color='b',linewidth=2)
plt.xlabel('Year')
plt.ylabel('Surface Residual Energy (W/m^2)')
plt.title('Q&D Global Average Surface Residual Energy')
#plt.legend(labels = ("Land","Ocean"),
#          loc='upper right', title = "E 2000 Control")

fig2 = plt.figure()
plt.plot(year,ts)
plt.xlabel('Year')
plt.ylabel('TS')
plt.title('Q&D Global Average TS')

plt.show()

f.close()
f2.close()
f3.close()
