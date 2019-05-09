## Analysis of mean surface temperature across som runs ##
## Ashley Dicks ##

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# set path for location of data
path = "/Users/ash/Documents/GitHub/GCM_Analysis/Data/"

# open data files
file1 = nc.Dataset(path+"E.2000_C5.TS.nc")
file2 = nc.Dataset(path+"E.2000_C5_Kor1.TS.nc")
file3 = nc.Dataset(path+"E.2000_C5_TeoKor_Dem1SLF.TS.nc")
file4 = nc.Dataset(path+"E.2000_C5_fin2.TS.nc")

# read in variables
ts_mod1 = file1.variables['TS']
ts_mod2 = file2.variables['TS']
ts_mod3 = file3.variables['TS']
ts_mod4 = file4.variables['TS']
lat = file1.variables['lat']
lon = file1.variables['lon']

## TODO: make into function ##
# calculate monthly average
# make sure to include weights
gw = [0.000136695007530996, 0.00109333584416416, 0.002185476141635, 
    0.0032752266530317, 0.00436139575216965, 0.00544279572908613, 
    0.00651824408878299, 0.00758656484426889, 0.00864658980248234, 
    0.00969715984169461, 0.0107371261789923, 0.0117653516264541, 
    0.0127807118346497, 0.0137820965220986, 0.0147684106893471, 
    0.0157385758163346, 0.0166915310417382, 0.0176262343230105, 
    0.0185416635758361, 0.0194368177917678, 0.0203107181328123, 
    0.0211624090017778, 0.0219909590872031, 0.0227954623817345, 
    0.0235750391728312, 0.0243288370047177, 0.0250560316105318, 
    0.0257558278136486, 0.0264274603971946, 0.0270701949408004, 
    0.0276833286236791, 0.0282661909931501, 0.0288181446977701, 
    0.029338586184267, 0.0298269463575161, 0.0302826912028381, 
    0.0307053223699359, 0.0310943777178331, 0.0314494318202186, 
    0.031770096430645, 0.0320560209070681, 0.0323068925952713, 
    0.0325224371707465, 0.0327024189386649, 0.0328466410916061, 
    0.0329549459247632, 0.0330272150083913, 0.0330633693173083, 
    0.0330633693173082, 0.0330272150083912, 0.0329549459247632, 
    0.0328466410916062, 0.0327024189386651, 0.0325224371707464, 
    0.0323068925952712, 0.0320560209070681, 0.0317700964306451, 
    0.0314494318202188, 0.031094377717833, 0.0307053223699357, 
    0.0302826912028381, 0.0298269463575162, 0.0293385861842671, 
    0.0288181446977702, 0.0282661909931499, 0.0276833286236791, 
    0.0270701949408004, 0.0264274603971947, 0.0257558278136487, 
    0.0250560316105317, 0.0243288370047176, 0.0235750391728311, 
    0.0227954623817346, 0.0219909590872033, 0.0211624090017778, 
    0.0203107181328123, 0.0194368177917678, 0.0185416635758361, 
    0.0176262343230104, 0.0166915310417383, 0.0157385758163346, 
    0.014768410689347, 0.0137820965220985, 0.0127807118346497, 
    0.0117653516264543, 0.0107371261789923, 0.00969715984169461, 
    0.00864658980248223, 0.00758656484426889, 0.0065182440887831, 
    0.00544279572908601, 0.00436139575216976, 0.0032752266530317, 
    0.00218547614163489, 0.00109333584416427, 0.00013669500753099]
# Average file1
ts_zonal_mod1 = np.average(ts_mod1, axis=2)
ts_avg_mod1 = np.average(ts_zonal_mod1, axis=1, weights=gw)
np.asarray(ts_avg_mod1)
# Average file2
ts_zonal_mod2 = np.average(ts_mod2, axis=2)
ts_avg_mod2 = np.average(ts_zonal_mod2, axis=1, weights=gw)
np.asarray(ts_avg_mod2)
# Average file3
ts_zonal_mod3 = np.average(ts_mod3, axis=2)
ts_avg_mod3 = np.average(ts_zonal_mod3, axis=1, weights=gw)
np.asarray(ts_avg_mod3)
# Average file4
ts_zonal_mod4 = np.average(ts_mod4, axis=2)
ts_avg_mod4 = np.average(ts_zonal_mod4, axis=1, weights=gw)
np.asarray(ts_avg_mod4)

# average for each year
ts_yr_avg_mod1 = []
ts_yr_avg_mod2 = []
ts_yr_avg_mod3 = []
ts_yr_avg_mod4 = []
for i in range(0,len(ts_avg_mod1)-12):
    yr1 = np.average(ts_avg_mod1[i:i+12])
    ts_yr_avg_mod1.append(yr1)
    yr2 = np.average(ts_avg_mod2[i:i+12])
    ts_yr_avg_mod2.append(yr2)
    yr3 = np.average(ts_avg_mod3[i:i+12])
    ts_yr_avg_mod3.append(yr3)
    yr4 = np.average(ts_avg_mod4[i:i+12])
    ts_yr_avg_mod4.append(yr4)
    i = i + 12

# quick and dirty plot
fig = plt.figure()
line1 = plt.plot(ts_yr_avg_mod1,color='black',linewidth=4)
line2 = plt.plot(ts_yr_avg_mod2,color="b")
line3 = plt.plot(ts_yr_avg_mod3,color="g")
line4 = plt.plot(ts_yr_avg_mod4,color="r")
plt.ylim((282,292))
plt.xlabel('Year')
plt.ylabel('Temperature (K)')
plt.title('Q&D Yearly Average Temperature')
plt.legend(labels = ("Control","Kor1","TeoKor","Fin2"),
           loc='lower right', title = "E 2000 Simulation")
save_loc = "/Users/ash/Documents/GitHub/GCM_Analysis/Figures/"
plt.savefig(save_loc+"E2000_ModelTempDiffs.png",dpi=500)
plt.show()

# close files
file1.close()
file2.close()
file3.close()
file4.close()


