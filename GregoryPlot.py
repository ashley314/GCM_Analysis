## Gregory Plots ##
## Ashley Dicks ##

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import netCDF4 as nc
from scipy.interpolate import spline
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# set path for location of data
path = "/Users/ash/Documents/Gradschool/Research/Miocene/Data/"

# all same simulation base
##f = nc.Dataset(path+"E.2000_C5.0-95.global_av_all.nc")
##f2 = nc.Dataset(path+"E.2000_C5_fin2.0-100.global_av_all.nc")
##f3 = nc.Dataset(path+"E.2000_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc")
##f4 = nc.Dataset(path+"E.2000_C5_Kor1.0-100.global_av_all.nc")

# all teokor
##f = nc.Dataset(path+"E.1850_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc")
##f2 = nc.Dataset(path+"E.2000_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc")
##f3 = nc.Dataset(path+"E.Mod_400_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc")
##f4 = nc.Dataset(path+"E.Mod_800_C5_TeoKor_Dem1SLF.0-30.global_av_all.nc")

# all 800 simulations
##f = nc.Dataset(path+"E.Mod_800_C5_CNTL.0-100.global_av_all.nc")
##f2 = nc.Dataset(path+"E.Mod_800_C5_Kor1.0-100.global_av_all.nc")
##f3 = nc.Dataset(path+"E.Mod_800_C5_fin2.0-100.global_av_all.nc")
##f4 = nc.Dataset(path+"E.Mod_800_C5_TeoKor_Dem1SLF.0-30.global_av_all.nc")

# B cases
f = nc.Dataset(path+"B.MIO_400_C5_MK_TeoKor_Dem1SLF_2.1400-1760.global_av_all.nc")
f2 = nc.Dataset(path+"B.MIO_400_C5_MK_Kor1.1270-1370.global_av_all.nc")
f3 = nc.Dataset(path+"B.MIO_800_C5_MK_TeoKor_Dem1SLF.1200-1360.global_av_all.nc")

# read in variables to plot
restom = f.variables['RESTOM']
ts = f.variables['TS']
restom_2 = f2.variables['RESTOM']
ts_2 = f2.variables['TS']
restom_3 = f3.variables['RESTOM']
ts_3 = f3.variables['TS']
timex = f.variables['time']
timex_2 = f2.variables['time']
timex_3 = f3.variables['time']
#restom_4 = f4.variables['RESTOM']
#ts_4 = f4.variables['TS']

# calculate delta TS (from previous year)
delta_ts = [ts[i]-ts[0] for i in range(1,len(ts))]
restom = restom[1:len(restom)]
delta_ts_2 = [ts_2[i]-ts_2[0] for i in range(1,len(ts_2))]
restom_2 = restom_2[1:len(restom_2)]
delta_ts_3 = [ts_3[i]-ts_3[0] for i in range(1,len(ts_3))]
restom_3 = restom_3[1:len(restom_3)]
#delta_ts_4 = [ts_4[i]-ts_4[0] for i in range(1,len(ts_4))]
#restom_4 = restom_4[1:len(restom_4)]

timexx = [timex[x]/365. for x in range(len(timex))]
timexx_2 = [timex_2[x]/365. for x in range(len(timex_2))]
timexx_3 = [timex_3[x]/365. for x in range(len(timex_3))]

# Calculate best fit lines for gregory plot
def GregLine(delta_ts, restom, stYR, endYR):
    x_val = np.unique(delta_ts[stYR:endYR])
    y_val = np.poly1d(np.polyfit(delta_ts[stYR:endYR],restom[stYR:endYR],1))(x_val)
    m = (y_val[-1]-y_val[0])/(x_val[-1]-x_val[0])
    b = y_val[0]-m*x_val[0]
    return(x_val,y_val,m,b)

# Calculate smooth fit lines
def smooth(xspan,data):
    x_points = np.linspace(xspan[0],xspan[-1],25) #change number for 'smoothness' level
    smooth_points = spline(xspan,data,x_points)
    return(x_points,smooth_points)

# Calculate linear regression lines
# use .fit() and .predict()
model= sm.OLS(restom,delta_ts)
results = model.fit()
print(results)

### plot: delta TS vs RESTOM
### four separate plots
##fig = plt.figure(figsize=(10,7))
##plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
##plt.suptitle("E 2000 Gregory Plots")
##
##plt.subplot(221)
##x = np.linspace(1.5,3.0,100)
##points1 = plt.plot(delta_ts[0:19],restom[0:19],'bo')
##points2 = plt.plot(delta_ts[19:],restom[19:],'o',color='red')
##x_val,y_val,m,b = GregLine(delta_ts,restom,0,19)
##line1 = plt.plot(x_val,y_val,"b")
##line2 = plt.plot(x,m*x+b, '--b')
##x_val,y_val,m,b = GregLine(delta_ts,restom,19,len(restom))
##line1 = plt.plot(x_val,y_val,"r")
##line2 = plt.plot(x,m*x+b, '--r')
##plt.xlabel("Temperature Change [K]")
##plt.ylabel("RESTOM [W/m^2]")
##plt.title("Control")
##plt.xlim(0,3.0)
##plt.ylim(-0.75,2.0)
##line_legend = [Line2D([0],[0],color='blue',lw=2), Line2D([0],[0],color='red',lw=2)]
##plt.legend(line_legend,["Years 1-20","Years 20-95"])
##
##plt.subplot(222)
##x = np.linspace(1.5,3.0,100)
##points1 = plt.plot(delta_ts_fin[0:19],restom_fin[0:19],'bo')
##x_val,y_val,m,b = GregLine(delta_ts_fin,restom_fin,0,19)
##line1 = plt.plot(x_val,y_val,"b")
##line2 = plt.plot(x,m*x+b, '--b')
##points2 = plt.plot(delta_ts_fin[19:],restom_fin[19:],'o',color='red')
##x_val,y_val,m,b = GregLine(delta_ts_fin,restom_fin,19,len(restom_fin))
##line1 = plt.plot(x_val,y_val,"red")
##line2 = plt.plot(x,m*x+b, '--',color="red")
##plt.xlabel("Temperature Change [K]")
##plt.ylabel("RESTOM [W/m^2]")
##plt.title("Fin2")
##plt.xlim(0,3.0)
##plt.ylim(-0.75,2.0)
##line_legend = [Line2D([0],[0],color='blue',lw=2), Line2D([0],[0],color='red',lw=2)]
##plt.legend(line_legend,["Years 1-20","Years 20-99"])
##
##plt.subplot(223)
##x = np.linspace(2.1,5.0,100)
##points1 = plt.plot(delta_ts_tk[0:19],restom_tk[0:19],'bo')
##x_val,y_val,m,b = GregLine(delta_ts_tk,restom_tk,0,19)
##line1 = plt.plot(x_val,y_val,"b")
##line2 = plt.plot(x,m*x+b, '--b')
##points2 = plt.plot(delta_ts_tk[19:],restom_tk[19:],'o',color='red')
##x_val,y_val,m,b = GregLine(delta_ts_tk,restom_tk,19,len(restom_tk))
##line1 = plt.plot(x_val,y_val,"red")
##line2 = plt.plot(x,m*x+b, '--',color="red")
##plt.xlabel("Temperature Change [K]")
##plt.ylabel("RESTOM [W/m^2]")
##plt.title("TeoKor_Dem1SLF")
##plt.xlim(0,5.0)
##plt.ylim(-0.75,2.0)
##line_legend = [Line2D([0],[0],color='blue',lw=2), Line2D([0],[0],color='red',lw=2)]
##plt.legend(line_legend,["Years 1-20","Years 20-99"])
##
##plt.subplot(224)
##x = np.linspace(-1.5,0,100)
##points1 = plt.plot(delta_ts_kor[0:19],restom_kor[0:19],'bo')
##x_val,y_val,m,b = GregLine(delta_ts_kor,restom_kor,0,19)
##line1 = plt.plot(x_val,y_val,"b")
##line2 = plt.plot(x,m*x+b, '--b')
##points2 = plt.plot(delta_ts_kor[19:],restom_kor[19:],'o',color='red')
##x_val,y_val,m,b = GregLine(delta_ts_kor,restom_kor,19,len(restom_kor))
##line1 = plt.plot(x_val,y_val,"red")
##line2 = plt.plot(x,m*x+b, '--',color="red")
##plt.xlabel("Temperature Change [K]")
##plt.ylabel("RESTOM [W/m^2]")
##plt.title("Kor1")
##plt.xlim(-4.0,0)
###plt.ylim(-0.75,2.0)
##line_legend = [Line2D([0],[0],color='blue',lw=2), Line2D([0],[0],color='red',lw=2)]
##plt.legend(line_legend,["Years 1-20","Years 20-99"])

# plot: delta TS vs RESTOM
# all on one plot
fig = plt.figure(figsize=(10,7))
plt.suptitle("Gregory Plots")

plt.xlim(-0.5,1.5)
plt.ylim(-0.75,3.0)
plt.xlabel("Temperature Change [K]")
plt.ylabel("RESTOM [W/m^2]")

x = np.linspace(0,10.0,100)
points1 = plt.plot(delta_ts[0:19],restom[0:19],'o',color="#7DCEA0")
points2 = plt.plot(delta_ts[19:],restom[19:],'o',color='#27AE60')
##x_val,y_val,m,b = GregLine(delta_ts,restom,0,19)
##line1 = plt.plot(x_val,y_val,color="#7DCEA0")
##line2 = plt.plot(x,m*x+b, '--',color="#7DCEA0")
###x_val,y_val,m,b = GregLine(delta_ts,restom,19,len(restom))
##x_val,y_val,m,b = GregLine(delta_ts,restom,19,40)
##line1 = plt.plot(x_val,y_val,color='#27AE60')
##line2 = plt.plot(x,m*x+b, '--',color='#27AE60')

x = np.linspace(2.5,10.0,100)
points1 = plt.plot(delta_ts_2[0:19],restom_2[0:19],'^',color="#85C1E9")
##x_val,y_val,m,b = GregLine(delta_ts_2,restom_2,0,19)
##line1 = plt.plot(x_val,y_val,color="#85C1E9")
##line2 = plt.plot(x,m*x+b, '--',color="#85C1E9")
points2 = plt.plot(delta_ts_2[19:],restom_2[19:],'^',color='#3498DB')
#x_val,y_val,m,b = GregLine(delta_ts_fin,restom_fin,19,len(restom_fin))
#x_val,y_val,m,b = GregLine(delta_ts_2,restom_2,19,40)
#line1 = plt.plot(x_val,y_val,color='#3498DB')
#line2 = plt.plot(x,m*x+b, '--',color='#3498DB')

x = np.linspace(2.5,10.0,100)
points1 = plt.plot(delta_ts_3[0:19],restom_3[0:19],'s',color='#C39BD3')
#x_val,y_val,m,b = GregLine(delta_ts_3,restom_3,0,19)
#line1 = plt.plot(x_val,y_val,color='#C39BD3')
#line2 = plt.plot(x,m*x+b, '--',color='#C39BD3')
points2 = plt.plot(delta_ts_3[19:],restom_3[19:],'s',color='#9B59B6')
#x_val,y_val,m,b = GregLine(delta_ts_tk,restom_tk,19,len(restom_tk))
#x_val,y_val,m,b = GregLine(delta_ts_3,restom_3,19,40)
#line1 = plt.plot(x_val,y_val,color='#9B59B6')
#line2 = plt.plot(x,m*x+b, '--',color='#9B59B6')
##
##x = np.linspace(7,10.0,100)
##points1 = plt.plot(delta_ts_4[0:19],restom_4[0:19],'X', color='#F1948A')
##x_val,y_val,m,b = GregLine(delta_ts_4,restom_4,0,19)
##line1 = plt.plot(x_val,y_val,color='#F1948A')
##line2 = plt.plot(x,m*x+b, '--',color='#F1948A')
##points2 = plt.plot(delta_ts_4[19:],restom_4[19:],'X',color='#E74C3C')
###x_val,y_val,m,b = GregLine(delta_ts_kor,restom_kor,19,len(restom_kor))
##x_val,y_val,m,b = GregLine(delta_ts_4,restom_4,19,40)
##line1 = plt.plot(x_val,y_val,color="#E74C3C")
##line2 = plt.plot(x,m*x+b, '--',color="#E74C3C")

line_legend = [Line2D([0],[0],color='#7DCEA0',lw=2), Line2D([0],[0],color='#27AE60',lw=2),
               Line2D([0],[0],color='#85C1E9',lw=2), Line2D([0],[0],color='#3498DB',lw=2),
               Line2D([0],[0],color='#C39BD3',lw=2), Line2D([0],[0],color='#9B59B6',lw=2)]
#               Line2D([0],[0],color='#F1948A',lw=2), Line2D([0],[0],color='#E74C3C',lw=2)]
##plt.legend(line_legend,["CO2=284.7: Years 1-20","CO2=284.7: Years 20-99",
##                        "CO2=367.0: Years 1-20","CO2=367.0: Years 20-99",
##                        "CO2=400: Years 1-20","CO2=400: Years 20-99",
##                        "CO2=800:Years 1-20","CO2=800: Years 20-29"])
##plt.legend(line_legend,["Control: Years 1-20","Control: Years 20-99",
##                        "Kor1 Years 1-20","Kor1: Years 20-99",
##                        "Fin2 Years 1-20","Fin2: Years 20-99",
##                        "TeoKor:Years 1-20","TeoKor: Years 20-29"])
plt.legend(line_legend,["TeoKor 400","TeoKor 400",
                        "Kor 400","Kor 400",
                        "TeoKor 800","TeoKor 800"])
#                        "TeoKor:Years 1-20","TeoKor: Years 20-29"])


# Temperature difference time series comparison
fig2 = plt.figure(figsize=(10,7))
plt.suptitle("Temperature Plots")
plt.ylabel("Temperature Change [K]")
plt.xlabel("Time [year]")
c1 = ['#7DCEA0' for i in range(0,19)]
[c1.append('#27AE60') for i in range(19,len(delta_ts))]
c2 = ['#85C1E9' for i in range(0,19)]
[c2.append('#3498DB') for i in range(19,len(delta_ts_2))]
c3 = ['#C39BD3' for i in range(0,19)]
[c3.append('#9B59B6') for i in range(19,len(delta_ts_3))]
##c4 = ['#F1948A' for i in range(0,19)]
##[c4.append('#E74C3C') for i in range(19,len(delta_ts_4))]
time_1 = [t for t in range(1,len(delta_ts))]
time_2 = [t for t in range(1,len(delta_ts_2))]
time_3 = [t for t in range(1,len(delta_ts_3))]
for i in range(len(delta_ts)-1):
    plt.plot(time_1[i],delta_ts[i],'o',color=c1[i])
for i in range(len(delta_ts_2)-1):
    plt.plot(time_2[i],delta_ts_2[i],'^',color=c2[i])
for i in range(len(delta_ts_3)-1):
    plt.plot(time_3[i],delta_ts_3[i],'s',color=c3[i])
##for k in range(len(delta_ts_4)):
## plt.plot(time[k],delta_ts_4[k],'X',color=c4[k])
line_legend = [Line2D([0],[0],color='#7DCEA0',lw=0,marker='o'), Line2D([0],[0],color='#27AE60',lw=0,marker='o'),
               Line2D([0],[0],color='#85C1E9',lw=0,marker='^'), Line2D([0],[0],color='#3498DB',lw=0,marker='^'),
               Line2D([0],[0],color='#C39BD3',lw=0,marker='s'), Line2D([0],[0],color='#9B59B6',lw=0,marker='s')]
#               Line2D([0],[0],color='#F1948A',lw=0,marker='X'), Line2D([0],[0],color='#E74C3C',lw=0,marker='X')]
##plt.legend(line_legend,["CO2=284.7: Years 1-20","CO2=284.7: Years 20-99",
##                        "CO2=367.0: Years 1-20","CO2=367.0: Years 20-99",
##                        "CO2=400: Years 1-20","CO2=400: Years 20-99",
##                        "CO2=800:Years 1-20","CO2=800: Years 20-29"])
plt.legend(line_legend,["TeoKor 400","TeoKor 400",
                        "Kor 400","Kor 400",
                        "TeoKor 800","TeoKor 800"])
 #                       "TeoKor:Years 1-20","TeoKor: Years 20-29"])

timex_axis = [x for x in range(1200,1760,1)]
# temperature time series
fig3 = plt.figure(figsize=(10,7))
plt.suptitle("Temperature Time Series")
plt.ylabel("Temperature [K]")
plt.xlabel("Time [model year]")
plt.plot(timexx,ts,color='#27AE60')
plt.plot(timexx_2,ts_2, color='#3498DB')
plt.plot(timexx_3,ts_3, color='#9B59B6')
line_legend = [Line2D([0],[0],color='#27AE60',lw=1), Line2D([0],[0],color='#3498DB',lw=1), Line2D([0],[0],color='#9B59B6',lw=1)]
plt.legend(line_legend,["TeoKor 400",
                        "Kor 400",
                        "TeoKor 800"])
 
plt.show()

# RESTOM time sereis comparison
fig3 = plt.figure(figsize=(10,7))
plt.suptitle("CO2 800 Simulation RESTOM Plots")
plt.ylabel("RESTOM [W/m2]")
plt.xlabel("Time [year]")
c1 = ['#7DCEA0' for i in range(0,19)]
[c1.append('#27AE60') for i in range(19,len(restom))]
c2 = ['#85C1E9' for i in range(0,19)]
[c2.append('#3498DB') for i in range(19,len(restom_2))]
c3 = ['#C39BD3' for i in range(0,19)]
[c3.append('#9B59B6') for i in range(19,len(restom_3))]
c4 = ['#F1948A' for i in range(0,19)]
[c4.append('#E74C3C') for i in range(19,len(restom_4))]
time = [t for t in range(1,99)]
# plot points
for i in range(len(restom)):
 plt.plot(time[i],restom[i],'o',color=c1[i])
 plt.plot(time[i],restom_2[i],'^',color=c2[i])
 plt.plot(time[i],restom_3[i],'s',color=c3[i])
for k in range(len(restom_4)):
 plt.plot(time[k],restom_4[k],'X',color=c4[k])
# add smooth lines
#x_points,smooth_points = smooth(time,restom)
#plt.plot(x_points,smooth_points,color='#27AE60')
line_legend = [Line2D([0],[0],color='#7DCEA0',lw=0,marker='o'), Line2D([0],[0],color='#27AE60',lw=0,marker='o'),
               Line2D([0],[0],color='#85C1E9',lw=0,marker='^'), Line2D([0],[0],color='#3498DB',lw=0,marker='^'),
               Line2D([0],[0],color='#C39BD3',lw=0,marker='s'), Line2D([0],[0],color='#9B59B6',lw=0,marker='s'),
               Line2D([0],[0],color='#F1948A',lw=0,marker='X'), Line2D([0],[0],color='#E74C3C',lw=0,marker='X')]
##plt.legend(line_legend,["CO2=284.7: Years 1-20","CO2=284.7: Years 20-99",
##                        "CO2=367.0: Years 1-20","CO2=367.0: Years 20-99",
##                        "CO2=400: Years 1-20","CO2=400: Years 20-99",
##                        "CO2=800:Years 1-20","CO2=800: Years 20-29"])
plt.legend(line_legend,["Control: Years 1-20","Control: Years 20-99",
                        "Kor1 Years 1-20","Kor1: Years 20-99",
                        "Fin2 Years 1-20","Fin2: Years 20-99",
                        "TeoKor:Years 1-20","TeoKor: Years 20-29"])

plt.show()

f.close()
f2.close()
f3.close()
f4.close()
