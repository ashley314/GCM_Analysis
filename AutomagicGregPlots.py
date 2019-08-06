## Gregory Plots - with linear regression ##
## update: less input needed
## Ashley Dicks ##

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import netCDF4 as nc
from scipy.interpolate import spline
import statsmodels.api as sm
import pandas as pd

##### Set by user #####

## set path for location of data
path = "/Users/ash/Documents/Gradschool/Research/Miocene/Data/"

## output to be analyzed
files = ['E.1850_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc',
         'E.2000_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc',
         'E.Mod_400_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc',
         'E.Mod_800_C5_TeoKor_Dem1SLF.0-30.global_av_all.nc']
## variables to analyzed
vari = ['RESTOM','TS']

## user defined legend names
legend_names_ud = ["PI Modified: Years 1-20","PI Modified: Years 20-99",
                    "Modern Modified: Years 1-20","Modern Modified: Years 20-99",
                    "Modern 400 Modified: Years 1-20","Modern 400 Modified: Years 20-99",
                    "Modern 800 Modified: Years 1-20","Modern 800 Modified: Years 20-99"]
legend_names_ud2 = ["PI Modified",
                    "Modern Modified",
                    "Modern 400 Modified",
                    "Modern 800 Modified"]
legend_type2 = [Line2D([0],[0],color='#27AE60',lw=3),
                Line2D([0],[0],color='#3498DB',lw=3),
                Line2D([0],[0],color='#9B59B6',lw=3),
                Line2D([0],[0],color='#E74C3C',lw=3)]

##### Calculations / Data manipulations #####

# read in datasets from files listed
var_data = []
for i in range(len(files)):
    for k in range(len(vari)):
        f = nc.Dataset(path+files[i])
        var_data.append(f.variables[vari[k]])
# calculate delta ts
for i in range(1,len(var_data),len(vari)):
    D_ts = [var_data[i][k]-var_data[i][0] for k in range(0,len(var_data[i]))]
    var_data.append(D_ts)
# get time as year


# calculate best fit lines for gregory plot
def GregLine(delta_ts, restom, stYR, endYR):
    x_val = np.unique(delta_ts[stYR:endYR])
    y_val = np.poly1d(np.polyfit(delta_ts[stYR:endYR],restom[stYR:endYR],1))(x_val)
    m = (y_val[-1]-y_val[0])/(x_val[-1]-x_val[0])
    b = y_val[0]-m*x_val[0]
    return(x_val,y_val,m,b)

# TODO: linear regression calculations 

##### PLOTTING #####

# define colors and markers for plotting ## TODO Changed colors
c1 = ['#7DCEA0' for i in range(0,19)]
[c1.append('#27AE60') for i in range(19,len(var_data[-4]))]
c2 = ['#85C1E9' for i in range(0,19)]
[c2.append('#3498DB') for i in range(19,len(var_data[-3]))]
c3 = ['#C39BD3' for i in range(0,19)]
[c3.append('#9B59B6') for i in range(19,len(var_data[-2]))]
c4 = ['#F1948A' for i in range(0,19)]
[c4.append('#E74C3C') for i in range(19,len(var_data[-1]))]
m1 = ['o' for i in range(0,len(var_data[-4]))]
m2 = ['^' for i in range(0,len(var_data[-3]))]
m3 = ['s' for i in range(0,len(var_data[-2]))]
m4 = ['X' for i in range(0,len(var_data[-1]))]
time = [t for t in range(1,100)]
colors = [c1,c1,c2,c2,c3,c3,c4,c4,c1,c2,c3,c4]
markers = [m1,m1,m2,m2,m3,m3,m4,m4,m1,m2,m3,m4]

# define legends
legend_type = [Line2D([0],[0],color='#7DCEA0',lw=0,marker='o'), Line2D([0],[0],color='#27AE60',lw=0,marker='o'),
               Line2D([0],[0],color='#85C1E9',lw=0,marker='^'), Line2D([0],[0],color='#3498DB',lw=0,marker='^'),
               Line2D([0],[0],color='#C39BD3',lw=0,marker='s'), Line2D([0],[0],color='#9B59B6',lw=0,marker='s'),
               Line2D([0],[0],color='#F1948A',lw=0,marker='X'), Line2D([0],[0],color='#E74C3C',lw=0,marker='X')]
legend_names = [files[0]+": Years 1-20",files[0]+": Years 20-99",
                        files[1]+": Years 1-20",files[1]+": Years 20-99",
                        files[2]+": Years 1-20",files[2]+": Years 20-99",
                        files[3]+": Years 1-20",files[3]+": Years 20-99"]

# FIGURE ONE: temperature difference time series comparison
fig1 = plt.figure(figsize=(10,7))
plt.suptitle("Temperature Difference Plots")
plt.ylabel("Temperature Change [K]")
plt.xlabel("Time [year]")
##for y in range(len(files)*len(vari),len(var_data)):
##    for i in range(len(var_data[y])):
##         plt.plot(time[i],var_data[y][i],marker=markers[y][i],color=colors[y][i])
##         plt.plot(time[i],var_data[y][i],marker=markers[y][i],color=colors[y][i])
##         plt.plot(time[i],var_data[y][i],marker=markers[y][i],color=colors[y][i])
##         plt.plot(time[i],var_data[y][i],marker=markers[y][i],color=colors[y][i])
plt.plot(var_data[8],color='#27AE60',lw=3)
plt.plot(var_data[9],color='#3498DB',lw=3)
plt.plot(var_data[10],color='#9B59B6',lw=3)
plt.plot(var_data[11],color='#E74C3C',lw=3)
# ADD control cases ls='dashed'
plt.legend(legend_type2,legend_names_ud2)

# FIGURE TWO: restom plots
fig2 = plt.figure(figsize=(10,7))
plt.suptitle("RESTOM Plots")
plt.ylabel("RESTOM [W/m2]")
plt.xlabel("Time [year]")
for y in range(0,len(var_data)-4,2):
    for i in range(len(var_data[y])):
         plt.plot(time[i],var_data[y][i],marker=markers[y][i],color=colors[y][i])
         plt.plot(time[i],var_data[y][i],marker=markers[y][i],color=colors[y][i])
         plt.plot(time[i],var_data[y][i],marker=markers[y][i],color=colors[y][i])
         plt.plot(time[i],var_data[y][i],marker=markers[y][i],color=colors[y][i])        
plt.legend(legend_type2,legend_names_ud)

# FIGURE THREE: temperature time series plot


# TODO
# FIGURE FOUR: temperature difference vs restom

# TODO: TABLE ONE: linear regression values

plt.show()
f.close()

##### plot: delta TS vs RESTOM
##### four separate plots
####fig = plt.figure(figsize=(10,7))
####plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
####plt.suptitle("E 2000 Gregory Plots")
####
####plt.subplot(221)
####x = np.linspace(1.5,3.0,100)
####points1 = plt.plot(delta_ts[0:19],restom[0:19],'bo')
####points2 = plt.plot(delta_ts[19:],restom[19:],'o',color='red')
####x_val,y_val,m,b = GregLine(delta_ts,restom,0,19)
####line1 = plt.plot(x_val,y_val,"b")
####line2 = plt.plot(x,m*x+b, '--b')
####x_val,y_val,m,b = GregLine(delta_ts,restom,19,len(restom))
####line1 = plt.plot(x_val,y_val,"r")
####line2 = plt.plot(x,m*x+b, '--r')
####plt.xlabel("Temperature Change [K]")
####plt.ylabel("RESTOM [W/m^2]")
####plt.title("Control")
####plt.xlim(0,3.0)
####plt.ylim(-0.75,2.0)
####line_legend = [Line2D([0],[0],color='blue',lw=2), Line2D([0],[0],color='red',lw=2)]
####plt.legend(line_legend,["Years 1-20","Years 20-95"])
####
####plt.subplot(222)
####x = np.linspace(1.5,3.0,100)
####points1 = plt.plot(delta_ts_fin[0:19],restom_fin[0:19],'bo')
####x_val,y_val,m,b = GregLine(delta_ts_fin,restom_fin,0,19)
####line1 = plt.plot(x_val,y_val,"b")
####line2 = plt.plot(x,m*x+b, '--b')
####points2 = plt.plot(delta_ts_fin[19:],restom_fin[19:],'o',color='red')
####x_val,y_val,m,b = GregLine(delta_ts_fin,restom_fin,19,len(restom_fin))
####line1 = plt.plot(x_val,y_val,"red")
####line2 = plt.plot(x,m*x+b, '--',color="red")
####plt.xlabel("Temperature Change [K]")
####plt.ylabel("RESTOM [W/m^2]")
####plt.title("Fin2")
####plt.xlim(0,3.0)
####plt.ylim(-0.75,2.0)
####line_legend = [Line2D([0],[0],color='blue',lw=2), Line2D([0],[0],color='red',lw=2)]
####plt.legend(line_legend,["Years 1-20","Years 20-99"])
####
####plt.subplot(223)
####x = np.linspace(2.1,5.0,100)
####points1 = plt.plot(delta_ts_tk[0:19],restom_tk[0:19],'bo')
####x_val,y_val,m,b = GregLine(delta_ts_tk,restom_tk,0,19)
####line1 = plt.plot(x_val,y_val,"b")
####line2 = plt.plot(x,m*x+b, '--b')
####points2 = plt.plot(delta_ts_tk[19:],restom_tk[19:],'o',color='red')
####x_val,y_val,m,b = GregLine(delta_ts_tk,restom_tk,19,len(restom_tk))
####line1 = plt.plot(x_val,y_val,"red")
####line2 = plt.plot(x,m*x+b, '--',color="red")
####plt.xlabel("Temperature Change [K]")
####plt.ylabel("RESTOM [W/m^2]")
####plt.title("TeoKor_Dem1SLF")
####plt.xlim(0,5.0)
####plt.ylim(-0.75,2.0)
####line_legend = [Line2D([0],[0],color='blue',lw=2), Line2D([0],[0],color='red',lw=2)]
####plt.legend(line_legend,["Years 1-20","Years 20-99"])
####
####plt.subplot(224)
####x = np.linspace(-1.5,0,100)
####points1 = plt.plot(delta_ts_kor[0:19],restom_kor[0:19],'bo')
####x_val,y_val,m,b = GregLine(delta_ts_kor,restom_kor,0,19)
####line1 = plt.plot(x_val,y_val,"b")
####line2 = plt.plot(x,m*x+b, '--b')
####points2 = plt.plot(delta_ts_kor[19:],restom_kor[19:],'o',color='red')
####x_val,y_val,m,b = GregLine(delta_ts_kor,restom_kor,19,len(restom_kor))
####line1 = plt.plot(x_val,y_val,"red")
####line2 = plt.plot(x,m*x+b, '--',color="red")
####plt.xlabel("Temperature Change [K]")
####plt.ylabel("RESTOM [W/m^2]")
####plt.title("Kor1")
####plt.xlim(-4.0,0)
#####plt.ylim(-0.75,2.0)
####line_legend = [Line2D([0],[0],color='blue',lw=2), Line2D([0],[0],color='red',lw=2)]
####plt.legend(line_legend,["Years 1-20","Years 20-99"])
##
### plot: delta TS vs RESTOM
### all on one plot
##fig = plt.figure(figsize=(10,7))
##plt.suptitle("Gregory Plots for 800 CO2 Simulations")
##
##plt.xlim(0,10.0)
##plt.ylim(-0.75,5.0)
##plt.xlabel("Temperature Change [K]")
##plt.ylabel("RESTOM [W/m^2]")
##
##x = np.linspace(0,10.0,100)
##points1 = plt.plot(delta_ts[0:19],restom[0:19],'o',color="#7DCEA0")
##points2 = plt.plot(delta_ts[19:],restom[19:],'o',color='#27AE60')
##x_val,y_val,m,b = GregLine(delta_ts,restom,0,19)
##line1 = plt.plot(x_val,y_val,color="#7DCEA0")
##line2 = plt.plot(x,m*x+b, '--',color="#7DCEA0")
###x_val,y_val,m,b = GregLine(delta_ts,restom,19,len(restom))
##x_val,y_val,m,b = GregLine(delta_ts,restom,19,40)
##line1 = plt.plot(x_val,y_val,color='#27AE60')
##line2 = plt.plot(x,m*x+b, '--',color='#27AE60')
##
##x = np.linspace(2.5,10.0,100)
##points1 = plt.plot(delta_ts_fin[0:19],restom_fin[0:19],'^',color="#85C1E9")
##x_val,y_val,m,b = GregLine(delta_ts_fin,restom_fin,0,19)
##line1 = plt.plot(x_val,y_val,color="#85C1E9")
##line2 = plt.plot(x,m*x+b, '--',color="#85C1E9")
##points2 = plt.plot(delta_ts_fin[19:],restom_fin[19:],'^',color='#3498DB')
###x_val,y_val,m,b = GregLine(delta_ts_fin,restom_fin,19,len(restom_fin))
##x_val,y_val,m,b = GregLine(delta_ts_fin,restom_fin,19,40)
##line1 = plt.plot(x_val,y_val,color='#3498DB')
##line2 = plt.plot(x,m*x+b, '--',color='#3498DB')
##
##x = np.linspace(2.5,10.0,100)
##points1 = plt.plot(delta_ts_tk[0:19],restom_tk[0:19],'s',color='#C39BD3')
##x_val,y_val,m,b = GregLine(delta_ts_tk,restom_tk,0,19)
##line1 = plt.plot(x_val,y_val,color='#C39BD3')
##line2 = plt.plot(x,m*x+b, '--',color='#C39BD3')
##points2 = plt.plot(delta_ts_tk[19:],restom_tk[19:],'s',color='#9B59B6')
###x_val,y_val,m,b = GregLine(delta_ts_tk,restom_tk,19,len(restom_tk))
##x_val,y_val,m,b = GregLine(delta_ts_tk,restom_tk,19,40)
##line1 = plt.plot(x_val,y_val,color='#9B59B6')
##line2 = plt.plot(x,m*x+b, '--',color='#9B59B6')
##
##x = np.linspace(7,10.0,100)
##points1 = plt.plot(delta_ts_kor[0:19],restom_kor[0:19],'X', color='#F1948A')
##x_val,y_val,m,b = GregLine(delta_ts_kor,restom_kor,0,19)
##line1 = plt.plot(x_val,y_val,color='#F1948A')
##line2 = plt.plot(x,m*x+b, '--',color='#F1948A')
##points2 = plt.plot(delta_ts_kor[19:],restom_kor[19:],'X',color='#E74C3C')
###x_val,y_val,m,b = GregLine(delta_ts_kor,restom_kor,19,len(restom_kor))
##x_val,y_val,m,b = GregLine(delta_ts_kor,restom_kor,19,40)
##line1 = plt.plot(x_val,y_val,color="#E74C3C")
##line2 = plt.plot(x,m*x+b, '--',color="#E74C3C")
##
##line_legend = [Line2D([0],[0],color='#7DCEA0',lw=2), Line2D([0],[0],color='#27AE60',lw=2),
##               Line2D([0],[0],color='#85C1E9',lw=2), Line2D([0],[0],color='#3498DB',lw=2),
##               Line2D([0],[0],color='#C39BD3',lw=2), Line2D([0],[0],color='#9B59B6',lw=2),
##               Line2D([0],[0],color='#F1948A',lw=2), Line2D([0],[0],color='#E74C3C',lw=2)]
####plt.legend(line_legend,["CO2=284.7: Years 1-20","CO2=284.7: Years 20-99",
####                        "CO2=367.0: Years 1-20","CO2=367.0: Years 20-99",
####                        "CO2=400: Years 1-20","CO2=400: Years 20-99",
####                        "CO2=800:Years 1-20","CO2=800: Years 20-29"])
##plt.legend(line_legend,["Control: Years 1-20","Control: Years 20-99",
##                        "Kor1 Years 1-20","Kor1: Years 20-99",
##                        "Fin2 Years 1-20","Fin2: Years 20-99",
##"TeoKor:Years 1-20","TeoKor: Years 20-29"])

##
### define colors and markers for plotting ## Changed colors
##c1 = ['#7DCEA0' for i in range(0,19)]
##[c1.append('#27AE60') for i in range(19,len(var_data[-4]))]
##c2 = ['#85C1E9' for i in range(0,19)]
##[c2.append('#3498DB') for i in range(19,len(var_data[-3]))]
##c3 = ['#C39BD3' for i in range(0,19)]
##[c3.append('#9B59B6') for i in range(19,len(var_data[-2]))]
##c4 = ['#F1948A' for i in range(0,19)]
##[c4.append('#E74C3C') for i in range(19,len(var_data[-1]))]
##m1 = ['o' for i in range(0,len(var_data[-4]))]
##m2 = ['^' for i in range(0,len(var_data[-3]))]
##m3 = ['s' for i in range(0,len(var_data[-2]))]
##m4 = ['X' for i in range(0,len(var_data[-1]))]
##time = [t for t in range(1,100)]
##colors = [c1,c1,c2,c2,c3,c3,c4,c4,c1,c2,c3,c4]
##markers = [m1,m1,m2,m2,m3,m3,m4,m4,m1,m2,m3,m4]
