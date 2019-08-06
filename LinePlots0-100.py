## Line Plots - with linear regression ? ##
## update: less input needed
## Ashley Dicks ##

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import netCDF4 as nc
from scipy.interpolate import spline
from scipy import stats
import statsmodels.api as sm
import pandas as pd

##### Set by user #####

## set path for location of data
path = "/Users/ash/Documents/Gradschool/Research/Miocene/Data/"

## output to be analyzed
files = ["E.1850_C5.0-100.global_av_all.nc",
         'E.1850_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc',
         "E.2000_C5.0-100.global_av_all.nc",
         'E.2000_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc',
         "E.Mod_C5_CNTL.0-100.global_av_all.nc",
         'E.Mod_400_C5_TeoKor_Dem1SLF.0-100.global_av_all.nc',
         "E.Mod_800_C5_CNTL.0-100.global_av_all.nc",
         'E.Mod_800_C5_TeoKor_Dem1SLF.0-30.global_av_all.nc']
## variables to analyzed
vari = ['RESTOM','TS']

## user defined legend names
legend_names_ud2 = ["PI Control","PI Modified",
                    "Modern Control","Modern Modified",
                    "Modern 400 Control","Modern 400 Modified",
                    "Modern 800 Control","Modern 800 Modified"]
legend_type2 = [Line2D([0],[0],color='#27AE60',lw=2,ls='dashed'),
                Line2D([0],[0],color='#27AE60',lw=2),
                Line2D([0],[0],color='#3498DB',lw=2,ls="dashed"),
                Line2D([0],[0],color='#3498DB',lw=2),
                Line2D([0],[0],color='#9B59B6',lw=2,ls="dashed"),
                Line2D([0],[0],color='#9B59B6',lw=2),
                Line2D([0],[0],color='#E74C3C',lw=2,ls="dashed"),
                Line2D([0],[0],color='#E74C3C',lw=2)]

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
    #D_ts = [var_data[i][0]-var_data[i][k] for k in range(0,len(var_data[i]))]
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

### fix for TS MOD control

# FIGURE ONE: temperature difference time series comparison
fig1 = plt.figure(figsize=(10,7))
#plt.suptitle("Temperature Difference")
plt.ylabel("Longwave Cloud Forcing Change")
plt.xlabel("Time [year]")
plt.plot(var_data[16],color='#27AE60',lw=2,ls="dashed")
plt.plot(var_data[17],color='#27AE60',lw=2)
plt.plot(var_data[18],color='#3498DB',lw=2,ls="dashed")
plt.plot(var_data[19],color='#3498DB',lw=2)
plt.plot(var_data[20],color='#9B59B6',lw=2,ls="dashed")
plt.plot(var_data[21],color='#9B59B6',lw=2)
plt.plot(var_data[22],color='#E74C3C',lw=2,ls="dashed")
plt.plot(var_data[23],color='#E74C3C',lw=2)
# ADD control cases ls='dashed'
plt.legend(legend_type2,legend_names_ud2)

# FIGURE TWO: temperature time series comparison
fig2 = plt.figure(figsize=(10,7))
#plt.suptitle("Temperature Difference")
plt.ylabel("Temperature Change [K]")
plt.xlabel("Time [year]")
plt.plot(var_data[1],color='#27AE60',lw=3,ls="dashed")
plt.plot(var_data[3],color='#27AE60',lw=3)
plt.plot(var_data[5],color='#3498DB',lw=3,ls="dashed")
plt.plot(var_data[7],color='#3498DB',lw=3)
plt.plot(var_data[9],color='#9B59B6',lw=3,ls="dashed")
plt.plot(var_data[11],color='#9B59B6',lw=3)
plt.plot(var_data[13],color='#E74C3C',lw=3,ls="dashed")
plt.plot(var_data[15],color='#E74C3C',lw=3)
# ADD control cases ls='dashed'
plt.legend(legend_type2,legend_names_ud2)

### FIGURE THREE: difference in temperature with other variable changes
##comp_vars = ["SWCF"]
##diff_data = []
##for i in range(len(files)):
##    for k in range(len(comp_vars)):
##        f2 = nc.Dataset(path+files[i])
##        diff_data.append(f2.variables[comp_vars[k]])
### calculate difference of control and model
##d1 = diff_data[1][:]-diff_data[0][:]
##d2 = diff_data[3][:]-diff_data[2][:]
##d3 = diff_data[5][0:77]-diff_data[4][:]
##d4 = diff_data[7][:]-diff_data[6][0:30]
##    
##fig3 = plt.figure(figsize=(10,7))
##plt.xlabel("Temperature [K]")
##plt.ylabel("Change in "+comp_vars[0]+" (Modified-Contorl)")
###plt.ylim(-2,2)
##plt.scatter(var_data[3],d1,color='#27AE60')
##plt.scatter(var_data[7],d2,color='#3498DB')
##plt.scatter(var_data[11][0:77],d3,color='#9B59B6')
##plt.scatter(var_data[15],d4,color='#E74C3C')
##plt.legend([Line2D([0],[0],color='#27AE60',lw=0,marker="o"),
##            Line2D([0],[0],color='#3498DB',lw=0,marker="o"),
##            Line2D([0],[0],color='#9B59B6',lw=0,marker="o"),
##            Line2D([0],[0],color='#E74C3C',lw=0,marker="o")],
##           ["Pre-industrial","Modern","Modern 400","Modern 800"])

# FIGURE FOUR: av change in quantiy vs av temp (at end of run)
# calculations
comp_var1 = ["FLNT"]
diff_data1 = []
for i in range(len(files)):
    for k in range(len(comp_var1)):
        f2 = nc.Dataset(path+files[i])
        diff_data1.append(f2.variables[comp_var1[k]])
d1 = np.average(diff_data1[1][-10:-1])-np.average(diff_data1[0][-10:-1])
d2 = np.average(diff_data1[3][-10:-1])-np.average(diff_data1[2][-10:-1])
d3 = np.average(diff_data1[5][-10:-1])-np.average(diff_data1[4][-10:-1])
d4 = np.average(diff_data1[7][-10:-1])-np.average(diff_data1[6][-10:-1])
#y1 = [d1,d2,d3,d4]
y1 = [int(d1),int(d3),int(d4)]

end_t1 = var_data[3][-1]
end_t2 = var_data[7][-1]
end_t3 = var_data[11][-1]
end_t4 = var_data[15][-1]
#x = [end_t1,end_t2,end_t3,end_t4]
x = [int(end_t1),int(end_t3),int(end_t4)]

comp_var2 = ["FSNT"]
diff_data2 = []
for i in range(len(files)):
    for k in range(len(comp_var2)):
        f3 = nc.Dataset(path+files[i])
        diff_data2.append(f3.variables[comp_var2[k]])
d1_v2 = np.average(diff_data2[1][-10:-1])-np.average(diff_data2[0][-10:-1])
d2_v2 = np.average(diff_data2[3][-10:-1])-np.average(diff_data2[2][-10:-1])
d3_v2 = np.average(diff_data2[5][-10:-1])-np.average(diff_data2[4][-10:-1])
d4_v2 = np.average(diff_data2[7][-10:-1])-np.average(diff_data2[6][-10:-1])
#y2 = [d1_v2,d2_v2,d3_v2,d4_v2]
y2 = [d1_v2,d3_v2,d4_v2]

comp_var3 = ["FSNS"]
diff_data3 = []
for i in range(len(files)):
    for k in range(len(comp_var3)):
        f4 = nc.Dataset(path+files[i])
        diff_data3.append(f4.variables[comp_var3[k]])
d1_v3 = np.average(diff_data3[1][-10:-1])-np.average(diff_data3[0][-10:-1])
d2_v3 = np.average(diff_data3[3][-10:-1])-np.average(diff_data3[2][-10:-1])
d3_v3 = np.average(diff_data3[5][-10:-1])-np.average(diff_data3[4][-10:-1])
d4_v3 = np.average(diff_data3[7][-10:-1])-np.average(diff_data3[6][-10:-1])
#y3 = [d1_v3,d2_v3,d3_v3,d4_v3]
y3 = [d1_v3,d3_v3,d4_v3]

comp_var4 = ["FLNS"]
diff_data4 = []
for i in range(len(files)):
    for k in range(len(comp_var4)):
        f5 = nc.Dataset(path+files[i])
        diff_data4.append(f5.variables[comp_var4[k]])
d1_v4 = np.average(diff_data4[1][-10:-1])-np.average(diff_data4[0][-10:-1])
d2_v4 = np.average(diff_data4[3][-10:-1])-np.average(diff_data4[2][-10:-1])
d3_v4 = np.average(diff_data4[5][-10:-1])-np.average(diff_data4[4][-10:-1])
d4_v4 = np.average(diff_data4[7][-10:-1])-np.average(diff_data4[6][-10:-1])
#y4 = [d1_v4,d2_v4,d3_v4,d4_v4]
y4 = [d1_v4,d3_v4,d4_v4]

comp_var6 = ["SWCF"]
diff_data6 = []
for i in range(len(files)):
    for k in range(len(comp_var6)):
        f7 = nc.Dataset(path+files[i])
        diff_data6.append(f7.variables[comp_var6[k]])
d1_v6 = np.average(diff_data6[1][-10:-1])-np.average(diff_data6[0][-10:-1])
d2_v6 = np.average(diff_data6[3][-10:-1])-np.average(diff_data6[2][-10:-1])
d3_v6 = np.average(diff_data6[5][-10:-1])-np.average(diff_data6[4][-10:-1])
d4_v6 = np.average(diff_data6[7][-10:-1])-np.average(diff_data6[6][-10:-1])
#y6 = [d1_v6,d2_v6,d3_v6,d4_v6]
y6 = [d1_v6,d3_v6,d4_v6]

comp_var7 = ["LWCF"]
diff_data7 = []
for i in range(len(files)):
    for k in range(len(comp_var7)):
        f8 = nc.Dataset(path+files[i])
        diff_data7.append(f8.variables[comp_var7[k]])
d1_v7 = -1*(np.average(diff_data7[1][-10:-1])-np.average(diff_data7[0][-10:-1]))
d2_v7 = -1*(np.average(diff_data7[3][-10:-1])-np.average(diff_data7[2][-10:-1]))
d3_v7 = -1*(np.average(diff_data7[5][-10:-1])-np.average(diff_data7[4][-10:-1]))
d4_v7 = -1*(np.average(diff_data7[7][-10:-1])-np.average(diff_data7[6][-10:-1]))
#y7 = [d1_v7,d2_v7,d3_v7,d4_v7]
y7 = [d1_v7,d3_v7,d4_v7]

### linear regression calculations
# Generated linear fit
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x,y1)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x,y2)
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(x,y3)
slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(x,y4)
slope6, intercept6, r_value6, p_value6, std_err6 = stats.linregress(x,y6)
slope7, intercept7, r_value7, p_value7, std_err7 = stats.linregress(x,y7)

fig4 = plt.figure(figsize=(8,4))
ax = fig4.add_subplot(111)
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Change in varaibles (Modified-Contorl)")
ax.scatter(x,y1,marker="x",color="#7D3C98")
ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y1, 1))(np.unique(x)), color = "#7D3C98")
#ax.scatter(x,y1,marker="o",color=['#27AE60','#3498DB','#9B59B6','#E74C3C'])
ax.scatter(x,y2,marker="x",color="#D4AC0D")
ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y2, 1))(np.unique(x)), color = "#D4AC0D")
ax.scatter(x,y3,marker="x",color="#F39C12")
ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y3, 1))(np.unique(x)), color = "#F39C12")
ax.scatter(x,y4,marker="x",color="#52BE80")
ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y4, 1))(np.unique(x)), color = "#52BE80")
ax.scatter(x,y6,marker="x",color="#E74C3C")
ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y6, 1))(np.unique(x)), color = "#E74C3C")
ax.scatter(x,y7,marker="x",color="#3498DB")
ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y7, 1))(np.unique(x)), color = "#3498DB")
ax.set_ylim([-19,-5])
ax.text(288,-18.5,"Pre-Industrial",color="k")
ax.text(290.3,-18.5,"Modern 400",color='k')
ax.text(296.1,-18.5,"Modern 800",color='k')
ax.legend([Line2D([0],[0],color='#7D3C98',lw=1,marker="x"),
            Line2D([0],[0],color='#D4AC0D',lw=1,marker="x"),
            Line2D([0],[0],color='#F39C12',lw=1,marker="x"),
            Line2D([0],[0],color='#52BE80',lw=1,marker="x"),
            Line2D([0],[0],color='#E74C3C',lw=1,marker="x"),
            Line2D([0],[0],color='#3498DB',lw=1,marker="x")],
           ["FLNT","FSNT","FSNS","FLNS","SWCF","LWCF"],loc='center left')

# Table of values
fig6 = plt.figure(figsize=(8,4))
ax = fig6.add_subplot(111)
plt.table(cellText = [[round(slope1,3), round(intercept1,3), round(r_value1,3), round(p_value1,3), round(std_err1,3)],
                      [round(slope2,3), round(intercept2,3), round(r_value2,3), round(p_value2,3), round(std_err2,3)],
                      [round(slope3,3), round(intercept3,3), round(r_value3,3), round(p_value3,3), round(std_err3,3)],
                      [round(slope4,3), round(intercept4,3), round(r_value4,3), round(p_value4,3), round(std_err4,3)],
                      [round(slope6,3), round(intercept6,3), round(r_value6,3), round(p_value6,3), round(std_err6,3)],
                      [round(slope7,3), round(intercept7,3), round(r_value7,3), round(p_value7,3), round(std_err7,3)]],
          rowLabels = ["FLNT","FSNT","FSNS","FLNS","SWCF","LWCF"],
          colLabels = ["Slope","Intercept","R-Val","P-Val","STD Error"],
          loc = 'center')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
for pos in ['right','top','bottom','left']:
    plt.gca().spines[pos].set_visible(False)

##ax.legend([Line2D([0],[0],color='k',lw=0,marker="o"),
##            Line2D([0],[0],color='k',lw=0,marker="^"),
##            Line2D([0],[0],color='k',lw=0,marker="x"),
##            Line2D([0],[0],color='k',lw=0,marker="s"),
##            Line2D([0],[0],color='k',lw=0,marker="D"),
##            Line2D([0],[0],color='k',lw=0,marker="P"),
##            Line2D([0],[0],color='#27AE60',lw=5),
##            Line2D([0],[0],color='#3498DB',lw=5),
##            Line2D([0],[0],color='#9B59B6',lw=5),
##            Line2D([0],[0],color='#E74C3C',lw=5)],
##           ["FLNT","FSNT","FSNS","FLNS","SWCF","LWCF",
##            "Pre-industrial (284.7 ppm)","Modern (367 ppm)","Modern (400 ppm)","Modern (800 ppm)"])


# FIGURE FIVE: temperature difference
comp_var5 = ["TS"]
diff_data5 = []
for i in range(len(files)):
    for k in range(len(comp_var5)):
        f6 = nc.Dataset(path+files[i])
        diff_data5.append(f6.variables[comp_var5[k]])
d1_v5 = np.average(diff_data5[1][-10:-1])-np.average(diff_data5[0][-10:-1])
d2_v5 = np.average(diff_data5[3][-10:-1])-np.average(diff_data5[2][-10:-1])
d3_v5 = np.average(diff_data5[5][-10:-1])-np.average(diff_data5[4][-10:-1])
d4_v5 = np.average(diff_data5[7][-10:-1])-np.average(diff_data5[6][-10:-1])
#y5 = [d1_v5,d2_v5,d3_v5,d4_v5]
y5 = [d1_v5,d3_v5,d4_v5]

fig5 = plt.figure(figsize=(5,3))
plt.xlabel("Temperature [K]")
plt.ylabel("Change in Temperature (Modified-Contorl)")
plt.scatter(x,y5,marker="o",color=['#27AE60','#9B59B6','#E74C3C'])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y5, 1))(np.unique(x)), color = "k")
plt.legend([Line2D([0],[0],color='#27AE60',lw=0,marker="o"),
            Line2D([0],[0],color='#9B59B6',lw=0,marker="o"),
            Line2D([0],[0],color='#E74C3C',lw=0,marker="o")],
           ["Pre-industrial (284.7 ppm)","Modern (400 ppm)","Modern (800 ppm)"])


plt.show()
f.close()
f2.close()
f3.close()
f4.close()
f6.close()
                                    
