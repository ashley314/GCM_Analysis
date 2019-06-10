## Plot all varaibles from global averages ##
## Ashley Dicks ##

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib.lines import Line2D
import pandas as pd

# USER DEFINE: set path for location of data
path = "/Users/ash/Documents/Gradschool/Research/Miocene/Data/"

# USER DEFINE: output to be analyzed
names = ['E.Mod_800_C5_CNTL','E.Mod_800_C5_Kor1','E.Mod_800_C5_fin2','E.Mod_800_C5_TeoKor_Dem1SLF']

# USER DEFINE: variables to analyzed (SST?)
vari = ['RESSURF','RESTOM','TS','SOLIN',
        'CLDHGH','CLDLOW','CLDMED','CLDTOT',
        'FLNS','FLNSC','FLNT','FLNTC',
        'FSDS', 'FSDSC','FSNSC','FSNTC','LHFLX',
        'LWCF','PRECT','PS','PSL','QFLX','SHFLX','SNOWHLND',
        'SWCF','T850','TGCLDLWP','TGCLDIWP',
        'AODVIS','TREFHT']

# read in variables from files
files = [names[0]+'.0-100.global_av_all.nc',
         names[1]+'.0-100.global_av_all.nc',
         names[2]+'.0-100.global_av_all.nc',
         names[3]+'.0-30.global_av_all.nc']
var_data = []
for i in range(len(files)):
    for k in range(len(vari)):
        f = nc.Dataset(path+files[i])
        var_data.append(f.variables[vari[k]])

# time for files (need to change depending on file name ie for B cases)
year = [t for t in range(1,len(var_data[0])+1)]


## PLOTTING ##

# colors for each file
color_options = ['#27AE60','#3498DB','#9B59B6','#E74C3C'] # four options for now, can add more
colors = []
[colors.append([color_options[c] for n in range(len(year))]) for c in range(len(files))]

# legend
legend_type = [Line2D([0],[0],color='#27AE60',lw=1),
               Line2D([0],[0],color='#3498DB',lw=1),
               Line2D([0],[0],color='#9B59B6',lw=1),
               Line2D([0],[0],color='#E74C3C',lw=1)]
legend_names = names

# variable time series plots
for var in range(len(vari)):
    fig = plt.figure(figsize=(10,7))
    plt.title(vari[var]+" Time Series")
    plt.xlabel("Time [model year]")
    plt.ylabel(vari[var])
    plt.plot(var_data[var],color=color_options[0])
    plt.plot(var_data[var+len(vari)],color=color_options[1])
    plt.plot(var_data[var+2*len(vari)],color=color_options[2])
    plt.plot(var_data[var+3*len(vari)],color=color_options[3])
    plt.legend(legend_type,legend_names)


## TABLE ##

# annual means (last 10 years)
columns_n = list(names)
rows = vari
# calculate mean
data_mean = []
for v in range(len(vari)):
    m1 = np.mean(var_data[v][-10:])
    m2 = np.mean(var_data[v+len(vari)][-10:])
    m3 = np.mean(var_data[v+2*len(vari)][-10:])
    m4 = np.mean(var_data[v+3*len(vari)][-10:])
    means = [m1,m2,m3,m4]
    data_mean.append(means)
# put in dataframe
df = pd.DataFrame(data_mean,columns=columns_n)
df2 = df.round(2) 
# plot
fig = plt.figure(figsize=(14,7))
fig.patch.set_visible(False)
plt.axis('off')
plt.grid('off')
table = plt.table(cellText=df2.values, rowLabels=rows, colLabels=df2.columns,
                 loc='center',colWidths=[0.2]*4)
table.auto_set_font_size(False)
table.set_fontsize(10)
fig.tight_layout()
plt.title("Table of Global Means",fontsize=12)
plt.subplots_adjust(top=0.95)
tablefile = 'EMod800Table.png'
plt.savefig('/Users/ash/Documents/Gradschool/Research/Miocene/Figures/'+tablefile)

plt.show()
f.close()
    


