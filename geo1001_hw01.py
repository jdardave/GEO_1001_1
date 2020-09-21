#-- GEO1001.2020--hw01
#-- [Ioannis Dardavesis] 
#-- [5372666]
import sys
import numpy as np																	
import matplotlib.pyplot as plt														
import pandas as pd																	
import numpy as np
import scipy.stats as stats
import seaborn as sns
import xlrd
import statistics
import xlsxwriter
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('figure', max_open_warning = 0)

#Import heat data in csv type
df_A = pd.read_csv("HEAT-A_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)],header=None)
df_B = pd.read_csv("HEAT-B_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)],header=None)
df_C = pd.read_csv("HEAT-C_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)],header=None)
df_D = pd.read_csv("HEAT-D_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)],header=None)
df_E = pd.read_csv("HEAT-E_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)],header=None)

#Computation of mean, standard deviation & variance for the 5 files

mean_Heat_A=df_A.mean()
var_Heat_A=df_A.var()
std_Heat_A=df_A.std()
statistics_Heat_A=([[mean_Heat_A],[var_Heat_A],[std_Heat_A]])
print (statistics_Heat_A)

mean_Heat_B=df_B.mean()
var_Heat_B=df_B.var()
std_Heat_B=df_B.std()
statistics_Heat_B=([[mean_Heat_B],[var_Heat_B],[std_Heat_B]])
print (statistics_Heat_B)

mean_Heat_C=df_C.mean()
var_Heat_C=df_C.var()
std_Heat_C=df_C.std()
statistics_Heat_C=([[mean_Heat_C],[var_Heat_C],[std_Heat_C]])
print (statistics_Heat_C)

mean_Heat_D=df_D.mean()
var_Heat_D=df_D.var()
std_Heat_D=df_D.std()
statistics_Heat_D=([[mean_Heat_D],[var_Heat_D],[std_Heat_D]])
print (statistics_Heat_D)

mean_Heat_E=df_E.mean()
var_Heat_E=df_E.var()
std_Heat_E=df_E.std()
statistics_Heat_E=([[mean_Heat_E],[var_Heat_E],[std_Heat_E]])
print (statistics_Heat_E)

np.savetxt("Sensors_Stats.csv", [mean_Heat_A,var_Heat_A,std_Heat_A,mean_Heat_B,var_Heat_B,std_Heat_B,
mean_Heat_C,var_Heat_C,std_Heat_C,mean_Heat_D,var_Heat_D,std_Heat_D,mean_Heat_E,var_Heat_E,std_Heat_E,], delimiter=",")

#Temperature histograms
temperature_Heat_A=df_A[4]
temperature_Heat_B=df_B[4]
temperature_Heat_C=df_C[4]
temperature_Heat_D=df_D[4]
temperature_Heat_E=df_E[4]

fig1=plt.figure(1)
plt.hist([temperature_Heat_A,temperature_Heat_B,temperature_Heat_C,temperature_Heat_D,temperature_Heat_E],bins=50,
label=['HEAT A','HEAT B','HEAT C','HEAT D','HEAT E'])
plt.title("Temperature Histogram with 50 bins")
plt.xlabel('Temperature (celsius)')
plt.ylabel('Frequency') 
plt.legend(loc='upper right')

fig2=plt.figure(2)
plt.hist([temperature_Heat_A,temperature_Heat_B,temperature_Heat_C,temperature_Heat_D,temperature_Heat_E],bins=5,
label=['HEAT A','HEAT B','HEAT C','HEAT D','HEAT E'])
plt.title("Temperature Histogram with 5 bins")
plt.xlabel('Temperature (celsius)')
plt.ylabel('Frequency') 
plt.legend(loc='upper right')

# Frequency polygons plot of temperature values for the 5 sensors 
fig3=plt.figure(3)

y_A,edges = np.histogram(temperature_Heat_A,bins=27)
y_B,edges= np.histogram(temperature_Heat_B,bins=27)
y_C,edges = np.histogram(temperature_Heat_C,bins=27)
y_D,edges= np.histogram(temperature_Heat_D,bins=27)
y_E,edges = np.histogram(temperature_Heat_E,bins=27)
centers = 0.5*(edges[1:]+ edges[:-1])
plt.plot(centers,y_A,'-*',label=('Sensor A'))
plt.plot(centers,y_B,'-*',label=('Sensor B'))
plt.plot(centers,y_C,'-*',label=('Sensor C'))
plt.plot(centers,y_D,'-*',label=('Sensor D'))
plt.plot(centers,y_E,'-*',label=('Sensor E'))
plt.legend(loc='upper right')
plt.xlabel('Temperature (celsius)')
plt.ylabel('Cumulative Frequency')
plt.title('Frequency polygons of 5 sensors')

#Box plots for wind speed in m/s
WS_A=df_A[1]
WS_B=df_B[1]
WS_C=df_C[1]
WS_D=df_D[1]
WS_E=df_E[1]

#Box plots for wind direction
WD_A=df_A[0]
WD_B=df_B[0]
WD_C=df_C[0]
WD_D=df_D[0]
WD_E=df_E[0]

fig4=plt.figure(4)
plt.boxplot([WS_A,WS_B,WS_C,WS_D,WS_E],showmeans=True,labels=['Sensor A','Sensor B','Sensor C','Sensor D','Sensor E'])
plt.ylabel("Wind Speed (m/s)")
plt.title("Boxplots of Wind Speed")

fig5=plt.figure(5)
plt.boxplot([WD_A,WD_B,WD_C,WD_D,WD_E],showmeans=True,labels=['Sensor A','Sensor B','Sensor C','Sensor D','Sensor E'])
plt.ylabel("Wind Direction (degrees)")
plt.title("Boxplots of Wind Direction")

fig6=plt.figure(6)
plt.boxplot([temperature_Heat_A,temperature_Heat_B,temperature_Heat_C,temperature_Heat_D,temperature_Heat_E],
showmeans=True,labels=['Sensor A','Sensor B','Sensor C','Sensor D','Sensor E'])
plt.ylabel("Temperature (celsius)")
plt.title("Boxplots of Temperature")

# PART A2

#compute and plot pmf
def pmf(sample):
	c = sample.value_counts()
	p = c/len(sample)
	return p

df_pmf_A = pmf(temperature_Heat_A)
c_A = df_pmf_A.sort_index()

df_pmf_B = pmf(temperature_Heat_B)
c_B = df_pmf_B.sort_index()

df_pmf_C = pmf(temperature_Heat_C)
c_C = df_pmf_C.sort_index()

df_pmf_D = pmf(temperature_Heat_D)
c_D = df_pmf_D.sort_index()

df_pmf_E = pmf(temperature_Heat_E)
c_E = df_pmf_E.sort_index()

fig7, axs = plt.subplots(5,sharex=True,sharey=True)
fig7.suptitle("PMF for temperature values (Sensors A-E)")
axs[2].set_ylabel("Probability")
axs[0].bar(c_A.index,c_A)
axs[1].bar(c_B.index,c_B)
axs[2].bar(c_C.index,c_C)
axs[3].bar(c_D.index,c_D)
axs[4].bar(c_E.index,c_E)
axs[4].set_xlabel('Temperature (celsius)')

# # plot pdf

fig8, axs = plt.subplots(5,sharex=True,sharey=True)
fig8.suptitle("PDF for temperature values (Sensors A-E)")
sns.distplot([temperature_Heat_A.astype(float)], ax=axs[0])
sns.distplot([temperature_Heat_B.astype(float)], ax=axs[1])
sns.distplot([temperature_Heat_C.astype(float)], ax=axs[2])
sns.distplot([temperature_Heat_D.astype(float)], ax=axs[3])
sns.distplot([temperature_Heat_E.astype(float)], ax=axs[4])
axs[0].hist([temperature_Heat_A.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
axs[1].hist([temperature_Heat_B.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
axs[2].hist([temperature_Heat_C.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
axs[3].hist([temperature_Heat_D.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
axs[4].hist([temperature_Heat_E.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
axs[2].set_ylabel('Probability Density')
axs[4].set_xlabel('Temperature (celsius)')

# # plot cdf
fig9, axs = plt.subplots(5,sharex=True,sharey=True)
fig9.suptitle("CDF for temperature values (Sensors A-E)")
a1=axs[0].hist([temperature_Heat_A.astype(float)],bins=27, cumulative=True,alpha=0.7, rwidth=0.85)
a2=axs[1].hist([temperature_Heat_B.astype(float)],bins=27,cumulative=True,alpha=0.7, rwidth=0.85)
a3=axs[2].hist([temperature_Heat_C.astype(float)],bins=27, cumulative=True,alpha=0.7, rwidth=0.85)
a4=axs[3].hist([temperature_Heat_D.astype(float)],bins=27, cumulative=True,alpha=0.7, rwidth=0.85)
a5=axs[4].hist([temperature_Heat_E.astype(float)],bins=27, cumulative=True,alpha=0.7, rwidth=0.85)
axs[0].plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
axs[1].plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
axs[2].plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
axs[3].plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
axs[4].plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')
axs[2].set_ylabel('CDF')
axs[4].set_xlabel('Temperature (celsius)')

#Plot PDF & KDE

#Sensor A
fig10, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF of Wind Speed (Sensor A)')
axs[0].set_ylabel('Probability Density')
axs[0].hist([WS_A.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WS_A.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WS_A)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor A)')
axs[1].set_ylabel('Density')
axs[1].set_xlabel('Wind Speed (m/s)')

#Sensor B
fig11, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF of Wind Speed (Sensor B)')
axs[0].set_ylabel('Probability Density')
axs[0].hist([WS_B.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WS_B.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WS_B)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor B)')
axs[1].set_ylabel('Density')
axs[1].set_xlabel('Wind Speed (m/s)')

#Sensor C
fig12, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF of Wind Speed (Sensor C)')
axs[0].set_ylabel('Probability Density')
axs[0].hist([WS_C.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WS_C.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WS_C)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor C)')
axs[1].set_ylabel('Density')
axs[1].set_xlabel('Wind Speed (m/s)')

#Sensor D
fig13, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF of Wind Speed (Sensor D)')
axs[0].set_ylabel('Probability Density')
axs[0].hist([WS_D.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WS_D.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WS_D)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor C)')
axs[1].set_ylabel('Density')
axs[1].set_xlabel('Wind Speed (m/s)')

#Sensor E
fig14, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF of Wind Speed (Sensor E)')
axs[0].set_ylabel('Probability Density')
axs[0].hist([WS_E.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WS_E.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WS_E)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor C)')
axs[1].set_ylabel('Density')
axs[1].set_xlabel('Wind Speed (m/s)')

#A3
#Compute correlation
#Import Crosswind speed variables

Crosswind_A=df_A[2]
Crosswind_B=df_B[2]
Crosswind_C=df_C[2]
Crosswind_D=df_D[2]
Crosswind_E=df_E[2]

#Import Wet Bulb Global Temperature variables
WBGT_A=df_A[16]
WBGT_B=df_B[16]
WBGT_C=df_C[16]
WBGT_D=df_D[16]
WBGT_E=df_E[16]


#Heat Map A-Heat Map B

pcoef_temp_AB = stats.pearsonr(temperature_Heat_A,temperature_Heat_B)
prcoef_temp_AB = stats.spearmanr(temperature_Heat_A,temperature_Heat_B)

pcoef_Crosswind_AB = stats.pearsonr(Crosswind_A,Crosswind_B)
prcoef_Crosswind_AB = stats.spearmanr(Crosswind_A,Crosswind_B)

pcoef_WBGT_AB = stats.pearsonr(WBGT_A,WBGT_B)
prcoef_WBGT_AB = stats.spearmanr(WBGT_A,WBGT_B)


# #Heat Map A-Heat Map C

# #Interpolate to equal size samples
temperature_Heat_A_interp = np.interp(np.linspace(0,len(temperature_Heat_C),len(temperature_Heat_C)),
np.linspace(0,len(temperature_Heat_A),len(temperature_Heat_A)),temperature_Heat_A)

Crosswind_A_interp = np.interp(np.linspace(0,len(Crosswind_C),len(Crosswind_C)),
np.linspace(0,len(Crosswind_A),len(Crosswind_A)),Crosswind_A)

WBGT_A_interp = np.interp(np.linspace(0,len(WBGT_C),len(WBGT_C)),
np.linspace(0,len(WBGT_A),len(WBGT_A)),WBGT_A)

pcoef_temp_AC = stats.pearsonr(temperature_Heat_A_interp,temperature_Heat_C)
prcoef_temp_AC = stats.spearmanr(temperature_Heat_A_interp,temperature_Heat_C)

pcoef_Crosswind_AC = stats.pearsonr(Crosswind_A_interp,Crosswind_C)
prcoef_Crosswind_AC = stats.spearmanr(Crosswind_A_interp,Crosswind_C)

pcoef_WBGT_AC = stats.pearsonr(WBGT_A_interp,WBGT_C)
prcoef_WBGT_AC = stats.spearmanr(WBGT_A_interp,WBGT_C)

# #Heat Map A-Heat Map D
# #Interpolate to equal size samples

temperature_Heat_A_interp = np.interp(np.linspace(0,len(temperature_Heat_D),len(temperature_Heat_D)),
np.linspace(0,len(temperature_Heat_A),len(temperature_Heat_A)),temperature_Heat_A)

Crosswind_A_interp = np.interp(np.linspace(0,len(Crosswind_D),len(Crosswind_D)),
np.linspace(0,len(Crosswind_A),len(Crosswind_A)),Crosswind_A)

WBGT_A_interp = np.interp(np.linspace(0,len(WBGT_D),len(WBGT_D)),
np.linspace(0,len(WBGT_A),len(WBGT_A)),WBGT_A)

pcoef_temp_AD = stats.pearsonr(temperature_Heat_A_interp,temperature_Heat_D)
prcoef_temp_AD = stats.spearmanr(temperature_Heat_A_interp,temperature_Heat_D)

pcoef_Crosswind_AD = stats.pearsonr(Crosswind_A_interp,Crosswind_D)
prcoef_Crosswind_AD = stats.spearmanr(Crosswind_A_interp,Crosswind_D)

pcoef_WBGT_AD = stats.pearsonr(WBGT_A_interp,WBGT_D)
prcoef_WBGT_AD = stats.spearmanr(WBGT_A_interp,WBGT_D)


# #Heat Map A-Heat Map E
# #Interpolate to equal size samples

temperature_Heat_A_interp = np.interp(np.linspace(0,len(temperature_Heat_E),len(temperature_Heat_E)),
np.linspace(0,len(temperature_Heat_A),len(temperature_Heat_A)),temperature_Heat_A)

Crosswind_A_interp = np.interp(np.linspace(0,len(Crosswind_E),len(Crosswind_E)),
np.linspace(0,len(Crosswind_A),len(Crosswind_A)),Crosswind_A)

WBGT_A_interp = np.interp(np.linspace(0,len(WBGT_E),len(WBGT_E)),
np.linspace(0,len(WBGT_A),len(WBGT_A)),WBGT_A)

pcoef_temp_AE = stats.pearsonr(temperature_Heat_A_interp,temperature_Heat_E)
prcoef_temp_AE = stats.spearmanr(temperature_Heat_A_interp,temperature_Heat_E)

pcoef_Crosswind_AE = stats.pearsonr(Crosswind_A_interp,Crosswind_E)
prcoef_Crosswind_AE = stats.spearmanr(Crosswind_A_interp,Crosswind_E)

pcoef_WBGT_AE = stats.pearsonr(WBGT_A_interp,WBGT_E)
prcoef_WBGT_AE = stats.spearmanr(WBGT_A_interp,WBGT_E)


# #Heat Map B-Heat Map C
# #Interpolate to equal size samples

temperature_Heat_B_interp = np.interp(np.linspace(0,len(temperature_Heat_C),len(temperature_Heat_C)),
np.linspace(0,len(temperature_Heat_B),len(temperature_Heat_B)),temperature_Heat_B)

Crosswind_B_interp = np.interp(np.linspace(0,len(Crosswind_C),len(Crosswind_C)),
np.linspace(0,len(Crosswind_B),len(Crosswind_B)),Crosswind_B)

WBGT_B_interp = np.interp(np.linspace(0,len(WBGT_C),len(WBGT_C)),
np.linspace(0,len(WBGT_B),len(WBGT_B)),WBGT_B)

pcoef_temp_BC = stats.pearsonr(temperature_Heat_B_interp,temperature_Heat_C)
prcoef_temp_BC = stats.spearmanr(temperature_Heat_B_interp,temperature_Heat_C)

pcoef_Crosswind_BC = stats.pearsonr(Crosswind_B_interp,Crosswind_C)
prcoef_Crosswind_BC = stats.spearmanr(Crosswind_B_interp,Crosswind_C)

pcoef_WBGT_BC = stats.pearsonr(WBGT_B_interp,WBGT_C)
prcoef_WBGT_BC = stats.spearmanr(WBGT_B_interp,WBGT_C)


# #Heat Map B-Heat Map D
# #Interpolate to equal size samples

temperature_Heat_B_interp = np.interp(np.linspace(0,len(temperature_Heat_D),len(temperature_Heat_D)),
np.linspace(0,len(temperature_Heat_B),len(temperature_Heat_B)),temperature_Heat_B)

Crosswind_B_interp = np.interp(np.linspace(0,len(Crosswind_D),len(Crosswind_D)),
np.linspace(0,len(Crosswind_B),len(Crosswind_B)),Crosswind_B)

WBGT_B_interp = np.interp(np.linspace(0,len(WBGT_D),len(WBGT_D)),
np.linspace(0,len(WBGT_B),len(WBGT_B)),WBGT_B)

pcoef_temp_BD = stats.pearsonr(temperature_Heat_B_interp,temperature_Heat_D)
prcoef_temp_BD = stats.spearmanr(temperature_Heat_B_interp,temperature_Heat_D)

pcoef_Crosswind_BD = stats.pearsonr(Crosswind_B_interp,Crosswind_D)
prcoef_Crosswind_BD = stats.spearmanr(Crosswind_B_interp,Crosswind_D)

pcoef_WBGT_BD = stats.pearsonr(WBGT_B_interp,WBGT_D)
prcoef_WBGT_BD = stats.spearmanr(WBGT_B_interp,WBGT_D)


# #Heat Map B-Heat Map E
# #Interpolate to equal size samples

temperature_Heat_B_interp = np.interp(np.linspace(0,len(temperature_Heat_E),len(temperature_Heat_E)),
np.linspace(0,len(temperature_Heat_B),len(temperature_Heat_B)),temperature_Heat_B)

Crosswind_B_interp = np.interp(np.linspace(0,len(Crosswind_E),len(Crosswind_E)),
np.linspace(0,len(Crosswind_B),len(Crosswind_B)),Crosswind_B)

WBGT_B_interp = np.interp(np.linspace(0,len(WBGT_E),len(WBGT_E)),
np.linspace(0,len(WBGT_B),len(WBGT_B)),WBGT_B)

pcoef_temp_BE = stats.pearsonr(temperature_Heat_B_interp,temperature_Heat_E)
prcoef_temp_BE = stats.spearmanr(temperature_Heat_B_interp,temperature_Heat_E)

pcoef_Crosswind_BE = stats.pearsonr(Crosswind_B_interp,Crosswind_E)
prcoef_Crosswind_BE = stats.spearmanr(Crosswind_B_interp,Crosswind_E)

pcoef_WBGT_BE = stats.pearsonr(WBGT_B_interp,WBGT_E)
prcoef_WBGT_BE = stats.spearmanr(WBGT_B_interp,WBGT_E)


# #Heat Map C-Heat Map D
# #Interpolate to equal size samples

temperature_Heat_C_interp = np.interp(np.linspace(0,len(temperature_Heat_D),len(temperature_Heat_D)),
np.linspace(0,len(temperature_Heat_C),len(temperature_Heat_C)),temperature_Heat_C)

Crosswind_C_interp = np.interp(np.linspace(0,len(Crosswind_D),len(Crosswind_D)),
np.linspace(0,len(Crosswind_C),len(Crosswind_C)),Crosswind_C)

WBGT_C_interp = np.interp(np.linspace(0,len(WBGT_D),len(WBGT_D)),
np.linspace(0,len(WBGT_C),len(WBGT_C)),WBGT_C)

pcoef_temp_CD = stats.pearsonr(temperature_Heat_C_interp,temperature_Heat_D)
prcoef_temp_CD = stats.spearmanr(temperature_Heat_C_interp,temperature_Heat_D)

pcoef_Crosswind_CD = stats.pearsonr(Crosswind_C_interp,Crosswind_D)
prcoef_Crosswind_CD = stats.spearmanr(Crosswind_C_interp,Crosswind_D)

pcoef_WBGT_CD = stats.pearsonr(WBGT_C_interp,WBGT_D)
prcoef_WBGT_CD = stats.spearmanr(WBGT_C_interp,WBGT_D)


# #Heat Map C-Heat Map E
# #Interpolate to equal size samples

temperature_Heat_C_interp = np.interp(np.linspace(0,len(temperature_Heat_E),len(temperature_Heat_E)),
np.linspace(0,len(temperature_Heat_C),len(temperature_Heat_C)),temperature_Heat_C)

Crosswind_C_interp = np.interp(np.linspace(0,len(Crosswind_E),len(Crosswind_E)),
np.linspace(0,len(Crosswind_C),len(Crosswind_C)),Crosswind_C)

WBGT_C_interp = np.interp(np.linspace(0,len(WBGT_E),len(WBGT_E)),
np.linspace(0,len(WBGT_C),len(WBGT_C)),WBGT_C)

pcoef_temp_CE = stats.pearsonr(temperature_Heat_C_interp,temperature_Heat_E)
prcoef_temp_CE = stats.spearmanr(temperature_Heat_C_interp,temperature_Heat_E)

pcoef_Crosswind_CE = stats.pearsonr(Crosswind_C_interp,Crosswind_E)
prcoef_Crosswind_CE = stats.spearmanr(Crosswind_C_interp,Crosswind_E)

pcoef_WBGT_CE = stats.pearsonr(WBGT_C_interp,WBGT_E)
prcoef_WBGT_CE = stats.spearmanr(WBGT_C_interp,WBGT_E)


# #Heat Map D-Heat Map E
# #Interpolate to equal size samples

temperature_Heat_D_interp = np.interp(np.linspace(0,len(temperature_Heat_E),len(temperature_Heat_E)),
np.linspace(0,len(temperature_Heat_D),len(temperature_Heat_D)),temperature_Heat_D)

Crosswind_D_interp = np.interp(np.linspace(0,len(Crosswind_E),len(Crosswind_E)),
np.linspace(0,len(Crosswind_D),len(Crosswind_D)),Crosswind_D)

WBGT_D_interp = np.interp(np.linspace(0,len(WBGT_E),len(WBGT_E)),
np.linspace(0,len(WBGT_D),len(WBGT_D)),WBGT_D)

pcoef_temp_DE = stats.pearsonr(temperature_Heat_D_interp,temperature_Heat_E)
prcoef_temp_DE = stats.spearmanr(temperature_Heat_D_interp,temperature_Heat_E)

pcoef_Crosswind_DE = stats.pearsonr(Crosswind_D_interp,Crosswind_E)
prcoef_Crosswind_DE = stats.spearmanr(Crosswind_D_interp,Crosswind_E)

pcoef_WBGT_DE = stats.pearsonr(WBGT_D_interp,WBGT_E)
prcoef_WBGT_DE = stats.spearmanr(WBGT_D_interp,WBGT_E)


Workbook=xlsxwriter.Workbook("correlation.xlsx")
Worksheet=Workbook.add_worksheet()
Worksheet.write("B1","Pearson Coefficient")
Worksheet.write("C1","Spearman coefficient")
Worksheet.write("B2",pcoef_temp_AB[0])
Worksheet.write("B3",pcoef_Crosswind_AB[0])
Worksheet.write("B4",pcoef_WBGT_AB[0])

Worksheet.write("C2",prcoef_temp_AB[0])
Worksheet.write("C3",prcoef_Crosswind_AB[0])
Worksheet.write("C4",prcoef_WBGT_AB[0])

Worksheet.write("B5",pcoef_temp_AC[0])
Worksheet.write("B6",pcoef_Crosswind_AC[0])
Worksheet.write("B7",pcoef_WBGT_AC[0])

Worksheet.write("C5",prcoef_temp_AC[0])
Worksheet.write("C6",prcoef_Crosswind_AC[0])
Worksheet.write("C7",prcoef_WBGT_AC[0])

Worksheet.write("B8",pcoef_temp_AD[0])
Worksheet.write("B9",pcoef_Crosswind_AD[0])
Worksheet.write("B10",pcoef_WBGT_AD[0])

Worksheet.write("C8",prcoef_temp_AD[0])
Worksheet.write("C9",prcoef_Crosswind_AD[0])
Worksheet.write("C10",prcoef_WBGT_AD[0])

Worksheet.write("B11",pcoef_temp_AE[0])
Worksheet.write("B12",pcoef_Crosswind_AE[0])
Worksheet.write("B13",pcoef_WBGT_AE[0])

Worksheet.write("C11",prcoef_temp_AE[0])
Worksheet.write("C12",prcoef_Crosswind_AE[0])
Worksheet.write("C13",prcoef_WBGT_AE[0])

Worksheet.write("B14",pcoef_temp_BC[0])
Worksheet.write("B15",pcoef_Crosswind_BC[0])
Worksheet.write("B16",pcoef_WBGT_BC[0])

Worksheet.write("C14",prcoef_temp_BC[0])
Worksheet.write("C15",prcoef_Crosswind_BC[0])
Worksheet.write("C16",prcoef_WBGT_BC[0])

Worksheet.write("B17",pcoef_temp_BD[0])
Worksheet.write("B18",pcoef_Crosswind_BD[0])
Worksheet.write("B19",pcoef_WBGT_BD[0])

Worksheet.write("C17",prcoef_temp_BD[0])
Worksheet.write("C18",prcoef_Crosswind_BD[0])
Worksheet.write("C19",prcoef_WBGT_BD[0])

Worksheet.write("B20",pcoef_temp_BE[0])
Worksheet.write("B21",pcoef_Crosswind_BE[0])
Worksheet.write("B22",pcoef_WBGT_BE[0])

Worksheet.write("C20",prcoef_temp_BE[0])
Worksheet.write("C21",prcoef_Crosswind_BE[0])
Worksheet.write("C22",prcoef_WBGT_BE[0])

Worksheet.write("B23",pcoef_temp_CD[0])
Worksheet.write("B24",pcoef_Crosswind_CD[0])
Worksheet.write("B25",pcoef_WBGT_CD[0])

Worksheet.write("C23",prcoef_temp_CD[0])
Worksheet.write("C24",prcoef_Crosswind_CD[0])
Worksheet.write("C25",prcoef_WBGT_CD[0])

Worksheet.write("B26",pcoef_temp_CE[0])
Worksheet.write("B27",pcoef_Crosswind_CE[0])
Worksheet.write("B28",pcoef_WBGT_CE[0])

Worksheet.write("C26",prcoef_temp_CE[0])
Worksheet.write("C27",prcoef_Crosswind_CE[0])
Worksheet.write("C28",prcoef_WBGT_CE[0])

Worksheet.write("B29",pcoef_temp_DE[0])
Worksheet.write("B30",pcoef_Crosswind_DE[0])
Worksheet.write("B31",pcoef_WBGT_DE[0])

Worksheet.write("C29",prcoef_temp_DE[0])
Worksheet.write("C30",prcoef_Crosswind_DE[0])
Worksheet.write("C31",prcoef_WBGT_DE[0])
Workbook.close()


#Pearson and Spearman Coefficient for temperature

Pearson=[pcoef_temp_AB[0],pcoef_temp_AC[0],pcoef_temp_AD[0],pcoef_temp_AE[0],pcoef_temp_BC[0],pcoef_temp_BD[0],pcoef_temp_BE[0],
pcoef_temp_CD[0],pcoef_temp_CE[0],pcoef_temp_DE[0]]
Spearman=[prcoef_temp_AB[0],prcoef_temp_AC[0],prcoef_temp_AD[0],prcoef_temp_AE[0],prcoef_temp_BC[0],prcoef_temp_BD[0],prcoef_temp_BE[0],
prcoef_temp_CD[0],prcoef_temp_CE[0],prcoef_temp_DE[0]]

A1=[1,2,3,4,5,6,7,8,9,10]
Labels=["AB","AC","AD","AE","BC","BD","BE","CD","CE","DE"]

fig15= plt.figure(15)
plt.title("Temperature Correlations with Pearson and Spearman coefficient")
plt.xticks(A1,Labels)
Pearson_scatter=plt.scatter(A1,Pearson)
Spearman_scatter=plt.scatter(A1,Spearman)
plt.legend((Pearson_scatter,Spearman_scatter),('Pearson Coefficient','Spearman Coefficient'),loc='upper right')
plt.xlabel("Relationship between Sensors")
plt.ylabel("Correlation")

#Pearson and Spearman Coefficient for Crosswind
Pearson=[pcoef_Crosswind_AB[0],pcoef_Crosswind_AC[0],pcoef_Crosswind_AD[0],pcoef_Crosswind_AE[0],pcoef_Crosswind_BC[0],pcoef_Crosswind_BD[0],pcoef_Crosswind_BE[0],
pcoef_temp_CD[0],pcoef_temp_CE[0],pcoef_temp_DE[0]]
Spearman=[prcoef_Crosswind_AB[0],prcoef_Crosswind_AC[0],prcoef_Crosswind_AD[0],prcoef_Crosswind_AE[0],prcoef_Crosswind_BC[0],prcoef_Crosswind_BD[0],prcoef_Crosswind_BE[0],
prcoef_Crosswind_CD[0],prcoef_Crosswind_CE[0],prcoef_Crosswind_DE[0]]

fig16= plt.figure(16)
plt.title("Crosswind Correlations with Pearson and Spearman coefficient")
plt.xticks(A1,Labels)
Pearson_scatter=plt.scatter(A1,Pearson)
Spearman_scatter=plt.scatter(A1,Spearman)
plt.legend((Pearson_scatter,Spearman_scatter),('Pearson Coefficient','Spearman Coefficient'),loc='upper right')
plt.xlabel("Relationship between Sensors")
plt.ylabel("Correlation")

#Pearson and Spearman Coefficient for WBGT
Pearson=[pcoef_WBGT_AB[0],pcoef_WBGT_AC[0],pcoef_WBGT_AD[0],pcoef_WBGT_AE[0],pcoef_WBGT_BC[0],pcoef_WBGT_BD[0],pcoef_WBGT_BE[0],
pcoef_temp_CD[0],pcoef_temp_CE[0],pcoef_temp_DE[0]]
Spearman=[prcoef_WBGT_AB[0],prcoef_WBGT_AC[0],prcoef_WBGT_AD[0],prcoef_WBGT_AE[0],prcoef_WBGT_BC[0],prcoef_WBGT_BD[0],prcoef_WBGT_BE[0],
prcoef_WBGT_CD[0],prcoef_WBGT_CE[0],prcoef_WBGT_DE[0]]

fig17= plt.figure(17)
plt.title("WBGT Correlations with Pearson and Spearman coefficient")
plt.xticks(A1,Labels)
Pearson_scatter=plt.scatter(A1,Pearson)
Spearman_scatter=plt.scatter(A1,Spearman)
plt.legend((Pearson_scatter,Spearman_scatter),('Pearson Coefficient','Spearman Coefficient'),loc='upper right')
plt.xlabel("Relationship between Sensors")
plt.ylabel("Correlation")


#A4 

fig18, axs = plt.subplots(5,sharex=True,sharey=True)
fig18.suptitle("CDF for Wind Speed values")
a1=axs[0].hist([WS_A.astype(float)],bins=27, cumulative=True,alpha=0.7, rwidth=0.85)
a2=axs[1].hist([WS_B.astype(float)],bins=27,cumulative=True,alpha=0.7, rwidth=0.85)
a3=axs[2].hist([WS_C.astype(float)],bins=27, cumulative=True,alpha=0.7, rwidth=0.85)
a4=axs[3].hist([WS_D.astype(float)],bins=27, cumulative=True,alpha=0.7, rwidth=0.85)
a5=axs[4].hist([WS_E.astype(float)],bins=27, cumulative=True,alpha=0.7, rwidth=0.85)
axs[0].plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
axs[1].plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
axs[2].plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
axs[3].plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
axs[4].plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')
axs[2].set_ylabel('CDF')
axs[4].set_xlabel('Wind Speed (m/s)')


import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m, m+h

conf_temp_A=(mean_confidence_interval(temperature_Heat_A,confidence=0.95))
conf_temp_B=(mean_confidence_interval(temperature_Heat_B,confidence=0.95))
conf_temp_C=(mean_confidence_interval(temperature_Heat_C,confidence=0.95))
conf_temp_D=(mean_confidence_interval(temperature_Heat_D,confidence=0.95))
conf_temp_E=(mean_confidence_interval(temperature_Heat_E,confidence=0.95))

conf_WS_A=(mean_confidence_interval(WS_A,confidence=0.95))
conf_WS_B=(mean_confidence_interval(WS_B,confidence=0.95))
conf_WS_C=(mean_confidence_interval(WS_C,confidence=0.95))
conf_WS_D=(mean_confidence_interval(WS_D,confidence=0.95))
conf_WS_E=(mean_confidence_interval(WS_E,confidence=0.95))

Workbook=xlsxwriter.Workbook("confidence_intervals.csv")
Worksheet=Workbook.add_worksheet()
Worksheet.write("A1","m-h")
Worksheet.write("B1","m")
Worksheet.write("C1","m+h")
Worksheet.write("A2",conf_temp_A[0])
Worksheet.write("B2",conf_temp_A[1])
Worksheet.write("C2",conf_temp_A[2])
Worksheet.write("A3",conf_temp_B[0])
Worksheet.write("B3",conf_temp_B[1])
Worksheet.write("C3",conf_temp_B[2])
Worksheet.write("A4",conf_temp_C[0])
Worksheet.write("B4",conf_temp_C[1])
Worksheet.write("C4",conf_temp_C[2])
Worksheet.write("A5",conf_temp_D[0])
Worksheet.write("B5",conf_temp_D[1])
Worksheet.write("C5",conf_temp_D[2])
Worksheet.write("A6",conf_temp_E[0])
Worksheet.write("B6",conf_temp_E[1])
Worksheet.write("C6",conf_temp_E[2])

Worksheet.write("A7",conf_WS_A[0])
Worksheet.write("B7",conf_WS_A[1])
Worksheet.write("C7",conf_WS_A[2])
Worksheet.write("A8",conf_WS_B[0])
Worksheet.write("B8",conf_WS_B[1])
Worksheet.write("C8",conf_WS_B[2])
Worksheet.write("A9",conf_WS_C[0])
Worksheet.write("B9",conf_WS_C[1])
Worksheet.write("C9",conf_WS_C[2])
Worksheet.write("A10",conf_WS_D[0])
Worksheet.write("B10",conf_WS_D[1])
Worksheet.write("C10",conf_WS_D[2])
Worksheet.write("A11",conf_WS_E[0])
Worksheet.write("B11",conf_WS_E[1])
Worksheet.write("C11",conf_WS_E[2])
Workbook.close()


#T testing and p values

t_ED_temp,p_ED_temp = stats.ttest_ind(temperature_Heat_E,temperature_Heat_D)
t_DC_temp,p_DC_temp = stats.ttest_ind(temperature_Heat_D,temperature_Heat_C)
t_CB_temp,p_CB_temp = stats.ttest_ind(temperature_Heat_C,temperature_Heat_B)
t_BA_temp,p_BA_temp = stats.ttest_ind(temperature_Heat_B,temperature_Heat_A)

t_ED_WS,p_ED_WS = stats.ttest_ind(WS_E,WS_D)
t_DC_WS,p_DC_WS = stats.ttest_ind(WS_D,WS_C)
t_CB_WS,p_CB_WS = stats.ttest_ind(WS_C,WS_B)
t_BA_WS,p_BA_WS = stats.ttest_ind(WS_B,WS_A)

np.savetxt("T_test.csv", [[t_ED_temp,p_ED_temp],[t_DC_temp,p_DC_temp],[t_CB_temp,p_CB_temp]
,[t_BA_temp,p_BA_temp],[t_ED_WS,p_ED_WS],[t_DC_WS,p_DC_WS],[t_CB_WS,p_CB_WS],[t_BA_WS,p_BA_WS]], delimiter=",")
print (t_ED_temp,p_ED_temp)
print (t_DC_temp,p_DC_temp)
print (t_CB_temp,p_CB_temp)
print (t_BA_temp,p_BA_temp)
print (t_ED_WS,p_ED_WS)
print (t_DC_WS,p_DC_WS)
print (t_CB_WS,p_CB_WS)
print (t_BA_WS,p_BA_WS)

#Bonus question

def average_temperature(data):
    temperature=[data[0:72].mean(),data[72:144].mean(),data[144:216].mean(),data[216:288].mean(),data[288:360].mean(),data[360:432].mean(),data[432:504].mean(),
    data[504:576].mean(),data[576:648].mean(),data[648:720].mean(),data[720:792].mean(),data[792:864].mean(),data[864:936].mean(),data[936:1008].mean(),
    data[1008:1080].mean(),data[1080:1152].mean(),data[1152:1224].mean(),data[1224:1296].mean(),data[1296:1368].mean(),data[1368:1440].mean(),data[1440:1512].mean()
    ,data[1512:1584].mean(),data[1584:1656].mean(),data[1656:1728].mean(),data[1728:1800].mean(),data[1800:1872].mean(),data[1872:1944].mean(),data[1944:2016].mean(),
    data[2016:2088].mean(),data[2088:2160].mean(),data[2160:2232].mean(),data[2232:2304].mean(),data[2304:2376].mean(),data[2376:2448].mean(),data[2448:2476].mean()]
    dates=["6/10/2020","6/11/2020","6/12/2020","6/13/2020","6/14/2020","6/15/2020","6/16/2020","6/17/2020","6/18/2020","6/19/2020"," 6/20/2020",
    "6/21/2020","6/22/2020","6/23/2020","6/24/2020","6/25/2020","6/26/2020","6/27/2020","6/28/2020","6/29/2020","6/30/2020","7/1/2020","7/2/2020",
    "7/3/2020","7/4/2020","7/5/2020","7/6/2020","7/7/2020","7/8/2020","7/9/2020","7/10/2020","7/11/2020","7/12/2020","7/13/2020","7/14/2020"]    
    d={'Temperature':temperature,'Date':dates}
    df_print=pd.DataFrame(d)
    df_print.sort_values(by=['Temperature','Date'],axis=0,ascending=False)
    print ("The temperature and date of the hottest day are: ", df_print[df_print.Temperature == df_print.Temperature.max()]) 
    print ("The temperature and date of the coolest day are: ", df_print[df_print.Temperature == df_print.Temperature.min()]) 
    
   
print (average_temperature(df_A[4]))
print (average_temperature(df_B[4]))
print (average_temperature(df_C[4]))
print (average_temperature(df_D[4]))
print (average_temperature(df_E[4]))
plt.show()

