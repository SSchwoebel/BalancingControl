#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:23:25 2021

@author: baemm
"""
#%%

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sc
import seaborn as sns
from pandas.plotting import scatter_matrix


"""Einlesen der Daten, Formen des Dataframes und der fehlende Werte"""

header = []
daten = []

pfad = '/home/baemm/Coding/BalancingControl/dt'
dateien = os.listdir(pfad)
dateien = sorted(dateien)
for datei in dateien:
    with open(f'{pfad}/{datei}', 'r') as zu_lesen:
        reader = csv.reader(zu_lesen, delimiter=',')
        header = next(reader)
        daten.extend([row for row in reader])

ergebnis  = pd.DataFrame(data=daten, dtype=np.float32)
ergebnis = ergebnis.rename(columns=(dict(zip(ergebnis.columns,header))))
del ergebnis[""]
"""Modus und Mean der Verteilung berechnen - lambda pi"""
#(a-1)/(a+b-2)

np_data = ergebnis.to_numpy()
a = np_data[:,0]
b = np_data[:,1]

Modus = (a -1) / (a + b -2)
ergebnis['Modus_pi'] = Modus 

#mean a/(a+b)
Mean = (a) / (a + b)
ergebnis['Mean_pi'] = Mean

"""Modus und Mean der Verteilung berechnen - lambda r"""
#(a-1)/(a+b-2)

a = np_data[:,2]
b = np_data[:,3]

Modus_rr = (a -1) / (a + b -2)
ergebnis['Modus_r'] = Modus_rr

#mean a/(a+b)
Mean_rr = (a) / (a + b)
ergebnis['Mean_r'] = Mean_rr

"""Mean und Modus Gamma Verteilung - decision temperature"""
#modus = (a-1)/b bei a >1
a = np_data[:,4]
b = np_data[:,5]

for i in range(0,a.size):
    if a[i] <= 1:
        print("CAVE")
    
Modus_dtt = (a - 1 ) / (b)
ergebnis['Modus_dt'] = Modus_dtt

#mean = a/b
Mean_dtt  = (a/ b)
ergebnis['Mean_dt'] = Mean_dtt

"""ohne dt = 1.0"""
ergebnis_ohne = ergebnis[ergebnis['dt']!=1.0]

#%%
"""Funktionen definieren"""

def korr_plot(x,v1,v2):
    """x = DataFrame; v1=Variablen String, v2=Variablen String
    Rückgabe: Scatter, Scatter + Regressionline, Histogramme, Tabelle"""
    

    corr, pvalue = sc.pearsonr(x[v1], x[v2])
    print("Korrelationskoeffizient für "+v1+":", corr)
    print("P-Value für "+v1+":",pvalue)
    print("P-Value komplett für "+v1+":","{:0.30f}".format(pvalue))
    


# creating X-Y Plots With a Regression Line
# slope, intersept, and correlation coefficient calculation 
    slope, intercept, r, p, stderr = sc.linregress(x[v1], x[v2])
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
# plotting
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(x[v1], x[v2], linewidth=0, marker='s', label='Data points')
    ax.plot(x[v1], intercept + slope * x[v1], label=line)
    ax.set_xlabel('real ' + v1)
    ax.set_ylabel(v2)
    ax.legend(facecolor='white')
    
#ergebnis.plot(x ='dt', y='Modus_dt', kind = 'scatter')	 
    plt.figure(figsize=(14, 8))  
    plt.subplot(2,2,(1,2))
    plt.title("Scatter")
    plt.ylabel(v2)
    plt.xlabel('real ' + v1)
    plt.scatter(x[v1],x[v2])
    
# Histogramm und Tabellen
    plt.subplot(2,2,4)
    plt.title("Histogramm für " +v1)
    plt.hist(x[v1])

    
    plt.subplot(2,2,3)
    plt.title("Histogramm für " +v2)
    plt.hist(x[v2])
    plt.show()
    
    Tabelle1 = x[v1].value_counts(sort=True)
    Tabelle2 = x[v2].value_counts(sort=True)
    print("Häufigkeit für real "+v1+" in Zahlen:\n\n", Tabelle1)
    print("Häufigkeit für "+v2+" in Zahlen:\n\n", Tabelle2)

def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = sc.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix    
    
def plot_cor_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(14,10))
    sns.heatmap(corr, ax=ax,
            mask=mask,
            # cosmetics
            annot=True, vmin=-1, vmax=1, center=0, square=True, 
            cmap='coolwarm', linewidths=0.01, linecolor='black', cbar_kws={'orientation': 'vertical'})
            #sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            #              square=True, annot=True, ax=ax)

#%%% Auswertung
#korr_plot(ergebnis,"dt","Modus_dt")
#korr_plot(ergebnis,"rl","Modus_r")
#korr_plot(ergebnis,"pl","Modus_pi")
##Ohne dt = 1.0
korr_plot(ergebnis_ohne,"dt","Modus_dt")
korr_plot(ergebnis_ohne,"rl","Modus_r")
korr_plot(ergebnis_ohne,"pl","Modus_pi")

#%%
"""ELBO der Agenten"""

LOSSa= pd.DataFrame()
a = []
i=0
pfad = '/home/baemm/Coding/BalancingControl/LOSS'
dateien = os.listdir(pfad)
dateien = sorted(dateien)


for datei in dateien:
    with open(f'{pfad}/{datei}', 'r') as zu_lesen:
        reader = pd.read_csv(zu_lesen, delimiter=',')
        a =  reader
        a = pd.DataFrame(a["0"])
        LOSSa[str(i)] = a["0"]
    i = i+1 
    
# for Agentnr in range(0,LOSSa.shape[1]):
# #Agentnr= 0
# #Elbo Agenten 0 bis siehe size
#     plt.figure()
#     plt.title("Agent"+str(Agentnr))
#     plt.plot(LOSSa[str(Agentnr)])
#     plt.ylabel("ELBO")
#     plt.xlabel("iteration")
#     plt.show()  
    
#nicht konvergiert    
ergebnis_ohne = ergebnis_ohne.drop([242,241,34,42,25,21,35,69,93,126,29])
    

#%%
"""Masterarbeitsplots"""

#Auswertungsplots Iterationen    

pfad = '/home/baemm/Coding/BalancingControl/Iterationen'
dateien = os.listdir(pfad)
dateien = sorted(dateien)


datei = dateien[0] 
with open(f'{pfad}/{datei}', 'r') as zu_lesen:
    reader = pd.read_csv(zu_lesen, delimiter=',')
    a =  reader
    a = pd.DataFrame(a["0"])
    
datei = dateien[1] 
with open(f'{pfad}/{datei}', 'r') as zu_lesen:
    reader = pd.read_csv(zu_lesen, delimiter=',')
    b =  reader
    b = pd.DataFrame(b["0"])
    
datei = dateien[2] 
with open(f'{pfad}/{datei}', 'r') as zu_lesen:
    reader = pd.read_csv(zu_lesen, delimiter=',')
    c =  reader
    c = pd.DataFrame(c["0"])

    
plt.figure(figsize=(14, 8))  
plt.subplot(1,3,1)
plt.title('1000 Iterations')
plt.plot(a['0'])
plt.ylabel("ELBO")
plt.xlabel("Iteration")
plt.subplot(1,3,2)
plt.title('1500 Iterations')
plt.plot(b['0'])
plt.ylabel("ELBO")
plt.xlabel("Iteration")
plt.xlim(0,1500)
plt.subplot(1,3,3)
plt.title('2000 Iterations')
plt.plot(c['0'])
plt.ylabel("ELBO")
plt.xlabel("Iteration")
plt.xlim(0,2000)
plt.show()  


ergebnis_ohne = ergebnis_ohne.rename(columns={"dt": "$\gamma_{dt}$", "pl": "$\lambda_{pi}$", "rl": "$\lambda_{r}$",'Modus_dt': "mode $\gamma_{dt}$",'Modus_r' : "mode $\lambda_{r}$",'Modus_pi':"mode $\lambda_{pi}$","Mean_dt" : "$\overline{\gamma_{dt}}$", 'Mean_r': "$\overline{\lambda_{r}}$", 'Mean_pi': "$\overline{\lambda_{pi}}$"})

fig, axs = plt.subplots(ncols=3)
sns.regplot(x="$\gamma_{dt}$", y="mode $\gamma_{dt}$", data=ergebnis_ohne,x_jitter=0.1,line_kws={'color':"orange"},truncate=False,ax=axs[0]) 
#plt.title('Scatter',fontweight ="bold")
#plt.legend(labels =['Regression line','Data points'])
#axs.set_ylim([0, 8])
#axs.set_xlim([0, 8])

sns.regplot(x="$\lambda_{pi}$", y="mode $\lambda_{pi}$", data=ergebnis_ohne,x_jitter=.01,line_kws={'color':"orange"},truncate=False,ax=axs[1]) 
#plt.title('Scatter',fontweight ="bold")
#plt.legend(labels =['Regression line','Data points'])
#axs.set_ylim([0.1,1.0])
#axs.set_xlim([0.1,1.0])

sns.regplot(x="$\lambda_{r}$", y="mode $\lambda_{r}$", data=ergebnis_ohne,x_jitter=.01,line_kws={'color':"orange"},truncate=False,ax=axs[2]) 
#plt.title('Scatter',fontweight ="bold")
#plt.legend(labels =['Regression line','Data points'])
#axs.set_ylim([0.1,1.0])
#axs.set_xlim([0.1,1.0])

plt.suptitle('Scatterplot',fontweight ="bold")
plt.tight_layout()
fig.legend(labels =['Regression line','Data points'],prop={'size': 8}, loc='upper left')

plt.show()

###Korrelationsmatrix 

#Weniger Variablen
df = ergebnis_ohne[["$\lambda_{pi}$","$\lambda_{r}$","$\gamma_{dt}$","mode $\gamma_{dt}$","mode $\lambda_{r}$","mode $\lambda_{pi}$","$\overline{\gamma_{dt}}$","$\overline{\lambda_{r}}$","$\overline{\lambda_{pi}}$"]]

scatter_matrix(df, figsize=(14,8)) #sieht ein wenig umständlich aus daher das darüber
plt.show()

corr = df.corr()                            # get correlation
p_values = corr_sig(df)                     # get p-Value
mask = np.invert(np.tril(p_values<0.0013))    # mask - only get significant corr
plot_cor_matrix(corr,mask)
plt.title("Correlation Matrix",fontweight ="bold")

