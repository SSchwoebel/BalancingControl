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

def korr_plot(x,v1,v2):
    """x = DataFrame; v1=Variablen String, v2=Variablen String
    Rückgabe: Scatter, Scatter + Regressionline, Histogramme, Tabelle"""
    
#ergebnis.plot(x ='dt', y='Modus_dt', kind = 'scatter')	 
    plt.figure()   
    plt.title("Scatter")
    plt.ylabel(v2)
    plt.xlabel('real ' + v1)
    plt.scatter(x[v1],x[v2])
    plt.show()


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
    plt.show()
# Histogramm und Tabellen
    plt.figure()
    plt.title("Histogramm für " +v1)
    plt.hist(x[v1])
    plt.show()
    
    plt.figure()
    plt.title("Histogramm für " +v2)
    plt.hist(x[v2])
    plt.show()
    
    Tabelle1 = x[v1].value_counts(sort=True)
    Tabelle2 = x[v2].value_counts(sort=True)
    print("Häufigkeit für real "+v1+" in Zahlen:\n\n", Tabelle1)
    print("Häufigkeit für "+v2+" in Zahlen:\n\n", Tabelle2)

#%%
"""Einlesen der Daten, Formen des Dataframes und der fehlende Werte"""

header = []
daten = []

pfad = '/home/baemm/Coding/BalancingControl/dt'
dateien = os.listdir(pfad)

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
Mean = (a / a + b)
ergebnis['Mean_pi'] = Mean

"""Modus und Mean der Verteilung berechnen - lambda r"""
#(a-1)/(a+b-2)

a = np_data[:,2]
b = np_data[:,3]

Modus_rr = (a -1) / (a + b -2)
ergebnis['Modus_r'] = Modus_rr

#mean a/(a+b)
Mean_rr = (a / a + b)
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

#%%% Auswertung
korr_plot(ergebnis,"dt","Modus_dt")
korr_plot(ergebnis,"rl","Modus_r")
korr_plot(ergebnis,"pl","Modus_pi")



#%% Korrelationsmatrix 
"""Plotten + Korrelationen mit Pandas, SciPy """

#Korrelationsmatrix ##Background Styl geht nicht in Spyder. 
# Matrix = ergebnis.corr(method='pearson', min_periods=1)
# Matrix.style.background_gradient(cmap='coolwarm').set_precision(2) 
# print(Matrix)

#Weniger Variablen
df = ergebnis[['pl','rl','dt','Modus_dt','Modus_r','Modus_pi']]

#Gleiche wie unten mit relevanteren Variablen
f, ax = plt.subplots(figsize=(14, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot=True, ax=ax)

scatter_matrix(df, figsize=(14,8)) #sieht ein wenig umständlich aus daher das darüber
plt.show()

#SNS Korrelationsmatrix
f, ax = plt.subplots(figsize=(14, 8))
corr = ergebnis.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot=True, ax=ax)

#Scattermatrix
scatter_matrix(ergebnis, figsize=(14,8)) #sieht ein wenig umständlich aus daher das darüber
plt.show()

 