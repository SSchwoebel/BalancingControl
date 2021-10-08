#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:23:25 2021

@author: baemm
"""
"""Einlesen der Daten, Formen des Dataframes, Hinzufügen Header, Löschen des unnötigen"""
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sc

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

"""Plotten + Korrelationen mit Pandas, SciPy- dt """

#ergebnis.plot(x ='dt', y='Modus_dt', kind = 'scatter')	 
plt.figure()   
plt.title("Scatter")
plt.ylabel("Modus_dt")
plt.xlabel("real_dt")
plt.scatter(ergebnis["dt"],ergebnis["Modus_dt"])
plt.show()


corr, pvalue = sc.pearsonr(ergebnis["dt"], ergebnis["Modus_dt"])
print("Korrelationskoeffizient:", corr)
print("P-Value;",pvalue)
print("P-Value komplett;","{:0.30f}".format(pvalue))

# creating X-Y Plots With a Regression Line

# slope, intersept, and correlation coefficient calculation 
slope, intercept, r, p, stderr = sc.linregress(ergebnis["dt"], ergebnis["Modus_dt"])
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
# plotting
fig, ax = plt.subplots(figsize = (14,8))
ax.plot(ergebnis["dt"], ergebnis["Modus_dt"], linewidth=0, marker='s', label='Data points')
ax.plot(ergebnis["dt"], intercept + slope * ergebnis["dt"], label=line)
ax.set_xlabel('real_dt')
ax.set_ylabel('Modus_dt')
ax.legend(facecolor='white')
plt.show()



#Korrelationsmatrix
Matrix = ergebnis.corr(method='pearson', min_periods=1)
print(Matrix)

import seaborn as sns

f, ax = plt.subplots(figsize=(14, 8))
corr = ergebnis.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot=True, ax=ax)

from pandas.plotting import scatter_matrix
scatter_matrix(ergebnis, figsize=(14,8)) #sieht ein wenig umständlich aus daher das darüber
Matrix.style.background_gradient(cmap='coolwarm') # geht erst wenn keine NaNs mehr da sind
plt.show()