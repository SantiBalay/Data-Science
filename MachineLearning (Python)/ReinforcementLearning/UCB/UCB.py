#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:26:48 2020

@author: santiagobalay
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
#Son 10 ads para una campaña publicitaria. 1=hit 0=nohit. Quiero la mejor lo mas rapido. La idea es que esto se va a hacer ronda por ronda, aunque ya tenga todos los resultados.
#osea dependiendo de rewards anteriores, se van a mostrar algunos ads, y otros no.
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implemento UDB

d=10 #cantidad de arms
n = 10000 #cantidad de rounds
numero_selecciones = [0] * d
reward_total= [0] * d
ads_elegidos = []
reward_final=0

for j in range(0,n):
    ad = 0
    upper_bound_max = 0
    for i in range(0, d):
        if(numero_selecciones[i] > 0): #so la eleji minimo 1
            reward_average = reward_total[i]/numero_selecciones[i]
            delta_i = math.sqrt(3/2 * math.log(j+1)/numero_selecciones[i])
            upper_bound = reward_average + delta_i
        else:
            upper_bound = 1e400 #me aseguro que la hayan elegido minimo 1 vez
        if upper_bound > upper_bound_max:
            upper_bound_max = upper_bound
            ad = i
    ads_elegidos.append(ad)
    numero_selecciones[ad]= numero_selecciones[ad] + 1
    reward = dataset.values[j,ad] #si era =1, sino =0
    reward_total[ad] = reward_total[ad] + reward
    reward_final = reward_final + reward



plt.hist(ads_elegidos)
plt.title('Histograma de Ads Elegidos')
plt.xlabel('Ads')
plt

#claramente el 4 resulta ser el mejor ad a utilizar, y es un trend que se puede observar mucho antes de procesar el
























































