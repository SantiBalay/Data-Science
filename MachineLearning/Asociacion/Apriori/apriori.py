#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:40:53 2020

@author: santiagobalay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from apyori import apriori as ap

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) #in a week

#apriori espera una lista de listas no un X normal

facturas = []

for i in range(0,7501):
    facturas.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#train apriori
#quiero los que estense compren minimo 3 veces al dia.   7*4/7500 =~ 0,003
support = 0.003
#me quiero asegurar que la relacion no sea por cantidad de que se compran y por una relacion intrisica que tengan y el default es 0.8 que es una banda. Con 0.4 me dio cosas que se compraban mucho, asique lo baje 0.2.
confidence = 0.2
#quiero un lift mayor a 3 para encontrar buenas relaciones
lift=3
    
rules = ap(facturas, min_length=2, min_confidence=confidence, min_lift=lift, min_support=support)

results = list(rules)
results2 = map(lambda x : x.items, results)

my_set = list(results2)
print(my_set)


shit = "banana"
shit


bar = {}
bar['foo'] = 'foobar'

len(shit)