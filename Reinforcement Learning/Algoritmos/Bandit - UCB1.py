
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:22:08 2020

@author: santiagobalay
"""


# -*- coding: utf-8 -*-

#Resolucion de Arm Bandit 3 - UCB1

import numpy as np
import random
import matplotlib.pyplot as plt
import math

class Maquina:
    def __init__(self,trueMean):
        self.trueMean = trueMean
        self.calcMean = 0
        self.n = 0
        
    def pull(self):
        return np.random.randn() + self.trueMean
        
    def actualizarMean(self,x):
        self.n += 1
        self.calcMean = (1 - 1.0/self.n) * self.calcMean + (1.0/ self.n) * x
        
m1 = Maquina(3) #distintos means que representan la chance de que se gane con la maquina
m2 = Maquina(2)
m3 = Maquina(1)

maquinas = [m1,m2,m3]

def runExperiment(maquinas,iteraciones):
    
    data = np.empty(iteraciones)
    
    for t in range(1,iteraciones):
        maquina = mejorMaquina(maquinas,t)
        nuevoValor = maquina.pull()
        maquina.actualizarMean(nuevoValor)
        
        data[t] = nuevoValor
        
    cumulativeAvg = np.cumsum(data) / (np.arange(iteraciones) + 1)
    
    plt.plot(cumulativeAvg)
    plt.plot(np.ones(iteraciones)*maquinas[0].trueMean) #barras de referencia
    plt.plot(np.ones(iteraciones)*maquinas[1].trueMean)
    plt.plot(np.ones(iteraciones)*maquinas[2].trueMean)
    plt.xscale('log')
    plt.show()
    
    return cumulativeAvg
        
    
def elegirMaquina(maquinas):
        return mejorMaquina(maquinas)
        
def mejorMaquina(maquinas,t):
    ind = np.argmax(list(map(lambda maquina: (maquina.calcMean + math.sqrt(2*math.log(t) / (maquina.n + 0.00000000000001 ))), maquinas))) ##
    print("Maquina",ind)
    return maquinas[ind]
        

runExperiment(maquinas,10000)