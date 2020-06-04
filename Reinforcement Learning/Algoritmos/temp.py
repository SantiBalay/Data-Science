# -*- coding: utf-8 -*-

#Resolucion de Arm Bandit 1

import numpy as np
import random
import matplotlib.pyplot as plt


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
        
m1 = Maquina(40.5) #distintos means que representan la chance de que se gane con la maquina
m2 = Maquina(40)
m3 = Maquina(39)

maquinas = [m1,m2,m3]


def runExperiment(maquinas,iteraciones,eps):
    
    data = np.empty(iteraciones)
    
    for t in range(iteraciones):
        maquina = elegirMaquina(maquinas,eps)
        nuevoValor = maquina.pull()
        maquina.actualizarMean(nuevoValor)
        
        data[t] = nuevoValor
        
    cumulativeAvg = np.cumsum(data) / (np.arange(iteraciones) + 1)
    

    plt.plot(cumulativeAvg)
    plt.plot(np.ones(iteraciones)*40.5) #barras de referencia
    plt.plot(np.ones(iteraciones)*40)
    plt.plot(np.ones(iteraciones)*39)
    plt.xscale('log')
    plt.show()
    
    return cumulativeAvg
        
    
def elegirMaquina(maquinas,eps):
    if(random.uniform(0, 1) < eps):
        print("EPSILON")
        ind = np.random.choice(3)
        print('Maquina',ind)
        return maquinas[ind]
    else:
        return mejorMaquina(maquinas)
        
def mejorMaquina(maquinas):
    ind = np.argmax(list(map(lambda x: x.calcMean, maquinas))) ##
    print("Maquina",ind)
    return maquinas[ind]
        
def shit():
    return random.uniform(0, 1)

runExperiment(maquinas,10000,0.1)
runExperiment(maquinas,10000,0.05)
runExperiment(maquinas,10000,0.01)


