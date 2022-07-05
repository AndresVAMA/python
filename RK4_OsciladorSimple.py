# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 19:40:47 2022

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt

#Datos
n=10000.0
E_0=0.5

x = 0.0 #Inicio de la integracion
xf = 2.0*np.pi #Fin de la integracion (tiempo)
y = np.array([1.0, 0.0])
h = xf/n

def F(x,y):
   F = np.zeros(2) #Ya que tenemos 2 ec's vamos a hacer un arreglo de 2 0's
   F[0] = y[1]
   F[1] = -1.0*y[0]
   return F 

def integra(F,x,y,xf,h):
    
    X=[]   #arreglo vacio de los valores X de la solucion
    Y=[]   #arreglo vacio de los valores Y de la solucion
    
    while x<xf:
        k_0 = h*F(x,y)
        k_1 = h*F(x+h/2.0,y+k_0/2.0)
        k_2 = h*F(x+h/2.0,y+k_1/2.0)
        k_3 = h*F(x+h,y+k_2)
        y = y + (1.0/6.0)*(k_0 + 2*k_1 + 2*k_2 + k_3)
        x=x+h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

X ,Y= integra(F,x,y,xf,h)
plt.plot(X,E_0-(0.5*(Y[:,0])**2+0.5*(Y[:,1])**2),'-')
plt.grid(True)
plt.xlabel('t'); plt.ylabel('E')
plt.legend(('Numérico'),loc=2)
plt.show()