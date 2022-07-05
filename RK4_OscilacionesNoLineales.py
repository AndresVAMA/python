# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 20:05:24 2022

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt

#Datos conocidos
m=1
p=9
w=20
#Datos para la integracion
t=2*np.pi/w
x = 0 #Inicio de la integracion
xf = 3*t #Fin de la integracion (tiempo)
y = np.array([10.0, 0.0])
h=t/200000

def F(x,y):
    F = np.zeros(2)
    F[0] =y[1]
    F[1] =-w**2*np.abs(y[0])**(p-1)*y[0]/np.abs(y[0])
          
    return F

def integra(F,x,y,xf,h):
    
    X = [ ]
    Y = [ ]
    X.append(x)
    Y.append(y)
    #Creamos los arreglos para las k's
    k_0=np.zeros(4)
    k_1=np.zeros(4)
    k_2=np.zeros(4)
    
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

X ,Y = integra(F,x,y,xf,h)

plt.plot(X,Y[:,0],'-')
plt.grid(True)
plt.xlabel('x'); plt.ylabel('y')
plt.legend(('Numérico','Exacto'),loc=0)
plt.show()