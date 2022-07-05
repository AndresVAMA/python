# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:35:16 2022

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt

#Datos para la integracion
x=0.0 #Inicio de la integracion
xf=100.0 #Fin de la integracion (tiempo)
y=np.array([(np.pi/2.0),0.0,(np.pi/2.0),0.0]) #Nuestra "y" la vamos a analizar en 1/8 de la esfera
h=0.005

#Datos conocidos
g=9.81
l=0.4 #Longitud de los brazos
#La masa la podemos despreciar, ya que es 1kg

def F(x,y):
    F=np.zeros(4) #Ya que tenemos 4 ec's vamos a hacer un arreglo de 4 0's
    #y[0]=theta1
    #y[1]=theta1_punto
    #y[2]=theta2
    #y[3]=theta2_punto
    #F[0]=w1
    #F[1]=w1_punto
    #F[2]=w2
    #F[3]=w2_punto
    
    divisor=3.0-np.cos(2.0*(y[0]-y[2]))
    t1=np.sin(y[0]-y[2])
    t2=np.sin(2.0*(y[0]-y[2]))
    t3=np.sin(y[0]-2.0*y[2])
    t4=np.sin(2.0*y[0]-y[2])
    t5=np.sin(y[0])
    t6=np.sin(y[2])
    
    F[0]=y[1] #w1=theta1_punto
      
    F[1]=-((y[1]**2)*t2+2.0*(y[3]**2.0)*t1+(g/l)*(t3-3.0*t5) )/divisor #w1_punto 
    
    F[2]=y[3] #w2=theta2_punto
      
    F[3]=(4.0*(y[1]**2)*t1+(y[3]**2.0)*t2+2.0*(g/l)*(t4-t6) )/divisor #w2_punto
          
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

X,Y=integra(F,x,y,xf,h)
E=(l**2)*((Y[:,1]**2)+((1/2)*(Y[:,3]**2))+(Y[:,1]*Y[:,3])*np.cos(Y[:,0]-Y[:,2]))- (g*l)*(2*np.cos(Y[:,0])+np.cos(Y[:,2]))

plt.plot(X,Y[:,0],'o',X,E,'-')
plt.grid(True)
plt.xlabel('x');plt.ylabel('y')
plt.legend(('Numerico','Exacto'),loc=0)
plt.show()