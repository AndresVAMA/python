# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:56:57 2022

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt


xi=0 #Inicio del intervalo
xf=20 #Fi del intervalo
h=0.0005 #len de los pasos


#Primero definimos nuestras F, G y la solucion exacta

def exacta(x):
    exacta= 1-(1/2)*(x+2)*np.exp(-x)
    
    return exacta

def F(x):
    F=0.0  
    
    return F

def G(x):
    G=(-1/2)*x*np.exp(-x)
    
    return G

#Ahora definimos las y's y variables nercesarias

x=xi #Empezamos desde xi

sol=0.0 #Y en 0

n=int((xf-xi)/h) #Las veces que vamos a partir "el pastel"

y=np.zeros(n+1) #Un arreglo de n+1 con la solucion para cada paso de h

y[1]= exacta(h)  #En y[1] tenemos el valor de la solucion exacta

#Numerov

def integra(F,G,xf,h,x,y):
    
    X=[]
    Y=[]
    
    X.append(x)
    Y.append(y)
    
    for i in range(1,n): #Pa que recorra desde 1 hasta n=((xf-xi)/h)-1
        
        sol= 2.0*y[i]-y[i-1]+((h**2)/12)*(G(x+h)+10.0*G(x)+G(x-h)) #Lo sacamos de la presentacion
        x=x+h #Que vaya recorriendo
        
        X.append(x)
        Y.append(sol)
        
        y[i+1]=sol
    
    return np.array(X),np.array(y)

X,Y= integra(F,G,xf,h,x,y)

yexacta=exacta(X)
plt.plot(X,Y[0:n],'.',X,yexacta,'-') #Con forme va recorriendo x, le vamos geenerando una y
plt.grid(True)
plt.xlabel('x');plt.ylabel('y')
plt.legend(('numerico','exacto'),loc=5)
plt.show()

print(Y)