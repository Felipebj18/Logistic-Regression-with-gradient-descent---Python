
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:19:34 2022

@author: PC GAMER
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix

plt.close("all")
data = pd.read_csv("Social_Network_Ads.csv",sep=",")



#Dummies para el género
data['Gender']=data['Gender'].astype('category')
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)


#Variable objetivo: Purchased
#La variable User ID es irrelevante
data = data.drop(['User ID'],axis = 1)
#Correlación de los datos
co = data.corr()
#plt.figure()
#Gráfica de correlaciones
#sns.heatmap(co, annot=True)
#plt.tight_layout()
 
#Los valores absoluto de las correlaciones de la variable Purchased (objetivo) con las variables Age y Estimated Salary 
#son considerablemente altos, mientras que con Gender la correlación es bastante baja (su valor absoluto es mucho menor a 0.1)
#Por lo tanto, esta última variable se descarta por no aportar una cantidad considerable de información en este caso.
#Nuestros datos quedan con dos características y una etiqueta

#1.Datos: 2 características y una etiqueta
data = data.drop(['Gender_Male'],axis=1)

edad = data['Age']
salario = data['EstimatedSalary']

#Normalización de los datos
m_edad = np.mean(edad)
m_salario = np.mean(salario)
sd_edad = np.std(edad)
sd_salario = np.std(salario)
edad_n = (edad- m_edad)/sd_edad
salario_n = (salario - m_salario)/sd_salario

data['Edad'] = edad_n#Edad normalizada
data['Salario'] = salario_n#salario normalizado

data = data.drop('Age',axis=1)
data = data.drop('EstimatedSalary',axis=1)

    #Division de datos 60-40
data_train, data2 = train_test_split(data, test_size = 0.4)
    
    #División del 40% en mitades para entrenamiento y validación cruzada
data_test,data_cv = train_test_split(data2, test_size = 0.5)


edad_train = data_train['Edad']#X1
salario_train = data_train['Salario']#X2

edad_test = data_test['Edad']
salario_test = data_test['Salario']

edad_cv = data_cv['Edad']
salario_cv = data_cv['Salario']





#2. Gráfica de dispersión para la variable objetivo 
colores = {1:"Cyan",0:"Purple"}
color_prediccion = data.Purchased.map(colores)
fig, ax = plt.subplots()
ax.scatter(edad_n,salario_n, color = color_prediccion)

plt.show()

#3. Modelo de regresion Logística

theta = np.array([-25,0,10])

x1 = np.arange(-3.5,3.5,0.1)
x2 = (-theta[1]*x1/theta[2]-theta[0]/theta[2])
y = data_train['Purchased']#Etiqueta


m=len(edad_train)
alfa=0.3

H = list()

lambd=np.array([100000,1000,100,1,0.0001])


  
#Predicciones
def prediccion (X1,X2,y,lambd,theta,alfa,iteraciones):
    h=1/(1+np.exp(-theta[0]*1-theta[1]*X1-theta[2]*X2))
    for i in lambd:    
        for j in range(iteraciones):  
                    
                        theta=theta-alfa/m*(np.array([sum((h-y)*1),sum((h-y)*X1),sum((h-y)*X2)])+i/m*theta)
                        h=1/(1+np.exp(-theta[0]*1-theta[1]*X1-theta[2]*X2))
                        
                        #hg=1/(1+np.exp(-theta[0]-theta[1]*xg))
                        #g,=plt.plot(xg,hg,"b")
                        x2=(-theta[1]*x1/theta[2]-theta[0]/theta[2])
                        #theta[1]*x1-theta[2]*x2
                        #g,=plt.plot(x1,x2,"r")
                    
                        #plt.pause(0.001)
                        #g.remove()
        print(i)
        H.append(h)

    return H
    
#4. Cross Validation con el 20% de los datos para los 5 valores de lambda
H=prediccion(data_cv["Edad"],data_cv["Salario"],data_cv["Purchased"],lambd,theta,0.5,100)


H0 = list(H[0])#Con lambda = 1
H1 = list(H[1])#Con lambda = 0.1
H2 = list(H[2])#Con lambda = 0.01
H3 = list(H[3])#Con lambda = 0.001
H4 =  list(H[4])#Con lambda = 0.0001


def clasificar(lista,umbral):
    lista_p = list()
    for i in lista:
        if i<umbral:
            lista_p.append(0)
        else:
            lista_p.append(1)
    return lista_p


#Calcular accuracy
#accuracy = (TP+TN)/(TP+FP+TN+FN)


def calcular_precision (data,pred,umbral):
    #Saco la matriz de confusión
    mc = confusion_matrix(data, pd.DataFrame(clasificar(pred,umbral)))
    TP = mc[0,0]
    TN = mc[1,1]
    FP = mc[0,1]
    FN = mc[1,0]
    precision = (TP)/(TP+FP)
    return precision

#Calculo las precisiones para cada lambda

prc0 = calcular_precision(data_cv['Purchased'],H0,0.5)
prc1 = calcular_precision(data_cv['Purchased'],H1,0.5)
prc2 = calcular_precision(data_cv['Purchased'],H2,0.5)
prc3 = calcular_precision(data_cv['Purchased'],H3,0.5)
prc4 = calcular_precision(data_cv['Purchased'],H4,0.5)


prc = [prc0,prc1,prc2,prc3]


#Los valores que dan mejor precision para esta particion de datos son lamd[3]=1 y lambd[4]=0.0001

#5.Animación sobre la gráfica del punto 2 con el valor de lambda escogido, optimizando theta

lam = 0.0001
alfa = 0.5
y=data["Purchased"]

h=1/(1+np.exp(-theta[0]*1-theta[1]*edad_n-theta[2]*salario_n))
m=len(edad_n)
theta = np.array([30,-10,8])
for j in range(1000):  
                
        theta=theta-alfa/m*(np.array([sum((h-y)*1),sum((h-y)*edad_n),sum((h-y)*salario_n)])+(lam/m)*theta)
        h=1/(1+np.exp(-theta[0]*1-theta[1]*edad_n-theta[2]*salario_n))
        
        #hg=1/(1+np.exp(-theta[0]-theta[1]*xg))
        #g,=plt.plot(xg,hg,"b")
        x2=(-theta[1]*x1/theta[2]-theta[0]/theta[2])
        #theta[1]*x1-theta[2]*x2
        g,=plt.plot(x1,x2,"r")
        
        plt.pause(0.001)
        g.remove()
    
        H.append(h)

print(theta)





