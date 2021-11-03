import pandas as pd
import random
import numpy as np

#Generamos parametros (N,P y W) y valores de peso y valor al azar
def generarVectorAlAzar():
    N = random.randint(5,20)
    P = int((random.randint(50,100)/100)*N)
    peso= []
    valor= []
    values=[]
    for i in range(N):
        peso.append(random.randint(1,10))
        valor.append(random.randint(1,10))
    for i in range(N):
        values.append([peso[i],valor[i]])
    suma = np.sum(peso)//2
    stdDev = int(np.std(peso))
    W=int(suma-stdDev)
    print('N: ',N)
    print('P: ',P)
    print('W: ',W)
    print('Values: ',values)
    return N,P,W,values

#Generamos csv con los valores y los parametros
def generarCSV(values):
    problema = pd.DataFrame(values,columns=['Peso','Valor'])
    #p.csv contiene los valores de peso y valor
    problema.to_csv('p.csv')
    param = pd.DataFrame([N,P,W],columns=['Parametros'])
    #s.csv contiene los valores de los parametros N,P y W
    param.to_csv('s.csv')

#Funcion de lectura de los csv con el problema
def leerCSV():
    problema2 = pd.read_csv('p.csv')
    param = pd.read_csv('s.csv')
    lista = []
    for i in range(problema2['Peso'].size):
        item = [problema2['Peso'][i],problema2['Valor'][i]]
        lista.append(item)
    N = param['Parametros'][0]
    P = param['Parametros'][1]
    W = param['Parametros'][2]
    return N,P,W,lista

#Generamos problema
[N,P,W,values] = generarVectorAlAzar()
#Guardamos problema en los csv
generarCSV(values)
#Leemos el problema de los csv
[N,P,W,lista] = leerCSV()
print('N: ',N)
print('P: ',P)
print('W: ',W)
print('Values: ',lista)
