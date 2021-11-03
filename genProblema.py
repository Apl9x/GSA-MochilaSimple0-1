import pandas as pd
import random
import numpy as np

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

def generarCSV(values):
    problema = pd.DataFrame(values,columns=['Peso','Valor'])
    problema.to_csv('p.csv')
    param = pd.DataFrame([N,P,W],columns=['Parametros'])
    param.to_csv('s.csv')

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

[N,P,W,values] = generarVectorAlAzar()
generarCSV(values)
[N,P,W,lista] = leerCSV()
print('N: ',N)
print('P: ',P)
print('W: ',W)
print('Values: ',lista)
