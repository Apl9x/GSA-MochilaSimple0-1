import math
import random
import numpy as np
import pandas as pd

def calcularPeso(sol,values):
    peso = 0
    for i in range(N):
        peso = peso + sol[i]*(values[i][0])
    return peso

def calcularFitness(population,values,N,P):
    fitness=[]
    for i in range(P):
        valor = 0
        for j in range(N):
            valor = valor + population[i][j]*(values[j][1])
        fitness.append(valor)
    return fitness

def initialPopulation(values,N,P,W):
    population = []
    iteraciones = N
    for i in range(P):
        sol = [0]*N
        peso = 0  
        j = 0
        while peso < W and j < iteraciones:
            pos = random.randrange(0,N,1)
            sol[pos] = 1
            pesoAnt = peso
            peso = calcularPeso(sol,values)
            if peso > W:
                sol[pos] = 0
                peso = pesoAnt
            j=j+1
        population.append(sol)
    return population

def actualizarG(t,G0,alpha,maxIter):
    G = (-alpha * t)/maxIter
    G = math.exp(G)
    G = G * G0
    return G

def calcularMasas(fit,best,worst,P):
    m=[]
    M=[]
    for i in range(P):
        m.append((fitness[i]-worst)/(best-worst))
    for i in range(P):
        suma = 0
        for j in m:
            suma = suma + j   
        M.append(m[i]/suma)
    return M

def restPos(xi,xj,d):
    return xj[d] - xi[d]

def calcularR(x1,x2,N):
    suma=0
    for i in range(N):
        suma = suma + (x2[i]-x1[i])**2
    R=math.sqrt(suma)
    return R

def calcularFuerzas(P,N,G,M,population,e):
    FT=[]
    Ft=[]
    f=[]
    
    for i in range(P):
        Ft=[]
        for j in range(P):
            f=[]
            for d in range(N):
                value=0
                value = value+M[i]*M[j]
                value = value/(calcularR(population[i],population[j],N)+e)
                value = value*G*restPos(population[i],population[j],d)
                f.append(value)
                print
            Ft.append(f)
        FT.append(Ft)
    F=[]
    f=[]
    for i in range(P):
        f=[]
        for d in range(N): 
            value=0
            for j in range(P):
                if j != i :
                    r = random.random()
                    value = value +(r*np.array(FT)[i,j,d])
            f.append(value)
        F.append(f)
    return(F)

def calcularAceleracion(P,N,M,F):
    a=[]
    for i in range(P):
        vec=[]
        for j in range(N):
            if F[i][j] > 0 and M[i] > 0:
                v = F[i][j]/M[i]
                vec.append(v)
            else:
                vec.append(0)
        a.append(vec)
    return a

def calcularVelocidad(P,N,v,a):
    _v = []
    for i in range(P):
        vec=[]
        for j in range(N):
            value = random.random()
            value = value  * (v[i][j]+a[i][j])
            vec.append(value)
        _v.append(vec)
    return _v

def calcularPosiciones(P,N,population,v,values,W):
    for i in range(P):
        for j in range(N):
            if v[i][j] > 0 and population[i][j]==0:
                population[i][j]= 1
                if calcularPeso(population[i],values) > W:
                    population[i][j]= 0
            if v[i][j] < 0 and population[i][j]==1:
                population[i][j]= 0
    return population

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

[N,P,W,values] = leerCSV()
G0 = 100.0
population = initialPopulation(values,N,P,W)
v=list(np.zeros((P, N)))
best = []
worst = []
M=[]
F = []
a=[]
e=0.5
print('N: ',N)
print('P: ',P)
print('W: ',W)
print("Poblacion: ",population)

maxIter = (N//2)+ P
alpha = maxIter*0.2

for i in range(maxIter):
    fitness = calcularFitness(population,values,N,P)
    print("fitness: ",fitness)
    G = actualizarG(i,G0,alpha,maxIter)
    print("G: ",G)
    b = max(fitness)
    best = fitness.index(b)
    w = min(fitness)
    worst = fitness.index(w)
    print('Best: ' + str(population[best]) + str(fitness[best]))
    print('Worst: '+ str(population[worst]) + str(fitness[worst]))
    print('Media de Fitness: ',np.mean(fitness))
    print('Desviacion estandar: ',np.std(fitness))
    M = calcularMasas(fitness,b,w,P)
    print("Masas: ",M)
    F = calcularFuerzas(P,N,G,M,population,e)
    print("Fuerzas sobre cada agente: ")
    print(np.array(F))
    a = calcularAceleracion(P,N,M,F)
    print("Aceleracion: ",a)
    v = calcularVelocidad(P,N,v,a)
    print("Velocidad: ",v)
    population = calcularPosiciones(P,N,population,v,values,W)
    print("-------------------------------------------------------------------")
    print("Nueva Poblacion: ",population)