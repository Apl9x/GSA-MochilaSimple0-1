import math
import random
import numpy as np

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
    for i in range(P):
        sol = [0]*N
        peso = 0  
        j = 0
        while peso < W and j < 5:
            pos = random.randrange(0,4,1)
            sol[pos] = random.randrange(0,2,1)
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

def calcularFuerzas(P,N,G,M,population):
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
                value = value/(calcularR(population[i],population[j],N)+.1)
                value = value*G*restPos(population[i],population[j],d)
                f.append(value)
                print
            Ft.append(f)
        FT.append(Ft)
    F=[]
    # for i in range(P):
    #     value=0
    #     for j in range(P): 
    #         if j != i :
    #             for d in range(N):
    #                 r = random.random()
    #                 value = value +(r*np.array(FT)[i,j,d])
    #     F.append(value)
    return(FT)


values = [[2,3],[3,4],[4,5],[5,6]]
N = 4
P = 4
W = 5
G0 = 100.0
alpha = 2
maxIter = 10
population = initialPopulation(values,N,P,W)
best = []
worst = []
M=[]
F = []
print(population)

for i in range(1):
    fitness = calcularFitness(population,values,N,P)
    print(fitness)
    G = actualizarG(i,G0,alpha,maxIter)
    # print(G)
    b = max(fitness)
    best = fitness.index(b)
    w = min(fitness)
    worst = fitness.index(w)
    # print('Best: ' + str(population[best]) + str(fitness[best]))
    # print('Worst: '+ str(population[worst]) + str(fitness[worst]))
    M = calcularMasas(fitness,b,w,P)
    print(M)
    F = calcularFuerzas(P,N,G,M,population)
    print(np.array(F))