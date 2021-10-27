import math
import random

def calcPeso(sol,values):
    peso = 0
    for i in range(N):
        peso = peso + sol[i]*(values[i][0])
    return peso

def calcFitness(population,values,N,P):
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
            peso = calcPeso(sol,values)
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
print(population)

for i in range(maxIter):
    fitness = calcFitness(population,values,N,P)
    G = actualizarG(i,G0,alpha,maxIter)
    print(G)
    b = max(fitness)
    best = fitness.index(b)
    w = min(fitness)
    worst = fitness.index(w)
    print(population[best], fitness[best])
    print(population[worst], fitness[worst])
    