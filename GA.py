import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("smart_home_energy.csv")

data = load_data()

st.title("Smart Home Energy Optimization using Genetic Algorithm")

# Sidebar parameters
st.sidebar.header("GA Parameters")
POP_SIZE = st.sidebar.slider("Population Size", 10, 200, 50)
GENS = st.sidebar.slider("Generations", 10, 300, 100)
MUT_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

ENERGY = data.iloc[:, -1].values
GENE_LENGTH = len(ENERGY)

# GA functions
def init_population():
    return [np.random.randint(0, 2, GENE_LENGTH) for _ in range(POP_SIZE)]

def fitness(ind):
    energy_used = np.sum(ind * ENERGY)
    penalty = max(0, 5 - np.sum(ind)) * 100
    return energy_used + penalty

def selection(pop):
    return min(random.sample(pop, 3), key=fitness)

def crossover(p1, p2):
    point = random.randint(1, GENE_LENGTH - 1)
    return np.concatenate([p1[:point], p2[point:]])

def mutation(ind):
    for i in range(GENE_LENGTH):
        if random.random() < MUT_RATE:
            ind[i] = 1 - ind[i]
    return ind

# Run GA
if st.button("Run Optimization"):
    population = init_population()
    best_fitness = []
    
    for _ in range(GENS):
        new_pop = []
        for _ in range(POP_SIZE):
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2)
            child = mutation(child)
            new_pop.append(child)
        population = new_pop
        best = min(population, key=fitness)
        best_fitness.append(fitness(best))
    
    st.success("Optimization Complete")

    # Plot convergence
    fig, ax = plt.subplots()
    ax.plot(best_fitness)
    ax.set_title("GA Convergence Curve")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)

    st.write("Best Energy Consumption:", min(best_fitness))
