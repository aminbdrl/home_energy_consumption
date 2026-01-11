import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
st.set_page_config(page_title="Smart Home Energy Scheduling", layout="wide")
st.title("Smart Home Energy Scheduling using Genetic Algorithm")

@st.cache_data
def load_data():
    # Load the uploaded dataset
    df = pd.read_csv("project_benchmark_data.csv")
    return df

df = load_data()
st.subheader("Appliance Dataset")
st.write("This dataset contains power ratings and preferred start times for various home appliances.")
st.dataframe(df)

# Separate appliances based on the 'Is_Shiftable' column
shiftable = df[df["Is_Shiftable"] == True].reset_index(drop=True)
non_shiftable = df[df["Is_Shiftable"] == False].reset_index(drop=True)

# ----------------------------------------------------------
# 2. Malaysia Time-of-Use Tariff
# ----------------------------------------------------------
TARIFF_PEAK = 0.50      # RM/kWh (8:00 AM - 10:00 PM)
TARIFF_OFFPEAK = 0.30   # RM/kWh (10:00 PM - 8:00 AM)

def get_tariff(hour):
    if 8 <= hour < 22:
        return TARIFF_PEAK
    return TARIFF_OFFPEAK

# ----------------------------------------------------------
# 3. Fixed Cost (Non-shiftable Appliances)
# ----------------------------------------------------------
fixed_cost = 0
for _, row in non_shiftable.iterrows():
    # Calculation based on preferred start time as they are non-shiftable
    tariff = get_tariff(row["Preferred_Start_Hour"])
    fixed_cost += row["Avg_Power_kW"] * row["Duration_Hours"] * tariff

# ----------------------------------------------------------
# 4. GA Parameters (Streamlit Controls)
# ----------------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 50)
GENERATIONS = st.sidebar.slider("Generations", 20, 500, 200)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.1, 0.95, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
ALPHA = st.sidebar.slider("Discomfort Weight (α)", 0.0, 2.0, 0.5)

# ----------------------------------------------------------
# 5. Genetic Algorithm Functions
# ----------------------------------------------------------
def create_individual():
    """Create a chromosome of start times (0-23) for shiftable appliances."""
    individual = []
    for _ in range(len(shiftable)):
        # Allowed window is the full 24-hour cycle
        individual.append(random.randint(0, 23))
    return individual

def fitness(individual):
    """Fitness = Cost + α * Discomfort"""
    total_cost = fixed_cost
    discomfort = 0

    for i, start_time in enumerate(individual):
        row = shiftable.iloc[i]
        
        # Calculate Energy Cost
        tariff = get_tariff(start_time)
        total_cost += row["Avg_Power_kW"] * row["Duration_Hours"] * tariff
        
        # Calculate Discomfort (difference from preferred start hour)
        discomfort += abs(start_time - row["Preferred_Start_Hour"])

    return total_cost + (ALPHA * discomfort)

def selection(population):
    """Tournament selection"""
    contenders = random.sample(population, 3)
    return min(contenders, key=fitness)

def crossover(parent1, parent2):
    """Single-point crossover"""
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1) if len(parent1) > 1 else 0
        return (
            parent1[:point] + parent2[point:],
            parent2[:point] + parent1[point:]
        )
    return parent1, parent2

def mutate(individual):
    """Random mutation: change start time of an appliance"""
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = random.randint(0, 23)
    return individual

# ----------------------------------------------------------
# 6. Run Genetic Algorithm
# ----------------------------------------------------------
if st.button("Run Energy Optimization"):
    # Initialize Population
    population = [create_individual() for _ in range(POP_SIZE)]
    best_fitness_history = []

    # Evolution Loop
    for _ in range(GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE // 2):
            p1, p2 = selection(population), selection(population)
            c1, c2 = crossover(p1, p2)
            new_population.extend([mutate(c1), mutate(c2)])
        
        population = new_population
        best_ind = min(population, key=fitness)
        best_fitness_history.append(fitness(best_ind))

    # Best Solution Found
    best_solution = min(population, key=fitness)
    
    # ----------------------------------------------------------
    # 7. Display Results
    # ----------------------------------------------------------
    st.subheader("Optimized Appliance Schedule")
    
    final_results = []
    # Add non-shiftable results
    for _, row in non_shiftable.iterrows():
        final_results.append({
            "Appliance": row["Appliance"],
            "Status": "Fixed",
            "Preferred Start": row["Preferred_Start_Hour"],
            "Scheduled Start": row["Preferred_Start_Hour"],
            "End Time": (row["Preferred_Start_Hour"] + row["Duration_Hours"]) % 24,
            "Power (kW)": row["Avg_Power_kW"]
        })

    # Add optimized shiftable results
    for i, start in enumerate(best_solution):
        row = shiftable.iloc[i]
        final_results.append({
            "Appliance": row["Appliance"],
            "Status": "Shifted",
            "Preferred Start": row["Preferred_Start_Hour"],
            "Scheduled Start": start,
            "End Time": (start + row["Duration_Hours"]) % 24,
            "Power (kW)": row["Avg_Power_kW"]
        })

    result_df = pd.DataFrame(final_results)
    st.dataframe(result_df)

    col1, col2 = st.columns(2)
    col1.metric("Total Optimized Cost", f"RM {fitness(best_solution):.2f}")
    
    # Calculate initial cost for comparison
    initial_cost = fixed_cost + sum([s["Avg_Power_kW"] * s["Duration_Hours"] * get_tariff(s["Preferred_Start_Hour"]) for _, s in shiftable.iterrows()])
    savings = initial_cost - fitness(best_solution)
    col2.metric("Estimated Savings", f"RM {max(0, savings):.2f}")

    # ----------------------------------------------------------
    # 8. Visualizations
    # ----------------------------------------------------------
    st.subheader("Optimization Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot Convergence
    ax[0].plot(best_fitness_history, color='blue')
    ax[0].set_title("GA Convergence Curve")
    ax[0].set_xlabel("Generation")
    ax[0].set_ylabel("Fitness (Cost + Discomfort)")
    
    # Plot Schedule Comparison
    labels = result_df['Appliance']
    pref = result_df['Preferred Start']
    sched = result_df['Scheduled Start']
    x = np.arange(len(labels))
    ax[1].bar(x - 0.2, pref, 0.4, label='Preferred')
    ax[1].bar(x + 0.2, sched, 0.4, label='Scheduled')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=45)
    ax[1].set_title("Start Time Comparison")
    ax[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
