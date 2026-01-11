
import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
st.title("Smart Home Energy Scheduling using Genetic Algorithm")

@st.cache_data
def load_data():
    return pd.read_csv("project_benchmark_data.csv")

df = load_data()
st.subheader("Appliance Dataset")
st.dataframe(df)

# ----------------------------------------------------------
# 2. Split Appliances
# ----------------------------------------------------------
shiftable = df[df["Is_Shiftable"] == True].reset_index(drop=True)
non_shiftable = df[df["Is_Shiftable"] == False].reset_index(drop=True)

# ----------------------------------------------------------
# 3. Malaysia Time-of-Use Tariff
# ----------------------------------------------------------
PEAK_RATE = 0.50      # RM/kWh
OFFPEAK_RATE = 0.30  # RM/kWh

def get_tariff(hour):
    return PEAK_RATE if 8 <= hour < 22 else OFFPEAK_RATE

# ----------------------------------------------------------
# 4. Fixed Cost (Non-shiftable Appliances)
# ----------------------------------------------------------
fixed_cost = 0
for _, row in non_shiftable.iterrows():
    tariff = get_tariff(row["Preferred_Start_Hour"])
    fixed_cost += row["Avg_Power_kW"] * row["Duration_Hours"] * tariff

# ----------------------------------------------------------
# 5. GA Parameters (Sidebar)
# ----------------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 30)
GENERATIONS = st.sidebar.slider("Generations", 50, 300, 150)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.1, 0.95, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.15)
ALPHA = st.sidebar.slider("Discomfort Weight (α)", 0.1, 3.0, 1.0)

# ----------------------------------------------------------
# 6. Scheduling Windows (IMPORTANT FIX)
# ----------------------------------------------------------
WINDOW_SIZE = 3  # ±3 hours

def get_start_window(pref):
    return max(0, pref - WINDOW_SIZE)

def get_end_window(pref):
    return min(23, pref + WINDOW_SIZE)

# ----------------------------------------------------------
# 7. Genetic Algorithm Functions
# ----------------------------------------------------------
def create_individual():
    """Random valid schedule"""
    schedule = []
    for _, row in shiftable.iterrows():
        start = random.randint(
            get_start_window(row["Preferred_Start_Hour"]),
            get_end_window(row["Preferred_Start_Hour"])
        )
        schedule.append(start)
    return schedule

def fitness(individual):
    """Minimize cost + discomfort"""
    total_cost = fixed_cost
    discomfort = 0

    for i, start_time in enumerate(individual):
        row = shiftable.iloc[i]

        tariff = get_tariff(start_time)
        total_cost += row["Avg_Power_kW"] * row["Duration_Hours"] * tariff

        discomfort += abs(start_time - row["Preferred_Start_Hour"])

    return total_cost + ALPHA * discomfort

def selection(population):
    contenders = random.sample(population, 3)
    return min(contenders, key=fitness)

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1, p2

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            pref = shiftable.iloc[i]["Preferred_Start_Hour"]
            individual[i] = random.randint(
                get_start_window(pref),
                get_end_window(pref)
            )
    return individual

# ----------------------------------------------------------
# 8. Run Optimization
# ----------------------------------------------------------
if st.button("Run Optimization"):

    population = [create_individual() for _ in range(POP_SIZE)]
    best_history = []

    for _ in range(GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE // 2):
            p1 = selection(population)
            p2 = selection(population)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            new_population.append(mutate(c2))

        population = new_population
        best = min(population, key=fitness)
        best_history.append(fitness(best))

    best_solution = min(population, key=fitness)
    best_cost = fitness(best_solution)

    # ----------------------------------------------------------
    # 9. Display Results
    # ----------------------------------------------------------
    st.subheader("Optimized Schedule")

    result = []
    for i, start in enumerate(best_solution):
        row = shiftable.iloc[i]
        result.append({
            "Appliance": row["Appliance"],
            "Preferred Start": row["Preferred_Start_Hour"],
            "Scheduled Start": start,
            "End Time": min(start + row["Duration_Hours"], 24),
            "Duration (h)": row["Duration_Hours"],
            "Power (kW)": row["Avg_Power_kW"]
        })

    st.dataframe(pd.DataFrame(result))
    st.metric("Total Optimized Cost (RM)", f"{best_cost:.2f}")

    # ----------------------------------------------------------
    # 10. Convergence Curve
    # ----------------------------------------------------------
    st.subheader("GA Convergence Curve")

    fig, ax = plt.subplots()
    ax.plot(best_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value")
    ax.set_title("GA Convergence")
    st.pyplot(fig)
