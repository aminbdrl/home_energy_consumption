import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================================
# 1. Page Configuration & Objectives
# ==========================================================
st.set_page_config(page_title="Smart Home Energy GA", layout="wide")
st.title("⚡ Smart Home Energy Scheduling Optimization")

st.sidebar.markdown

# ==========================================================
# 2. Data Loading
# ==========================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("project_benchmark_data.csv")
        df['Shiftable'] = df['Is_Shiftable'].astype(bool)
        df['Power_kW'] = df['Avg_Power_kW']
        df['Duration'] = df['Duration_Hours']
        df['Preferred'] = df['Preferred_Start_Hour']
        return df
    except:
        # Placeholder for demonstration if CSV is missing
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Please ensure 'project_benchmark_data.csv' is in the directory.")
else:
    shiftable_apps = df[df['Shiftable']].reset_index(drop=True)
    fixed_apps = df[~df['Shiftable']].reset_index(drop=True)

    # ==========================================================
    # 3. Parameters & Tariff
    # ==========================================================
    st.sidebar.header("GA Hyperparameters")
    pop_size = st.sidebar.slider("Population Size", 30, 200, 100)
    generations = st.sidebar.slider("Generations", 50, 500, 250)
    alpha = st.sidebar.slider("Discomfort Weight (α)", 0.0, 5.0, 0.5)

    MAX_POWER_LIMIT = 5.0  
    TARIFF_PEAK = 0.50     # RM/kWh (08:00 - 22:00)
    TARIFF_OFFPEAK = 0.30  # RM/kWh

    tariff_array = np.array([TARIFF_PEAK if 8 <= h < 22 else TARIFF_OFFPEAK for h in range(24)])

    # ==========================================================
    # 4. GA Logic
    # ==========================================================
    def calculate_fitness(individual, p_limit=MAX_POWER_LIMIT):
        total_discomfort = 0
        hourly_power = np.zeros(24)

        # Fixed Load
        for _, r in fixed_apps.iterrows():
            hours = np.arange(r['Preferred'], r['Preferred'] + r['Duration']) % 24
            hourly_power[hours.astype(int)] += r['Power_kW']

        # Shifted Load
        for i, start in enumerate(individual):
            r = shiftable_apps.iloc[i]
            total_discomfort += abs(start - r['Preferred'])
            hours = np.arange(start, start + r['Duration']) % 24
            hourly_power[hours.astype(int)] += r['Power_kW']

        total_cost = np.sum(hourly_power * tariff_array)
        
        # Penalizing power limit violations
        peak_exceeded = np.clip(hourly_power - p_limit, 0, None)
        peak_penalty = np.sum(peak_exceeded**2) * 1000 

        return total_cost + (alpha * total_discomfort) + peak_penalty

    def run_ga(p_limit=MAX_POWER_LIMIT):
        population = [np.random.randint(0, 24, size=len(shiftable_apps)).tolist() for _ in range(pop_size)]
        best_history = []
        
        for gen in range(generations):
            fitness_scores = [calculate_fitness(ind, p_limit) for ind in population]
            best_idx = np.argmin(fitness_scores)
            best_history.append(fitness_scores[best_idx])

            new_pop = [population[best_idx]] 

            while len(new_pop) < pop_size:
                p1 = population[np.argmin([fitness_scores[i] for i in random.sample(range(pop_size), 3)])]
                p2 = population[np.argmin([fitness_scores[i] for i in random.sample(range(pop_size), 3)])]
                cut = random.randint(1, len(p1)-1) if len(p1) > 1 else 0
                child = p1[:cut] + p2[cut:]
                if random.random() < 0.1:
                    child[random.randint(0, len(child)-1)] = random.randint(0, 23)
                new_pop.append(child)
            population = new_pop

        final_scores = [calculate_fitness(ind, p_limit) for ind in population]
        return population[np.argmin(final_scores)], best_history

    # ==========================================================
    # 5. Main Execution
    # ==========================================================
    if st.button("🚀 Run Optimization"):
        best_schedule, history = run_ga()

        # Load Calculations
        final_load = np.zeros(24)
        for _, r in fixed_apps.iterrows():
            final_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']
        for i, start in enumerate(best_schedule):
            r = shiftable_apps.iloc[i]
            final_load[np.arange(start, start + r['Duration'], dtype=int) % 24] += r['Power_kW']

        # Baseline Calculation
        baseline_load = np.zeros(24)
        for _, r in df.iterrows():
            baseline_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']

        # Metrics
        cost_opt = np.sum(final_load * tariff_array)
        cost_base = np.sum(baseline_load * tariff_array)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Savings", f"RM {cost_base - cost_opt:.2f}")
        col2.metric("Max Load", f"{np.max(final_load):.2f} kW")
        col3.metric("Final Cost", f"RM {cost_opt:.2f}")

        # Visualizations
        st.subheader("📊 Load Profile vs Tariff")
        
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(range(24), final_load, color='skyblue', alpha=0.8, label="Optimized Load")
        ax1.axhline(MAX_POWER_LIMIT, color='red', linestyle='--', label="Limit (5kW)")
        ax1.set_ylabel("Power (kW)")
        ax1.set_xticks(range(24))

        ax2 = ax1.twinx()
        ax2.step(range(24), tariff_array, where='post', color='orange', label="Tariff RM")
        ax2.set_ylabel("Price (RM/kWh)", color='orange')
        st.pyplot(fig)

        st.subheader("📈 Convergence Analysis")
        st.line_chart(history)

    # ==========================================================
    # 6. Sensitivity Analysis (Extended Analysis)
    # ==========================================================
    st.divider()
    st.header("🔍 Sensitivity Analysis: Constraint vs. Cost")
    st.write("Analyzing how strict power limits increase daily electricity costs.")

    if st.button("📊 Generate Sensitivity Report"):
        test_limits = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        results = []
        
        with st.spinner("Calculating trade-offs..."):
            for limit in test_limits:
                best_s, _ = run_ga(p_limit=limit)
                # Quick load calculation for result
                res_load = np.zeros(24)
                for _, r in fixed_apps.iterrows():
                    res_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']
                for i, start in enumerate(best_s):
                    r = shiftable_apps.iloc[i]
                    res_load[np.arange(start, start + r['Duration'], dtype=int) % 24] += r['Power_kW']
                
                res_cost = np.sum(res_load * tariff_array)
                results.append({"Limit (kW)": limit, "Cost (RM)": res_cost})

        sensitivity_df = pd.DataFrame(results)
        
        st.line_chart(sensitivity_df.set_index("Limit (kW)"))
        st.dataframe(sensitivity_df)
