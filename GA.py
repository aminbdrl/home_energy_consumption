import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================================
# 1. Page Configuration & Header
# ==========================================================
st.set_page_config(page_title="EcoSmart Home Energy Optimizer", layout="wide")
st.title("⚡ Smart Home Energy Scheduling Optimization")
st.markdown("""
Optimize your appliance schedule to minimize costs and balance power demand using a Genetic Algorithm.
""")

# ==========================================================
# 2. Data Loading
# ==========================================================
@st.cache_data
def load_data():
    try:
        # Expects columns: Appliance, Avg_Power_kW, Duration_Hours, Preferred_Start_Hour, Is_Shiftable
        df = pd.read_csv("project_benchmark_data.csv")
        df['Shiftable'] = df['Is_Shiftable'].astype(bool)
        df['Power_kW'] = df['Avg_Power_kW']
        df['Duration'] = df['Duration_Hours']
        df['Preferred'] = df['Preferred_Start_Hour']
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Please ensure 'project_benchmark_data.csv' is in the directory.")
else:
    shiftable_apps = df[df['Shiftable']].reset_index(drop=True)
    fixed_apps = df[~df['Shiftable']].reset_index(drop=True)

    # ==========================================================
    # 3. Parameters & Tariff Configuration
    # ==========================================================
    st.sidebar.header("Optimization Settings")
    pop_size = st.sidebar.slider("Population Size", 30, 200, 100)
    generations = st.sidebar.slider("Generations", 50, 500, 250)
    alpha = st.sidebar.slider("Discomfort Weight (α)", 0.0, 5.0, 0.5, 
                              help="Higher values prioritize your preferred start times.")

    MAX_POWER_LIMIT = 5.0  # Total household kW limit
    TARIFF_PEAK = 0.50     # RM/kWh (Peak: 08:00 - 22:00)
    TARIFF_OFFPEAK = 0.30  # RM/kWh (Off-Peak)

    # Precompute tariff array for 24 hours
    tariff_array = np.array([TARIFF_PEAK if 8 <= h < 22 else TARIFF_OFFPEAK for h in range(24)])

    # ==========================================================
    # 4. Optimization Engine (Genetic Algorithm)
    # ==========================================================
    def calculate_fitness(individual, p_limit=MAX_POWER_LIMIT):
        total_discomfort = 0
        hourly_power = np.zeros(24)

        # Apply Fixed Load (Non-shiftable)
        for _, r in fixed_apps.iterrows():
            hours = np.arange(r['Preferred'], r['Preferred'] + r['Duration']) % 24
            hourly_power[hours.astype(int)] += r['Power_kW']

        # Apply Scheduled Load (GA Output)
        for i, start in enumerate(individual):
            r = shiftable_apps.iloc[i]
            total_discomfort += abs(start - r['Preferred'])
            hours = np.arange(start, start + r['Duration']) % 24
            hourly_power[hours.astype(int)] += r['Power_kW']

        energy_cost = np.sum(hourly_power * tariff_array)
        
        # Power Limit Penalty (Exponential to enforce strict adherence)
        peak_exceeded = np.clip(hourly_power - p_limit, 0, None)
        peak_penalty = np.sum(peak_exceeded**2) * 1000 

        return energy_cost + (alpha * total_discomfort) + peak_penalty

    def run_ga(p_limit=MAX_POWER_LIMIT):
        # Initial Population
        population = [np.random.randint(0, 24, size=len(shiftable_apps)).tolist() for _ in range(pop_size)]
        best_history = []
        
        for gen in range(generations):
            fitness_scores = [calculate_fitness(ind, p_limit) for ind in population]
            best_idx = np.argmin(fitness_scores)
            best_history.append(fitness_scores[best_idx])

            # Elitism: Carry the best individual forward
            new_pop = [population[best_idx]] 

            while len(new_pop) < pop_size:
                # Tournament Selection
                p1 = population[np.argmin([fitness_scores[i] for i in random.sample(range(pop_size), 3)])]
                p2 = population[np.argmin([fitness_scores[i] for i in random.sample(range(pop_size), 3)])]
                
                # Single-point Crossover
                cut = random.randint(1, len(p1)-1) if len(p1) > 1 else 0
                child = p1[:cut] + p2[cut:]
                
                # Mutation
                if random.random() < 0.15:
                    child[random.randint(0, len(child)-1)] = random.randint(0, 23)
                
                new_pop.append(child)
            population = new_pop

        final_scores = [calculate_fitness(ind, p_limit) for ind in population]
        return population[np.argmin(final_scores)], best_history

    # ==========================================================
    # 5. Dashboard Results
    # ==========================================================
    if st.button("🚀 Run Optimization"):
        with st.spinner("Processing energy patterns..."):
            best_schedule, history = run_ga()

        # Final Load Calculation
        final_load = np.zeros(24)
        for _, r in fixed_apps.iterrows():
            final_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']
        for i, start in enumerate(best_schedule):
            r = shiftable_apps.iloc[i]
            final_load[np.arange(start, start + r['Duration'], dtype=int) % 24] += r['Power_kW']

        # Baseline Calculation (Preferred Times)
        baseline_load = np.zeros(24)
        for _, r in df.iterrows():
            baseline_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']

        # Financial Impact
        cost_opt = np.sum(final_load * tariff_array)
        cost_base = np.sum(baseline_load * tariff_array)
        
        st.subheader("Financial & Operational Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Daily Savings", f"RM {cost_base - cost_opt:.2f}")
        col2.metric("Peak Demand", f"{np.max(final_load):.2f} kW")
        col3.metric("Optimized Total Cost", f"RM {cost_opt:.2f}")

        # Visualization: Dual-Axis Load Profile
        st.subheader("📊 Load Distribution vs. Electricity Tariff")
        
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(range(24), final_load, color='skyblue', alpha=0.8, label="Optimized Load")
        ax1.axhline(MAX_POWER_LIMIT, color='red', linestyle='--', label=f"Safe Limit ({MAX_POWER_LIMIT}kW)")
        ax1.set_ylabel("Power Demand (kW)")
        ax1.set_xlabel("Hour of Day")
        ax1.set_xticks(range(24))

        ax2 = ax1.twinx()
        ax2.step(range(24), tariff_array, where='post', color='orange', linewidth=2, label="Tariff Rate")
        ax2.set_ylabel("Tariff (RM/kWh)", color='orange')
        
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        st.pyplot(fig)

        st.subheader("📈 Optimization Convergence")
        st.line_chart(history)

    # ==========================================================
    # 6. Trade-off Analysis
    # ==========================================================
    st.divider()
    st.header("🔍 Capacity Sensitivity Analysis")
    st.write("Understand how lowering your peak power threshold impacts your daily electricity costs.")

    if st.button("📊 Analyze Capacity Constraints"):
        test_limits = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        results = []
        
        with st.spinner("Testing scenarios..."):
            for limit in test_limits:
                best_s, _ = run_ga(p_limit=limit)
                res_load = np.zeros(24)
                for _, r in fixed_apps.iterrows():
                    res_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']
                for i, start in enumerate(best_s):
                    r = shiftable_apps.iloc[i]
                    res_load[np.arange(start, start + r['Duration'], dtype=int) % 24] += r['Power_kW']
                
                res_cost = np.sum(res_load * tariff_array)
                results.append({"Limit (kW)": limit, "Daily Cost (RM)": res_cost})

        sensitivity_df = pd.DataFrame(results)
        st.line_chart(sensitivity_df.set_index("Limit (kW)"))
        st.table(sensitivity_df)
