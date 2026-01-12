import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================================
# 1. Page Configuration
# ==========================================================
st.set_page_config(page_title="EcoSmart Home Energy GA", layout="wide")
st.title("⚡ Smart Home Energy Scheduling Optimization")

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
    st.sidebar.header("Optimization Settings")
    pop_size = st.sidebar.slider("Population Size", 30, 200, 100)
    generations = st.sidebar.slider("Generations", 50, 500, 250)
    alpha = st.sidebar.slider("Discomfort Weight (α)", 0.0, 5.0, 0.5)

    MAX_POWER_LIMIT = 5.0  
    TARIFF_PEAK = 0.50     
    TARIFF_OFFPEAK = 0.30  

    tariff_array = np.array([TARIFF_PEAK if 8 <= h < 22 else TARIFF_OFFPEAK for h in range(24)])

    # ==========================================================
    # 4. GA Logic (Optimized for Speed)
    # ==========================================================
    def calculate_fitness(individual, p_limit, current_alpha):
        hourly_power = np.zeros(24)
        total_discomfort = 0

        # Non-shiftable load
        for _, r in fixed_apps.iterrows():
            hours = np.arange(r['Preferred'], r['Preferred'] + r['Duration']) % 24
            hourly_power[hours.astype(int)] += r['Power_kW']

        # Shiftable load
        for i, start in enumerate(individual):
            r = shiftable_apps.iloc[i]
            total_discomfort += abs(start - r['Preferred'])
            hours = np.arange(start, start + r['Duration']) % 24
            hourly_power[hours.astype(int)] += r['Power_kW']

        cost = np.sum(hourly_power * tariff_array)
        peak_penalty = np.sum(np.clip(hourly_power - p_limit, 0, None)**2) * 1000 

        return cost + (current_alpha * total_discomfort) + peak_penalty

    def run_ga(p_limit, g_size, p_count):
        population = [np.random.randint(0, 24, size=len(shiftable_apps)).tolist() for _ in range(p_count)]
        best_history = []
        
        for _ in range(g_size):
            fitness_scores = [calculate_fitness(ind, p_limit, alpha) for ind in population]
            best_idx = np.argmin(fitness_scores)
            best_history.append(fitness_scores[best_idx])
            
            new_pop = [population[best_idx]]
            while len(new_pop) < p_count:
                p1 = population[np.argmin([fitness_scores[i] for i in random.sample(range(p_count), 3)])]
                p2 = population[np.argmin([fitness_scores[i] for i in random.sample(range(p_count), 3)])]
                cut = random.randint(1, len(p1)-1) if len(p1) > 1 else 0
                child = p1[:cut] + p2[cut:]
                if random.random() < 0.1:
                    child[random.randint(0, len(child)-1)] = random.randint(0, 23)
                new_pop.append(child)
            population = new_pop
        return population[0], best_history

    # ==========================================================
    # 5. Execution & Automatic Sensitivity Analysis
    # ==========================================================
    if st.button("🚀 Run Full Optimization"):
        # Run Main GA
        with st.spinner("Finding optimal schedule..."):
            best_schedule, history = run_ga(MAX_POWER_LIMIT, generations, pop_size)

        # Show Results Metrics
        final_load = np.zeros(24)
        for _, r in fixed_apps.iterrows():
            final_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']
        for i, start in enumerate(best_schedule):
            r = shiftable_apps.iloc[i]
            final_load[np.arange(start, start + r['Duration'], dtype=int) % 24] += r['Power_kW']

        st.subheader("Results Summary")
        col1, col2 = st.columns(2)
        col1.metric("Optimized Daily Cost", f"RM {np.sum(final_load * tariff_array):.2f}")
        col2.metric("Peak Power Recorded", f"{np.max(final_load):.2f} kW")

        # Load Profile Plot
        
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(range(24), final_load, color='skyblue', label="Load (kW)")
        ax1.axhline(MAX_POWER_LIMIT, color='red', linestyle='--', label="Limit")
        ax1.set_ylabel("Power (kW)")
        ax2 = ax1.twinx()
        ax2.step(range(24), tariff_array, where='post', color='orange', label="Tariff")
        ax2.set_ylabel("RM/kWh")
        st.pyplot(fig)

        # --- AUTOMATIC SENSITIVITY ANALYSIS (FASTER) ---
        st.divider()
        st.subheader("🔍 Capacity Sensitivity Analysis")
        st.write("Automatic comparison of different household power limits:")
        
        test_limits = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        results = []
        
        # Use a "Lite" GA for the sensitivity analysis (50 gen, 50 pop) to ensure speed
        with st.spinner("Analyzing trade-offs..."):
            for limit in test_limits:
                # Running a smaller version of GA for speed
                s_best, _ = run_ga(limit, 50, 50) 
                s_load = np.zeros(24)
                for _, r in fixed_apps.iterrows():
                    s_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']
                for i, start in enumerate(s_best):
                    r = shiftable_apps.iloc[i]
                    s_load[np.arange(start, start + r['Duration'], dtype=int) % 24] += r['Power_kW']
                results.append({"Limit (kW)": limit, "Daily Cost (RM)": np.sum(s_load * tariff_array)})

        sensitivity_df = pd.DataFrame(results)
        
        st.line_chart(sensitivity_df.set_index("Limit (kW)"))
        st.dataframe(sensitivity_df)
