import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================================
# 1. Page Configuration & Professional Branding
# ==========================================================
st.set_page_config(page_title="EcoSmart Home Optimizer", layout="wide")
st.title("🏡 EcoSmart: AI Home Energy Scheduler")

st.sidebar.markdown("""
## About EcoSmart
This tool uses a **Genetic Algorithm** to find the most cost-effective schedule for your home appliances. 

By shifting flexible tasks to off-peak hours and balancing your total power load, it helps you save on electricity bills while staying within your home's power safety limits.
""")

# ==========================================================
# 2. Data Loading
# ==========================================================
@st.cache_data
def load_data():
    try:
        # Load the provided benchmark data
        df = pd.read_csv("project_benchmark_data.csv")
        df['Shiftable'] = df['Is_Shiftable'].astype(bool)
        df['Power_kW'] = df['Avg_Power_kW']
        df['Duration'] = df['Duration_Hours']
        df['Preferred'] = df['Preferred_Start_Hour']
        return df
    except:
        st.error("Error: 'project_benchmark_data.csv' not found. Please upload the file.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    shiftable_apps = df[df['Shiftable']].reset_index(drop=True)
    fixed_apps = df[~df['Shiftable']].reset_index(drop=True)

    # ==========================================================
    # 3. User Settings & Tariff Information
    # ==========================================================
    st.sidebar.header("Optimization Settings")
    alpha = st.sidebar.slider("Flexibility Weight", 0.0, 5.0, 0.5, 
                              help="Higher values prioritize your preferred times; lower values prioritize maximum savings.")
    
    # Malaysia Standard Peak/Off-Peak (ToU)
    MAX_POWER_LIMIT = 5.0  
    TARIFF_PEAK = 0.50     # RM/kWh (08:00 - 22:00)
    TARIFF_OFFPEAK = 0.30  # RM/kWh

    tariff_array = np.array([TARIFF_PEAK if 8 <= h < 22 else TARIFF_OFFPEAK for h in range(24)])

    # ==========================================================
    # 4. Core Optimization Logic (Genetic Algorithm)
    # ==========================================================
    def calculate_fitness(individual, p_limit=MAX_POWER_LIMIT):
        total_discomfort = 0
        hourly_power = np.zeros(24)

        # Fixed Load (Fridge, etc.)
        for _, r in fixed_apps.iterrows():
            hours = np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24
            hourly_power[hours] += r['Power_kW']

        # Flexible Load (Washing Machine, EV, etc.)
        for i, start in enumerate(individual):
            r = shiftable_apps.iloc[i]
            total_discomfort += abs(start - r['Preferred'])
            hours = np.arange(start, start + r['Duration'], dtype=int) % 24
            hourly_power[hours] += r['Power_kW']

        cost = np.sum(hourly_power * tariff_array)
        
        # Penalize if total power exceeds safety limit
        peak_exceeded = np.clip(hourly_power - p_limit, 0, None)
        peak_penalty = np.sum(peak_exceeded**2) * 1000 

        return cost + (alpha * total_discomfort) + peak_penalty

    def run_optimizer(p_limit=MAX_POWER_LIMIT):
        pop_size = 100
        generations = 200
        population = [np.random.randint(0, 24, size=len(shiftable_apps)).tolist() for _ in range(pop_size)]
        history = []
        
        for gen in range(generations):
            fitness_scores = [calculate_fitness(ind, p_limit) for ind in population]
            best_idx = np.argmin(fitness_scores)
            history.append(fitness_scores[best_idx])

            new_pop = [population[best_idx]] # Elitism

            while len(new_pop) < pop_size:
                # Tournament Selection
                p1 = population[np.argmin([fitness_scores[i] for i in random.sample(range(pop_size), 3)])]
                p2 = population[np.argmin([fitness_scores[i] for i in random.sample(range(pop_size), 3)])]
                
                # Crossover & Mutation
                cut = random.randint(1, len(p1)-1) if len(p1) > 1 else 0
                child = p1[:cut] + p2[cut:]
                if random.random() < 0.1: # 10% Mutation
                    child[random.randint(0, len(child)-1)] = random.randint(0, 23)
                new_pop.append(child)
            population = new_pop

        return population[0], history

    # ==========================================================
    # 5. Dashboard & Results
    # ==========================================================
    if st.button("🚀 Calculate Best Schedule"):
        with st.spinner("Analyzing energy patterns..."):
            best_ind, hist = run_optimizer()

        # Recalculate Final State
        final_load = np.zeros(24)
        for _, r in fixed_apps.iterrows():
            final_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']
        for i, start in enumerate(best_ind):
            r = shiftable_apps.iloc[i]
            final_load[np.arange(start, start + r['Duration'], dtype=int) % 24] += r['Power_kW']

        # Baseline for Comparison
        baseline_load = np.zeros(24)
        for _, r in df.iterrows():
            baseline_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']

        cost_opt = np.sum(final_load * tariff_array)
        cost_base = np.sum(baseline_load * tariff_array)

        # Performance Metrics
        st.subheader("Summary of Savings")
        m1, m2, m3 = st.columns(3)
        m1.metric("Daily Savings", f"RM {cost_base - cost_opt:.2f}")
        m2.metric("Peak Load", f"{np.max(final_load):.2f} kW")
        m3.metric("Monthly Est. Savings", f"RM {(cost_base - cost_opt)*30:.2f}")

        # Power Profile Chart
        st.subheader("24-Hour Power Consumption Profile")
        
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(range(24), final_load, color='#3498db', alpha=0.7, label="Optimized Usage")
        ax1.axhline(MAX_POWER_LIMIT, color='#e74c3c', linestyle='--', label="Safety Limit")
        ax1.set_ylabel("Power Demand (kW)")
        ax1.set_xlabel("Time of Day (Hour)")
        ax1.set_xticks(range(24))

        ax2 = ax1.twinx()
        ax2.step(range(24), tariff_array, where='post', color='#f39c12', linewidth=2, label="Tariff Rate")
        ax2.set_ylabel("Electricity Rate (RM/kWh)", color='#f39c12')
        
        plt.title("How Energy Use Was Shifted to Cheaper Off-Peak Hours")
        st.pyplot(fig)

        # Detailed Table
        st.subheader("Your Recommended Schedule")
        schedule_list = []
        for i, start in enumerate(best_ind):
            r = shiftable_apps.iloc[i]
            schedule_list.append({
                "Appliance": r['Appliance'],
                "Original Start": f"{r['Preferred']}:00",
                "New Start Time": f"{start}:00",
                "End Time": f"{(start + int(r['Duration'])) % 24}:00",
                "Saving Potential": "High" if (r['Preferred'] >= 8 and r['Preferred'] < 22 and (start < 8 or start >= 22)) else "Low"
            })
        st.table(pd.DataFrame(schedule_list))

    # ==========================================================
    # 6. Advanced Insights (What-if Scenario)
    # ==========================================================
    st.divider()
    st.header("💡 Insights: Impact of Power Limits")
    st.info("If you have many appliances running at once, the system might have to pick more expensive times to avoid tripping your breaker.")

    if st.button("🔍 Analyze Power vs. Cost"):
        test_limits = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        results = []
        
        with st.spinner("Simulating different scenarios..."):
            for limit in test_limits:
                best_s, _ = run_optimizer(p_limit=limit)
                res_load = np.zeros(24)
                # Quick load calculation
                for _, r in fixed_apps.iterrows():
                    res_load[np.arange(r['Preferred'], r['Preferred'] + r['Duration'], dtype=int) % 24] += r['Power_kW']
                for i, start in enumerate(best_s):
                    r = shiftable_apps.iloc[i]
                    res_load[np.arange(start, start + r['Duration'], dtype=int) % 24] += r['Power_kW']
                
                results.append({"Max Limit (kW)": limit, "Daily Cost (RM)": np.sum(res_load * tariff_array)})

        st.line_chart(pd.DataFrame(results).set_index("Max Limit (kW)"))
        st.write("This chart shows that as you allow for a higher **Max Limit**, the system has more freedom to group appliances in the cheapest hours, lowering your total cost.")
