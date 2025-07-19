import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

class EnergySystemCalculator:
    def __init__(self, consumption_profile: pd.DataFrame):
        self.consumption_profile = consumption_profile.set_index('timestamp')

    def simulate_pv_generation(self, pv_size_kw: float, years: int = 1) -> pd.Series:
        annual_generation_per_kw = 1000
        system_losses = 0.15
        degradation_rate = (1 - 0.85 ** (1/25))

        base_generation = pv_size_kw * annual_generation_per_kw * (1 - system_losses)
        hourly_generation = base_generation / 8760

        np.random.seed(42)
        seasonal_factor = np.tile(
            np.clip(np.sin(np.linspace(0, np.pi, 24)), 0, 1),
            len(self.consumption_profile) // 24
        )

        degradation_multiplier = (1 - degradation_rate) ** (years - 1)
        return pd.Series(hourly_generation * seasonal_factor * degradation_multiplier, index=self.consumption_profile.index)

    def calculate_autoconsumption(self, pv_generation: pd.Series, storage_power_kw: float, storage_capacity_kwh: float, battery_cycles: int = 4000):
        consumption = self.consumption_profile['consumption_kWh']
        storage_level = 0
        autokonsumed = 0
        battery_degradation = 0.2

        adjusted_capacity = storage_capacity_kwh * (1 - battery_degradation)
        storage_levels = []

        for time, pv_gen in pv_generation.items():
            demand = consumption.loc[time]
            excess = max(pv_gen - demand, 0)
            direct_use = min(pv_gen, demand)

            if excess > 0:
                charge = min(adjusted_capacity - storage_level, min(storage_power_kw, excess))
                storage_level += charge

            else:
                needed = demand - direct_use
                discharge = min(storage_level, min(storage_power_kw, needed))
                storage_level -= discharge
                direct_use += discharge

            autokonsumed += direct_use
            storage_levels.append((time, storage_level))

        storage_profile = pd.Series(dict(storage_levels))
        return autokonsumed / consumption.sum(), storage_profile

    def optimize_system(self, min_autoconsumption: float = 0.9):
        pv_size_kw = 10
        storage_power_kw = 5
        storage_capacity_kwh = 20

        while True:
            pv_generation = self.simulate_pv_generation(pv_size_kw)
            autocons, storage_profile = self.calculate_autoconsumption(
                pv_generation,
                storage_power_kw,
                storage_capacity_kwh
            )
            if autocons >= min_autoconsumption:
                break
            else:
                pv_size_kw += 2
                storage_power_kw += 1
                storage_capacity_kwh += 5

        return {
            'pv_size_kw': pv_size_kw,
            'storage_power_kw': storage_power_kw,
            'storage_capacity_kwh': storage_capacity_kwh,
            'autoconsumption': autocons,
            'pv_generation': pv_generation,
            'storage_profile': storage_profile
        }

st.title('Kalkulator Dopasowania Magazynu Energii i PV')

uploaded_file = st.file_uploader("Wgraj profil zużycia w formacie CSV (timestamp, consumption_kWh)")

if uploaded_file:
    consumption_profile = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    calculator = EnergySystemCalculator(consumption_profile)

    desired_autoconsumption = st.slider('Minimalny poziom autokonsumpcji (%)', 70, 100, 90) / 100
    optimize_button = st.button('Oblicz optymalne parametry systemu')

    if optimize_button:
        result = calculator.optimize_system(min_autoconsumption=desired_autoconsumption)
        st.success('Optymalna konfiguracja systemu:')
        st.write(f"Moc instalacji PV: {result['pv_size_kw']} kW")
        st.write(f"Moc magazynu energii: {result['storage_power_kw']} kW")
        st.write(f"Pojemność magazynu energii: {result['storage_capacity_kwh']} kWh")
        st.write(f"Poziom autokonsumpcji: {result['autoconsumption']*100:.2f}%")

        pv_gen = result['pv_generation']
        consumption = calculator.consumption_profile['consumption_kWh']

        fig, ax = plt.subplots(figsize=(12,6))
        consumption.resample('D').sum().plot(ax=ax, label='Dzienne zużycie energii (kWh)')
        pv_gen.resample('D').sum().plot(ax=ax, label='Dzienne generowanie PV (kWh)')
        ax.set_title('Porównanie dziennego zużycia energii i produkcji PV')
        ax.set_ylabel('Energia (kWh)')
        ax.legend()
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(12,6))
        result['storage_profile'].resample('D').mean().plot(ax=ax2, label='Średni poziom naładowania magazynu energii')
        ax2.set_title('Poziom naładowania magazynu energii (średnio dziennie)')
        ax2.set_ylabel('Energia (kWh)')
        ax2.legend()
        st.pyplot(fig2)

        monthly_consumption = consumption.resample('M').sum()
        monthly_pv = pv_gen.resample('M').sum()

        fig3, ax3 = plt.subplots(figsize=(12,6))
        monthly_consumption.plot(ax=ax3, label='Miesięczne zużycie energii (kWh)', kind='bar')
        monthly_pv.plot(ax=ax3, label='Miesięczna produkcja PV (kWh)', kind='bar', alpha=0.7)
        ax3.set_title('Miesięczne zużycie energii i produkcja PV')
        ax3.set_ylabel('Energia (kWh)')
        ax3.legend()
        st.pyplot(fig3)

        daily_autoconsumption = (pv_gen.clip(upper=consumption).resample('D').sum()) / consumption.resample('D').sum()
        fig4, ax4 = plt.subplots(figsize=(12,6))
        daily_autoconsumption.plot(ax=ax4)
        ax4.set_title('Dzienne wskaźniki autokonsumpcji')
        ax4.set_ylabel('Autokonsumpcja (%)')
        st.pyplot(fig4)

        hourly_storage = result['storage_profile'].groupby(result['storage_profile'].index.hour).mean()
        fig5, ax5 = plt.subplots(figsize=(12,6))
        hourly_storage.plot(ax=ax5)
        ax5.set_title('Średni poziom naładowania magazynu w ciągu dnia')
        ax5.set_xlabel('Godzina')
        ax5.set_ylabel('Energia (kWh)')
        st.pyplot(fig5)

else:
    st.info('Proszę wgrać plik z profilem zużycia.')