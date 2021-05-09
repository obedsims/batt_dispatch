from __future__ import division
import streamlit as st
import pandas as pd
from plot import plot_dispatch, plot_min_cost_dispatch
from dispatch import dispatch_max_sc, dispatch_max_sc_grid_pf, dispatch_min_costs
from analysis import print_analysis, print_min_cost_analysis
import time

st.set_page_config(layout="wide")

# Title
st.title("Designing a Building Battery Dispatch Strategy ")
st.markdown('A Web App by Obed Sims ([@obedsims](https://www.linkedin.com/in/obedsims/))')
st.markdown("")
st.markdown("Hey there! Welcome to my app. This app lets the user try out various battery"
            " dispatch methods (how the battery is used to meet demand and supply). You can"
            " vary any of the parameters to see how they affect the performance. Beneath the"
            " confusing graph it gives the user details on the performance of each dispatch"
            " method and further insights into the models. **Give it a go!**")
with st.beta_expander("Data Information"):
    st.markdown("The results are based on **15 minute** electricity consumption interval data for a building in "
                "the UK with a 1kW Solar PV system.")
st.markdown("")

# Sidebar Content
dispatch_method = st.sidebar.selectbox("Select Dispatch Method",
                                       ["Maximise Self Consumption", "Peak Shave", "Minimise Costs (LP Optimisation)"])
week = st.sidebar.slider(label="Select the week of data you want to view:",
                         min_value=1, max_value=52, value=20, format='Week %d')
st.sidebar.title("Parameters")
pv_size = st.sidebar.number_input('Enter the Nominal Solar PV Power (kW)', value=10, min_value=0)
batt_capacity = st.sidebar.number_input('Enter the Battery Capacity (kWh)', value=40, min_value=0)
batt_eff = st.sidebar.number_input('Enter the Battery Efficiency', value=0.90)
inv_eff = st.sidebar.number_input('Enter the Inverter Efficiency', value=0.85)
max_power = st.sidebar.number_input('Enter the Maximum Battery Power (kW)', value=20, min_value=0)
#init_charge = st.sidebar.number_input('Enter the Initial Battery Charge', value=20)
cost_to_buy = st.sidebar.number_input('Enter the Price to Buy Electricity (p/kWh)', value=18, min_value=0)
price_to_sell = st.sidebar.number_input('Enter the Price to Sell Electricity (p/kWh)', value=8, min_value=0)

# Data to be used for calculations
demand = pd.read_csv('data/demand.csv', index_col=0, header=None, parse_dates=True, squeeze=True)
pv_1kW = pd.read_csv('data/pv.csv', index_col=0, header=None, parse_dates=True, squeeze=True)

# PV parameters
pv = pv_1kW * pv_size

param_tech = {'BatteryCapacity': batt_capacity,  # kWh
               'BatteryEfficiency': batt_eff,    # dimensionless
               'InverterEfficiency': inv_eff,    # dimensionless
               'timestep': .25,                  # hourly timesteps (15 min intervals currently)
               'MaxPower': max_power,            # kW
               'InitialCharge': 20,              # kWh
               'Cost_to_buy': cost_to_buy,       # p
               'Price_to_sell': price_to_sell,   # p
               }


if dispatch_method == "Maximise Self Consumption":
    E1 = dispatch_max_sc(pv, demand, param_tech, return_series=False)
    plot_dispatch(pv, demand, E1, week=week)
    print_analysis(pv, demand, param_tech, E1)
if dispatch_method == "Peak Shave":
    E1 = dispatch_max_sc_grid_pf(pv, demand, param_tech, return_series=False)
    plot_dispatch(pv, demand, E1, week=week)
    print_analysis(pv, demand, param_tech, E1)
if dispatch_method == "Minimise Costs (LP Optimisation)":
    with st.spinner(text='Optimising...this may take about a minute'):
        time.sleep(65)
    E1 = dispatch_min_costs(pv, demand, param_tech, return_series=False)
    plot_min_cost_dispatch(pv, demand, E1, week=week)
    print_min_cost_analysis(pv, demand, param_tech, E1)

with st.beta_expander("See Battery Dispatch Algorithms Summary"):
    st.write("**Maximise Self Consumption** - The dispatch of the storage capacity is performed in such "
             "a way to maximize self-consumption.")
    st.write("**Peak Shave** - The dispatch of the storage capacity is performed in such a way to maximize "
             "self-consumption and relieve the grid by deferring the storage to peak hours. This would be a much better "
             "strategy under a variable electricity price.")
    st.write("**Minimise Costs** - The dispatch of the storage capacity is optimised in such a way to minimize "
             "the costs through maximising the profit generated selling electricity and minimize "
             "costs of electricity purchase from the grid.")

st.markdown('')

with st.beta_expander("Tell Me More (LP Model)"):
    st.title('Building the LP Model')

    st.markdown("""
    Factors to consider in the Linear Programming (LP) Model:
    - Which factors which will influence the optimal solution. These included the **network tariffs**, the **building consumption
     profile** and the **PV generation profile**.
    - How to obtain a global optimal solution quickly as an entire year worth of 15 minute interval data is being considered.
    - In order to properly assess the financial impact of different batteries, we need to decide on how the battery will
      be controlled. The controller could be a simple reactive one or we could do more sophisticated model predictive control.
      To keep things simple, it is assumed that the battery controller has access to perfect knowledge about what will 
      happen in the house.
    - We must initially define the variables that we are going to optimise. In this model they are the **battery state (kWh)**, 
    **battery charge (kW)** and **battery discharge (kW)**, the **total house power (kW)** and **power cost (pence)**.
    """)

    st.latex(
        """
        levelofcharge_t, \:charge_t, \:discharge_t, \:totalpower_t, \:costpower_t
        """
    )
    st.markdown(
        r"""
        The following equality and inequality constraints have to be met for each **timestep $i$**:
        """)

    st.latex(
        r"""\begin{aligned}
        totalpower[i] &= charge[i] - discharge[i] + load[i] - solarpv[i] \\
        costpower[i] &\geq timestep \cdot costbuy \cdot totalpower[i] \\
        costpower[i] &\geq timestep \cdot pricesell \cdot totalpower[i] \\
        levelofcharge[i] &\leq max\:battery\:capacity \\
        charge[i] &\leq max\:battery\:power \\
        discharge[i] &\leq max\:battery\:power \\
        \end{aligned}
        """)

    st.markdown(
        r"""
        The following expression shows the objective function which aims to minimise 
        the total cost of power over the year.
        """)

    st.latex(r"""
    min\:\sum_{t}^{Nsteps} costpower_t
    """)

    st.markdown(r"""
    where **$Nsteps$** is the total number of time steps over the year of data.

    Battery charge state constraints are as follows:
    """)

    st.latex(
        r"""\begin{aligned}
            levelofcharge[i] &= levelofcharge[i - 1] + timestep \cdot (eff \cdot charge[i] - discharge[i]) \\
            levelofcharge[0] &= initialcharge \\
            levelofcharge[Nsteps - 1] &= initialcharge
            \end{aligned}
        """
    )
    st.markdown(r"""
    where **$eff$** is the combined efficiency of the inverter and battery. The first expression states
    that the level of the battery at time $i$ has to equal the level of charge from the previous timestep
    plus the change in $charge$ and $discharge$ at the current timestep. The level of charge at the start of the 
    period is equal to $initialcharge$ and this is equal to the state of the battery at the end of the time period.
    """)




