"""This module contains functions to analyze the results of the dispatch algorithm"""

import numpy as np
import streamlit as st

def print_analysis(pv, demand, param, E):
    """ Print statistics and information of the dispatched solution
    Arguments
        pv (pd.Series): PV timeseries
        demand (pd.Series): demand timeseries
        param (dict): dictionary of technical parameters
        E (dict): dictionary of energy flows as estimated by the algorithm
    Returns
        none
    """
    timestep = param['timestep']
    buy_cost = param['Cost_to_buy']
    sell_price = param['Price_to_sell']
    SelfConsumption = np.sum(E['inv2load']) * timestep
    TotalFromGrid = np.sum(E['grid2load']) * timestep
    TotalToGrid = np.sum(E['inv2grid']) * timestep
    TotalLoad = demand.sum() * timestep
    TotalPV = pv.sum() * timestep
    TotalBatteryGeneration = np.sum(E['store2inv']) * timestep
    TotalBatteryConsumption = np.sum(E['pv2store']) * timestep
    BatteryLosses = TotalBatteryConsumption - TotalBatteryGeneration
    InverterLosses = (TotalPV - BatteryLosses) * (1 - param['InverterEfficiency'])
    SelfConsumptionRate = SelfConsumption / TotalPV * 100             # in %
    SelfSufficiencyRate = SelfConsumption / TotalLoad * 100
    AverageDepth = TotalBatteryGeneration / (365 * param['BatteryCapacity'])
    Nfullcycles = 365 * AverageDepth
    CostWithoutBattPV = buy_cost*TotalLoad/100
    residue = TotalPV + TotalFromGrid - TotalToGrid - BatteryLosses - InverterLosses - TotalLoad
    cost = ((TotalFromGrid * buy_cost) - (TotalToGrid * sell_price))/100
    CostSavings = CostWithoutBattPV - cost

    # Present results
    st.success("**Dispatch Results** :zap:")
    col1, col2 = st.beta_columns(2)

    col1.write('**Total yearly consumption:** {:,.0f} kWh'.format(TotalLoad))
    col1.write('**Total PV production:** {:,.0f} kWh'.format(TotalPV))
    col1.write('**Total fed to the grid:** {:,.0f} kWh'.format(TotalToGrid))
    col1.write('**Total bought from the grid:** {:,.0f} kWh'.format(TotalFromGrid))
    col1.write('**Total yearly electricity cost:** £ {:,.2f}'.format(cost))
    col2.write('**Total energy provided by the battery:** {:,.0f} kWh'.format(TotalBatteryGeneration))
    col2.write('**Average Charging/Discharging depth:** {:.3f}'.format(AverageDepth))
    col2.write('**Number of equivalent full cycles per year:** {:,.0f} '.format(Nfullcycles))
    col2.write('**Annual cost savings compared to a building without '
               'Solar PV or a Battery:** £ {:,.0f} '.format(CostSavings))


def print_min_cost_analysis(pv, demand, param, E):
    """ Print statistics and information of the dispatched solution
    Arguments
        pv (pd.Series): PV timeseries
        demand (pd.Series): demand timeseries
        param (dict): dictionary of technical parameters
        E (dict): dictionary of energy flows as estimated by the algorithm
    Returns
        none
    """
    param_tech2 = {'BatteryCapacity': 40,  # kWh
                   'BatteryEfficiency': .9,  # dimensionless
                   'InverterEfficiency': .85,  # dimensionless
                   'timestep': .25,  # hourly timesteps (15 min intervals currently)
                   'MaxPower': 20,  # kW
                   'InitialCharge': 20,  # kWh
                   'Cost_to_buy': 18,  # p
                   'Price_to_sell': 8,  # p
                   }

    timestep = param['timestep']
    buy_cost = param['Cost_to_buy']
    sell_price = param['Price_to_sell']
    cost = np.sum(E['cost'])/100
    TotalLoad = demand.sum() * timestep
    TotalPV = pv.sum() * timestep
    #TotalFromGrid = np.sum(E['total_power'] > 0) * timestep
    TotalBatteryGeneration = np.sum(E['charge']) * timestep
    TotalBatteryConsumption = np.sum(E['discharge']) * timestep
    AverageDepth = TotalBatteryGeneration / (365 * param['BatteryCapacity'])
    Nfullcycles = 365 * AverageDepth
    CostWithoutBattPV = buy_cost*TotalLoad/100
    CostSavings = CostWithoutBattPV - cost
    TotalToGrid = TotalPV - TotalLoad - TotalBatteryGeneration + TotalBatteryConsumption

    # Present results
    st.success("**Dispatch Method Results** :zap:")
    col1, col2 = st.beta_columns(2)

    col1.write('**Total yearly consumption:** {:,.0f} kWh'.format(TotalLoad))
    col1.write('**Total PV production:** {:,.0f} kWh'.format(TotalPV))
    col1.write('**Total fed to the grid:** {:,.0f} kWh'.format(TotalToGrid))
    col1.write('**Total yearly electricity cost:** £ {:,.2f}'.format(cost))
    col2.write('**Total Energy provided by the battery:** {:,.0f} kWh'.format(TotalBatteryGeneration))
    col2.write('**Average Charging/Discharging depth:** {:.03f}'.format(AverageDepth))
    col2.write('**Number of equivalent full cycles per year:** {:,.0f}'.format(Nfullcycles))
    col2.write('**Annual cost savings compared to a building without '
               'Solar PV or a Battery:** £ {:,.0f} '.format(CostSavings))
