from __future__ import division
import numpy as np
import pandas as pd


def dispatch_max_sc(pv, demand, param, return_series=False):
    """ Self consumption maximization pv + battery dispatch algorithm.
    The dispatch of the storage capacity is performed in such a way to maximize self-consumption:
    the battery is charged when the PV power is higher than the load and as long as it is not fully charged.
    It is discharged as soon as the PV power is lower than the load and as long as it is not fully discharged.
    Arguments:
        pv (pd.Series): Vector of PV generation, in kW DC (i.e. before the inverter)
        demand (pd.Series): Vector of household consumption, kW
        param (dict): Dictionary with the simulation parameters:
                timestep (float): Simulation time step (in hours)
                BatteryCapacity: Available battery capacity (i.e. only the the available DOD), kWh
                BatteryEfficiency: Battery round-trip efficiency, -
                InverterEfficiency: Inverter efficiency, -
                MaxPower: Maximum battery charging or discharging powers (assumed to be equal), kW
        return_series(bool): if True then the return will be a dictionary of series. Otherwise it will be a dictionary of ndarrays.
                        It is recommended to return ndarrays if speed is an issue (e.g. for batch runs).
    Returns:
        dict: Dictionary of Time series
    """

    bat_size_e_adj = param['BatteryCapacity']
    bat_size_p_adj = param['MaxPower']
    n_bat = param['BatteryEfficiency']
    n_inv = param['InverterEfficiency']
    timestep = param['timestep']
    init_charge = param['InitialCharge']
    # We work with np.ndarrays as they are much faster than pd.Series
    Nsteps = len(pv)
    LevelOfCharge = np.zeros(Nsteps)
    pv2store = np.zeros(Nsteps)
    #inv2grid = np.zeros(Nsteps)
    store2inv = np.zeros(Nsteps)
    grid2store = np.zeros(Nsteps)   # TODO Always zero for now.

    # Load served by PV - e.g. if the pv gen < demand,
    # then take the value of the pv gen as it's the amount that can be supplied to the inverter by the PV.
    # If the PV gen > demand then take the value of the demand
    pv2inv = np.minimum(pv, demand / n_inv)  # DC direct self-consumption

    # Residual load - This is the remaining energy available to supply the battery or return to the grid
    res_load = (demand - pv2inv * n_inv)  # AC
    inv2load = pv2inv * n_inv  # AC

    # Excess PV - If (pv gen - demand) > 0 then take the residual, otherwise take 0 as the excess
    res_pv = np.maximum(pv - demand/n_inv, 0)  # DC

    # PV to storage after eff losses
    pv2inv = pv2inv.values

    # first timestep = the initial charge (can be modified by user)
    LevelOfCharge[0] = init_charge  # bat_size_e_adj / 2  # DC

    for i in range(1, Nsteps):
        # PV to storage
        if LevelOfCharge[i-1] >= bat_size_e_adj:  # if battery is full
                pv2store[i] = 0
        else: # if battery is not full
            if LevelOfCharge[i-1] + res_pv[i] * n_bat * timestep > bat_size_e_adj:  # if battery will be full after putting excess
                pv2store[i] = min((bat_size_e_adj - LevelOfCharge[i-1]) / timestep, bat_size_p_adj)
            else:
                pv2store[i] = min(res_pv[i] * n_bat, bat_size_p_adj)

        #Storage to load
        store2inv[i] = min(bat_size_p_adj,  # DC
                           res_load[i] / n_inv,
                           LevelOfCharge[i-1] / timestep)

        #SOC
        LevelOfCharge[i] = min(LevelOfCharge[i-1] - (store2inv[i] - pv2store[i] - grid2store[i]) * timestep,  # DC
                               bat_size_e_adj)

    pv2inv = pv2inv + res_pv - pv2store
    inv2load = inv2load + store2inv * n_inv  # AC
    inv2grid = (res_pv - pv2store) * n_inv  # AC
    grid2load = demand - inv2load  # AC

    #MaxDischarge = np.minimum(LevelOfCharge[i-1]*BatteryEfficiency/timestep,MaxPower)


    #Potential Grid to storage  # TODO: not an option for now in this strategy

    out = {'pv2inv': pv2inv,
            'res_pv': res_pv,
            'pv2store': pv2store,
            'inv2load': inv2load,
            'grid2load': grid2load,
            'store2inv': store2inv,
            'LevelOfCharge': LevelOfCharge,
            'inv2grid': inv2grid
            # 'grid2store': grid2store
            }
    if not return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input pv
            out_pd[k] = pd.Series(v, index=pv.index)
        out = out_pd
    return out


def dispatch_max_sc_grid_pf(pv, demand, param_tech, return_series=False):
    """
    Battery dispatch algorithm.
    The dispatch of the storage capacity is performed in such a way to maximize self-consumption and relieve the grid by
    by deferring the storage to peak hours.
    the battery is charged when the PV power is higher than the load and as long as it is not fully charged.
    It is discharged as soon as the PV power is lower than the load and as long as it is not fully discharged.
    :param return_series:
    :param pv: Vector of PV generation, in kW DC (i.e. before the inverter)
    :param demand: Vector of household consumption, kW
    :param param_tech: Dictionary with the simulation parameters:
                    timestep: Simulation time step (in hours)
                    BatteryCapacity: Available battery capacity (i.e. only the the available DOD), kWh
                    BatteryEfficiency: Battery round-trip efficiency, -
                    InverterEfficiency: Inverter efficiency, -
                    MaxPower: Maximum battery charging or discharging powers (assumed to be equal), kW
    :return: Dictionary of Time series
    """
    bat_size_e_adj = param_tech['BatteryCapacity']
    bat_size_p_adj = param_tech['MaxPower']
    n_bat = param_tech['BatteryEfficiency']
    n_inv = param_tech['InverterEfficiency']
    timestep = param_tech['timestep']

    Nsteps = len(pv)
    LevelOfCharge = np.zeros(Nsteps)
    pv2store = np.zeros(Nsteps)
    #inv2load = np.zeros(Nsteps)
    inv2grid = np.zeros(Nsteps)
    store2inv = np.zeros(Nsteps)
    grid2store = np.zeros(Nsteps) # TODO Always zero for now.

    from scipy.optimize import brentq

    def find_threshold(pv_day_load, bat_size_e):
        """Find threshold of peak shaving (kW). The electricity fed to the grid is capped by a specific threshold.
        What is above that threshold is stored in the battery. The threshold is specified in such a way so that
        the energy amount above that threshold equals to the available storage for that day.
        pv_day_load: Daily pv production
        bat_size_e: Battery size
        """

        def get_residual_peak(thres):
            shaved_peak = np.maximum(pv_day_load - thres, 0)
            return sum(shaved_peak) * param_tech['timestep'] - bat_size_e

        if sum(pv_day_load) * param_tech['timestep'] <= bat_size_e:  # if the battery can cover the whole day
            return 0
        else:
            return brentq(get_residual_peak, 0, max(pv), rtol=1e-4)

    # It is better to use vectorize operations as much as we can before looping.
    # first self consume
    # Load served by PV
    pv2inv = np.minimum(pv, demand / n_inv)  # DC direct self-consumption

    # Residual load
    res_load = (demand - pv2inv * n_inv)  # AC
    inv2load = pv2inv * n_inv  # AC
    pv2inv = pv2inv.values

    # Excess PVs
    res_pv = np.maximum(pv - demand / n_inv, 0)  # DC
    res_pv_val = res_pv.values
    Nsteps = len(demand)
    LevelOfCharge[0] = 0  # bat_size_e_adj / 2 # Initial storage is empty # DC

    # For the residual pv find the threshold above which the energy should be stored (first day)
    threshold = find_threshold(res_pv_val[0: 0 + int(23 / timestep)], bat_size_e_adj - LevelOfCharge[0])

    for i in range(1, Nsteps):  # Loop hours
        # Every 24 hours find the threshold for the next day (assuming next 24 hours)
        if i % int(24/timestep) == 0:
            threshold = find_threshold(res_pv_val[i:i + int(23 / timestep)],
                                       bat_size_e_adj - LevelOfCharge[i])

        # PV to grid
        if res_pv[i] * n_inv < threshold:  # If residual load is below threshold
            inv2grid[i] = res_pv[i] * n_inv  # Sell to grid what is not consumed
        else:  # If load is above threshold
            inv2grid[i] = threshold * n_inv  # Sell to grid what is below the threshold
            pv2store[i] = min(max(0, (res_pv[i] - threshold) * n_bat / n_inv),
                              (bat_size_e_adj - LevelOfCharge[i - 1]) / timestep )  # Store what is above the threshold and fits in battery
        pv2inv[i] = pv2inv[i] + inv2grid[i] / n_inv  # DC

        store2inv[i] = min(bat_size_p_adj,  # DC
                           res_load[i] / n_inv,
                           LevelOfCharge[i - 1] / timestep)

        LevelOfCharge[i] = min(LevelOfCharge[i - 1] - (store2inv[i] - pv2store[i] - grid2store[i]) * timestep,
                               bat_size_e_adj ) # DC

    inv2load = inv2load + store2inv * n_inv  # AC
    grid2load = demand - inv2load  # AC

    out = {'pv2inv': pv2inv,
            'res_pv': res_pv,
            'pv2store': pv2store,
            'inv2load': inv2load,
            'grid2load': grid2load,
            'store2inv': store2inv,
            'LevelOfCharge': LevelOfCharge,
            'inv2grid': inv2grid
            # 'grid2store': grid2store
            }
    if not return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input pv
            out_pd[k] = pd.Series(v, index=pv.index)
        out = out_pd
    return out


def dispatch_min_costs(pv, demand, param, return_series=False):
    """
    Battery dispatch algorithm.
    The dispatch of the storage capacity is optimised in such a way to minimize the costs through maximising the profit
    generated selling electricity and minimize costs of electricity purchase from the grid.
    :param return_series:
    :param pv: Vector of PV generation, in kW DC (i.e. before the inverter)
    :param demand: Vector of household consumption, kW
    :param param: Dictionary with the simulation parameters:
                    timestep: Simulation time step (in hours)
                    BatteryCapacity: Available battery capacity (i.e. only the the available DOD), kWh
                    BatteryEfficiency: Battery round-trip efficiency, -
                    InverterEfficiency: Inverter efficiency, -
                    MaxPower: Maximum battery charging or discharging powers (assumed to be equal), kW
                    Cost_to_buy: Cost to buy electricity from the grid (p)
                    Price_to_sell: Price to sell electricity back to the grid (p)
    :return: Dictionary of Time series
    """
    # Parameters
    bat_size_e_adj = param['BatteryCapacity']   # battery capacity (kWh)
    bat_size_p_adj = param['MaxPower']      # Maximum battery power (kW)
    n_bat = param['BatteryEfficiency']      # battery efficiency
    n_inv = param['InverterEfficiency']     # inverter efficiency
    timestep = param['timestep']
    init_charge = param['InitialCharge']
    cost_buy = param['Cost_to_buy']         # buy price (p/kWh)
    price_sell = param['Price_to_sell']     # sell price (p/kWh) formulation works only when sell <= buy

    Nsteps = len(pv)  # number of time steps
    assert (len(demand) == len(pv))

    from pulp import LpVariable, LpProblem, LpStatus, lpSum, value

    # Power profiles
    pload = demand.to_numpy()   # household background load (kW)
    ppv = pv.to_numpy()         # PV generation (kW)
    res_pv = np.maximum(pv - demand / n_inv, 0)  # DC

    # Battery parameters
    eff = n_bat * n_inv  # combined battery and inverter efficiency

    # Battery variables
    LevelofCharge = [LpVariable('E_{}'.format(i), 0, None) for i in range(Nsteps)]  # battery energy (kWh)
    charge = [LpVariable('pc_{}'.format(i), 0, None) for i in range(Nsteps)]  # battery charge (kW)
    discharge = [LpVariable('pd_{}'.format(i), 0, None) for i in range(Nsteps)]  # battery discharge (kW)

    # Auxiliary variables
    total_power = [LpVariable('p_{}'.format(i), None, None) for i in range(Nsteps)]  # total house power (kW)
    cost_power = [LpVariable('cpow_{}'.format(i), None, None) for i in range(Nsteps)]  # power cost (p)

    # Optimisation problem
    prb = LpProblem('Battery Operation')

    # Objective
    prb += lpSum(cost_power)   # sum of electricity costs - sum of revenue gen
    # Constraints
    for i in range(Nsteps):
        prb += total_power[i] == charge[i] - discharge[i] + pload[i] - ppv[i]  # total power (kW)
        prb += cost_power[i] >= timestep * cost_buy * total_power[i]           # power cost constraint
        prb += cost_power[i] >= timestep * price_sell * total_power[i]         # power cost constraint
        prb += LevelofCharge[i] <= bat_size_e_adj                              # battery capacity
        prb += charge[i] <= bat_size_p_adj
        prb += discharge[i] <= bat_size_p_adj

    # Battery charge state constraints
    # Batteries must finish half charged
    prb += LevelofCharge[0] == init_charge  # starting energy
    prb += LevelofCharge[Nsteps - 1] == init_charge  # finishing energy
    for i in range(1, Nsteps):
        prb += LevelofCharge[i] == LevelofCharge[i - 1] + timestep * (eff * charge[i] - discharge[i])  # battery transitions

    # Solve problem
    prb.solve()

    optimal_sln = value(prb.objective)
    print('Status {}'.format(LpStatus[prb.status]))
    print('Cost {}'.format(optimal_sln))

    charge_plot = []
    discharge_plot = []
    total_power_plot = []
    LevelofCharge_plot = []
    cost_plot = []
    for i in charge:
        charge_plot.append(value(i))
    for i in discharge:
        discharge_plot.append(value(i))
    for i in total_power:
        total_power_plot.append(value(i))
    for i in LevelofCharge:
        LevelofCharge_plot.append(value(i))
    for i in cost_power:
        cost_plot.append(value(i))

    out = {'charge': charge_plot,
           'discharge': discharge_plot,
           'total_power': total_power_plot,
           'LevelofCharge': LevelofCharge_plot,
           'cost': cost_plot
           # 'grid2store': grid2store
           }
    if not return_series:
        out_pd = {}
        for k, v in out.items():  # Create dictionary of pandas series with same index as the input pv
            out_pd[k] = pd.Series(v, index=pv.index)
        out = out_pd
    return out

