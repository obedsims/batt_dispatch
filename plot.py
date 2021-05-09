"""Plotting functions"""
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_dispatch(pv, demand, E, week=30):
    """ Visualize dispatch algorithm for a specific week
    Parameters:
        demand (pd.Series): demand production
        pv (pd.Series): pv production
        E (dict):  Energy flows. Dictionary of pd.Series: res_pv, grid2load, store2inv, LevelOfCharge
    """

    sliced_index = (pv.index.isocalendar().week==week)
    pv_sliced = pv[sliced_index]
    demand_sliced = demand[sliced_index]
    self_consumption = E['inv2load'][sliced_index]
    res_pv_sliced = E['res_pv'][sliced_index]
    grid2load_sliced = E['grid2load'][sliced_index]
    store2inv_sliced = E['store2inv'][sliced_index]
    LevelOfCharge = E['LevelOfCharge'][sliced_index]
    inv2grid = E['inv2grid'][sliced_index]
    grid2load = E['grid2load'][sliced_index]

    f, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(17, 4*3), frameon=False,
                             gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.04})

    ax[0].fill_between(demand_sliced.index, 0, demand_sliced, color='orange',
                         alpha=.2, label='Building Demand (kW)')
    # axes[0].plot(self_consumption.index, self_consumption,
    #                      color='dodgerblue', alpha=.4, label='Self Consumption (kW)')
    ax[0].fill_between(pv_sliced.index, 0, pv_sliced,
                         color='grey', alpha=.2, label='Solar Generation (kW)')
    #axes[0].fill_between(res_pv_sliced.index, self_consumption, pv_sliced,
    #                    color='yellow', alpha=.2)
    ax[0].fill_between(store2inv_sliced.index, 0, store2inv_sliced,
                         color='purple', alpha=.3, label='Battery to Usage (kW)')
    #axes[0].fill_between(grid2load_sliced.index, pv_sliced + store2inv_sliced,
    #                     grid2load_sliced + pv_sliced + store2inv_sliced, color='green', alpha=.4,
    #                     label='Grid to Load (kW)')
    ax[0].plot(grid2load_sliced.index, grid2load_sliced, color='indianred', lw=1, label='Imported Electricity (kW)')
    ax[0].set_xlim(demand_sliced.index[0], demand_sliced.index[-1])
    ax[0].set_ylim([0, ax[0].get_ylim()[1]])
    ax[0].set_ylabel('Power (kW)')

    ax[1].fill_between(LevelOfCharge.index, 0, LevelOfCharge, color='dodgerblue', alpha=.6)
    ax[1].set_ylabel('State of Charge (kWh)')

    ax[2].fill_between(inv2grid.index, 0, inv2grid, color='green', alpha=.2)
    ax[2].fill_between(inv2grid.index, 0, -grid2load, color='red', alpha=.2)
    ax[2].set_ylabel('In/out from grid (kWh)')
    ax[0].legend(loc='best')
    st.pyplot(fig=f)
    return


def plot_min_cost_dispatch(pv, demand, E, week=30):
    """ Visualize the minimise costs dispatch algorithm for a specific week
    Parameters:
        demand (pd.Series): demand production
        pv (pd.Series): pv production
        E (dict):  Energy flows. Dictionary of pd.Series: res_pv, grid2load, store2inv, LevelOfCharge
    """

    sliced_index = (pv.index.isocalendar().week == week)
    pv_sliced = pv[sliced_index]
    demand_sliced = demand[sliced_index]
    grid_imp_ex = E['total_power'][sliced_index]
    battery_level = E['LevelofCharge'][sliced_index]
    charge = E['charge'][sliced_index]
    discharge = E['discharge'][sliced_index]

    to_grid = []
    from_grid = []
    for i in grid_imp_ex:
        if i < 0:
            to_grid.append(i*-1)
        else:
            to_grid.append(0)
    for i in grid_imp_ex:
        if i >= 0:
            from_grid.append(i*-1)
        else:
            from_grid.append(0)

    f, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(17, 4 * 3), frameon=False,
                         gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.04})

    ax[0].fill_between(demand_sliced.index, demand_sliced,
                 color='orange', alpha=.2, label='Building Demand (kW)')
    ax[0].fill_between(pv_sliced.index, 0, pv_sliced,
                 color='grey', alpha=.2, label='Solar Generation (kW)')
    sns.lineplot(charge.index, charge, ax=ax[0], color='green',
                 label='Charge (kW)')
    sns.lineplot(discharge.index, discharge, ax=ax[0], color='indianred',
                 label='Discharge (kW)')
    sns.lineplot(grid_imp_ex.index, grid_imp_ex, ax=ax[0],
                 color='black', label='Total Power Import/Export to Grid (kW)')
    ax[0].set_ylabel('Power (kW)')
    ax[0].set_xlim(charge.index[0], charge.index[-1])
    ax[1].fill_between(battery_level.index, 0, battery_level,
                 color='dodgerblue', alpha=.6)
    ax[1].set_ylabel('State of Charge (kWh)')

    ax[2].fill_between(grid_imp_ex.index, 0, to_grid, color='green', alpha=.2)
    ax[2].fill_between(grid_imp_ex.index, 0, from_grid, color='red', alpha=.2)
    ax[2].set_ylabel('In/out from grid (kW)')
    plt.xlabel('Time (hr)')
    plt.legend(loc='best')
    st.pyplot(fig=f)
