import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
################################################
#set the default font for all figures
font_families = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
#plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.family'] = 'helvet'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.usetex'] = True
# Global text size for figures
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
baseline_linewidth = 1.7
###################################
os.chdir('/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/scenario_plots/baseline_scenario/')  
file = 'Scenario_2_visible_plot_data_20260512_120059.csv'
df = pd.read_csv(file)
df = df[df['time'] <= 50]  # Filter data to include only time <= 50 years
################################################
def soil_plots(data_1, data_2, data_3, time_col, save_as=None):
    plt.figure(figsize=(2.8, 2.5), dpi=150)
    colors = ['lightcoral', 'red', 'darkred']
    for data, color in zip([data_1, data_2, data_3], colors):
        plt.plot(time_col, data, label=f'{data.columns[0]}', linewidth=baseline_linewidth, color=color)

    plt.text(0.02, 0.93, '(a)', transform=plt.gca().transAxes, fontsize=10)
    plt.ylabel(f'Soil Quality Index')
    plt.xlabel(f'Time [yr]')
    plt.xlim(0, time_col.max())
    plt.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()

def auto_ylim_with_padding(ax, y, frac=0.08, min_pad=1e-6):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    y_min, y_max = y.min(), y.max()
    span = y_max - y_min
    pad = span * frac if span > 0 else max(abs(y_min) * frac, min_pad)
    ax.set_ylim(y_min - pad, y_max + pad)

def regular_plot(data, time_col, ylabel, plot_color, panel_name, panel_xc=0.02, panel_yc=0.93, figsize=(2.8, 2.0), rescale=False, save_as=None):
    plt.figure(figsize=(figsize), dpi=150)
    if rescale:
        ydata=data / rescale
    else:        
        ydata=data
    
    plt.plot(time_col, ydata, linewidth=baseline_linewidth, color=plot_color)
    plt.text(panel_xc, panel_yc, panel_name, transform=plt.gca().transAxes, fontsize=10)
    plt.ylabel(ylabel)
    plt.xlabel(f'Time [yr]')
    plt.xlim(0, time_col.max())
    auto_ylim_with_padding(plt.gca(), ydata, frac=0.11)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()

def combined_plot(time_col, data_1, data_2, ylabel_1, ylabel_2, panel_name, panel_xc=0.02, figsize=(2.8, 2.0), xlim=None, color_1='blue', color_2='green', num_bins=6, save_as=None):
    # plot two curves on the same plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(figsize), dpi=150)
    ax2 = ax1.twinx()
    ax1.plot(time_col, data_1, linewidth=baseline_linewidth, color=color_1, linestyle='--', dashes=(5, 2))
    ax2.plot(time_col, data_2, linewidth=baseline_linewidth, color=color_2, linestyle='--', dashes=(7, 4))
    ax1.set_xlabel(f'Time [yr]')
    ax1.set_ylabel(ylabel_1, color=color_1)
    ax2.set_ylabel(ylabel_2, color=color_2)
    ax1.tick_params(axis='y', labelcolor=color_1)
    ax2.tick_params(axis='y', labelcolor=color_2)
    ax1.locator_params(axis='y', nbins=num_bins)  
    ax2.locator_params(axis='y', nbins=num_bins) 
    plt.text(panel_xc, 0.9, panel_name, transform=plt.gca().transAxes, fontsize=10)
    plt.xlim(0, xlim)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()

# A function to stackplot land production for the three parcels
def stacked_land_production_plot(data_1, data_2, data_3, time_col, save_as=None):
    plt.figure(figsize=(2.8, 2.5), dpi=150)
    plt.stackplot(time_col, data_1, data_2, data_3, labels=[f'Parcel 1', f'Parcel 2', f'Parcel 3'], colors=['lightcoral', 'red', 'darkred'])
    plt.text(0.02, 0.93, '(b)', transform=plt.gca().transAxes, fontsize=10)
    plt.ylabel(f'Food Production [Mt/yr]')
    plt.xlabel(f'Time [yr]')
    plt.xlim(0, time_col.max())
    plt.legend(loc='upper left')
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()


################################################
# Extract time column
time_col = df['time']
# Extract columns for soil index for each land
soil_land_1 = df['soil_Land_1']
soil_land_2 = df['soil_Land_2']
soil_land_3 = df['soil_Land_3']
# Change series names to Parcel 1, Parcel 2, and Parcel 3
soil_land_1.columns = ['Parcel 1']
soil_land_2.columns = ['Parcel 2']
soil_land_3.columns = ['Parcel 3']
# Land production data
land_prod_land_1 = df['prod_Land_1']
land_prod_land_2 = df['prod_Land_2']
land_prod_land_3 = df['prod_Land_3']
total_land_prod = df['prod_Scenario_total']
# Emissions
emissions = df['emis_Scenario_total']
# SSR data
self_sufficiency_ratio = df['ssr_Scenario']
# Food price index
food_price_index = df['food_price_Scenario']
# Palma ratio and Gini index
palma_ratio = df['palma_Scenario']
gini_index = df['gini_Scenario']
# Food security index
food_security_index = df['food_sec_Scenario']
# Food demand
food_demand = df['cal_demand_Scenario']
# Real income
real_income = df['inc_Scenario']
# Convert to thousands of pounds per year
real_income = real_income / 1e3
# Redistribution factor
redistribution_factor = df['lambda_t']
#If the value barely changes over time, set it to the initial value
if (redistribution_factor.iloc[0] - redistribution_factor.iloc[-1] < 1e-6):
    redistribution_factor.iloc[:] = redistribution_factor.iloc[0]
    #redistribution_factor.iloc[:] = redistribution_factor.iloc[0]

################################################
# Plots
soil_plots(soil_land_1, soil_land_2, soil_land_3, time_col, save_as='baseline_scenario_soil_plot.eps')
regular_plot(total_land_prod, time_col, ylabel='Food Production [Mt/yr]', panel_name = '(b)', plot_color='blue', rescale=1e6, figsize=(2.8, 2.5), save_as='baseline_scenario_production_plot.eps')
regular_plot(emissions, time_col, ylabel='Net Emissions [MtCO$_2$eq/yr]', rescale=1e6, panel_name = '(c)', plot_color="#D11F00", figsize=(2.8, 2.5), save_as='baseline_scenario_emissions_plot.eps')
regular_plot(food_demand, time_col, ylabel='Food consumption [Mt/yr]', panel_name = '(d)', panel_yc = 0.9, plot_color='cyan', rescale=1e6, save_as='baseline_scenario_food_demand_plot.eps')
regular_plot(self_sufficiency_ratio, time_col, ylabel='Self-sufficiency Ratio [$\%$]', panel_name = '(e)', panel_yc=0.90, plot_color='green', save_as='baseline_scenario_ssr_plot.eps')
regular_plot(food_price_index, time_col, ylabel='Food Price Index [£k/yr]', panel_name = '(f)', panel_yc=0.90, plot_color='orange', rescale=1e3, save_as='baseline_scenario_food_price_plot.eps')

combined_plot(time_col, gini_index, palma_ratio, num_bins=5, ylabel_1='Palma Ratio', ylabel_2='Gini Index',panel_name = '(h)', panel_xc= 0.85 , xlim=time_col.max(), color_1='purple', color_2='salmon', save_as='baseline_scenario_palma_gini_plot.eps')
combined_plot(time_col, real_income, redistribution_factor, num_bins=5, ylabel_1='Average Income [£k/yr]', ylabel_2='Redistribution Factor [$10^{-5}$]', panel_name = '(g)', xlim=time_col.max(), color_1='magenta', color_2='teal', save_as='baseline_scenario_income_lambda_plot.eps')
regular_plot(food_security_index, time_col, ylabel='Food Insecurity Index', panel_yc=0.90, panel_name = '(i)', plot_color='brown', save_as='baseline_scenario_food_security_plot.eps')

stacked_land_production_plot(land_prod_land_1, land_prod_land_2, land_prod_land_3, time_col, save_as='baseline_scenario_stacked_production_plot.eps')