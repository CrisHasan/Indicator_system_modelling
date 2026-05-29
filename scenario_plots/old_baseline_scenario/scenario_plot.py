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
os.chdir('/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/scenario_plots/old_baseline_scenario/')  
file = 'Scenario_1_visible_plot_data_20260410_113327.csv'
df = pd.read_csv(file)  
################################################
def soil_plots(data_1, data_2, data_3, save_as=None):
    plt.figure(figsize=(2.8, 2.5), dpi=150)
    colors = ['lightcoral', 'red', 'darkred']
    for data, color in zip([data_1, data_2, data_3], colors):
        plt.plot(data['time'], data['value'], label=f'{data["series_name"].iloc[0]}', linewidth=baseline_linewidth, color=color)

    plt.text(0.02, 0.9, '(a)', transform=plt.gca().transAxes, fontsize=10)
    plt.ylabel(f'Soil Quality Index')
    plt.xlabel(f'Time [yr]')
    plt.xlim(0, data_1['time'].max())
    plt.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()

def regular_plot(data, ylabel, xlim, plot_color, panel_name, panel_xc=0.02, figsize=(2.8, 2.0), rescale=False, save_as=None):
    plt.figure(figsize=(figsize), dpi=150)
    if rescale:
        ydata=data['value'] / rescale
    else:        
        ydata=data['value']
    
    plt.plot(data['time'], ydata, label=f'{data["series_name"].iloc[0]}', linewidth=baseline_linewidth, color=plot_color)
    plt.text(panel_xc, 0.9, panel_name, transform=plt.gca().transAxes, fontsize=10)
    plt.ylabel(ylabel)
    plt.xlabel(f'Time [yr]')
    plt.xlim(0, xlim)
    plt.xlim(0, data['time'].max())
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()

def combined_plot(data_1, data_2, ylabel_1, ylabel_2, panel_name, panel_xc=0.02, figsize=(2.8, 2.0), xlim=None, color_1='blue', color_2='green', num_bins=6, save_as=None):
    # plot two curves on the same plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(figsize), dpi=150)
    ax2 = ax1.twinx()
    ax1.plot(data_1['time'], data_1['value'], label=f'{data_1["series_name"].iloc[0]}', linewidth=baseline_linewidth, color=color_1, linestyle='--', dashes=(5, 2))
    ax2.plot(data_2['time'], data_2['value'], label=f'{data_2["series_name"].iloc[0]}', linewidth=baseline_linewidth, color=color_2, linestyle='--', dashes=(7, 4))
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

################################################
# Extract columns for soil index for each land
land_1 = df[(df['series_name'] == 'Land 1') & (df['plot_key'] == 'soil')]
land_2 = df[(df['series_name'] == 'Land 2') & (df['plot_key'] == 'soil')]
land_3 = df[(df['series_name'] == 'Land 3') & (df['plot_key'] == 'soil')]
# change serie names to Parcel 1, Parcel 2, and Parcel 3
land_1['series_name'] = 'Parcel 1'
land_2['series_name'] = 'Parcel 2'
land_3['series_name'] = 'Parcel 3'
# Production data
production = df[(df['series_name'] == 'Scenario total') & (df['plot_key'] == 'prod')]
# SSR data
self_sufficiency_ratio = df[(df['series_name'] == 'Scenario') & (df['plot_key'] == 'ssr')]
# Food price index
food_price_index = df[(df['series_name'] == 'Scenario') & (df['plot_key'] == 'food_price')]
# Emissions
emissions = df[(df['series_name'] == 'Scenario total') & (df['plot_key'] == 'emis')]
# Palma ratio and Gini index
palma_ratio = df[(df['series_name'] == 'Scenario') & (df['plot_key'] == 'palma')]
gini_index = df[(df['series_name'] == 'Scenario') & (df['plot_key'] == 'gini')]
# Food security index
food_security_index = df[(df['series_name'] == 'Scenario') & (df['plot_key'] == 'food_sec')]
# Food demand
food_demand = df[(df['series_name'] == 'Scenario') & (df['plot_key'] == 'cal_demand')]
# Real income
real_income = df[(df['series_name'] == 'Scenario') & (df['plot_key'] == 'inc')]
# convert to thousands of pounds per year
real_income['value'] = real_income['value'] / 1e3
redistribution_factor = df[df['plot_key'] == 'lambda']
#If the value barely changes over time, set it to the initial value
if (redistribution_factor['value'].iloc[0] - redistribution_factor['value'].iloc[-1] < 1e-6):
    redistribution_factor['value'].iloc[:] = redistribution_factor['value'].iloc[0] * 1e5
    #redistribution_factor['value'].iloc[:] = redistribution_factor['value'].iloc[0]

################################################
# Plots
soil_plots(land_1, land_2, land_3, save_as='baseline_scenario_soil_plot.eps')
regular_plot(production, ylabel='Food Production [Mt/yr]', panel_name = '(b)', xlim=production['time'].max(), plot_color='blue', rescale=1e6, figsize=(2.8, 2.5), save_as='baseline_scenario_production_plot.eps')
regular_plot(emissions, ylabel='Net Emissions [MtCO$_2$eq/yr]', rescale=1e6, panel_name = '(c)', xlim=emissions['time'].max(), plot_color="#D11F00", figsize=(2.8, 2.5), save_as='baseline_scenario_emissions_plot.eps')
regular_plot(food_demand, ylabel='Food consumption [Mt/yr]', xlim=food_demand['time'].max(),panel_name = '(d)', plot_color='cyan', rescale=1e6, save_as='baseline_scenario_food_demand_plot.eps')
regular_plot(self_sufficiency_ratio, ylabel='Self-sufficiency Ratio [$\%$]', panel_name = '(e)', panel_xc=0.90, xlim=self_sufficiency_ratio['time'].max(), plot_color='green', save_as='baseline_scenario_ssr_plot.eps')
regular_plot(food_price_index, ylabel='Food Price Index [£/yr]', panel_name = '(f)', xlim=food_price_index['time'].max(), plot_color='orange', save_as='baseline_scenario_food_price_plot.eps')
combined_plot(gini_index, palma_ratio, num_bins=5, ylabel_1='Palma Ratio', ylabel_2='Gini Index',panel_name = '(h)', panel_xc= 0.85 , xlim=palma_ratio['time'].max(), color_1='purple', color_2='salmon', save_as='baseline_scenario_palma_gini_plot.eps')
combined_plot(real_income, redistribution_factor, num_bins=5, ylabel_1='Average Income [£k/yr]', ylabel_2='Redistribution Factor [$10^{-5}$]', panel_name = '(g)', xlim=real_income['time'].max(), color_1='magenta', color_2='teal', save_as='baseline_scenario_income_lambda_plot.eps')
regular_plot(food_security_index, ylabel='Food Security Index', panel_name = '(i)', xlim=food_security_index['time'].max(), plot_color='brown', save_as='baseline_scenario_food_security_plot.eps')

