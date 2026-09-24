from math import gamma
import os
import re
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gamma
import matplotlib.lines as mlines
################################################
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
################################################
TARGET_DIR = "/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/paper_simulations/scenario_analysis"
os.chdir(TARGET_DIR)
ROOT = Path(TARGET_DIR)
SCENARIO_FILE = ROOT / "scenario_analysis.py"

F_baseline = float(0.9)
eta_b = float(0.4)
T_max = float(80.0)
SCENARIOS = {
    "baseline_scenario": {"mean_tau": 0.3, "income_growth_rate": 0.002, "redistribution_rate": -0.005, "F_asymptotic": F_baseline, "F_baseline": F_baseline, "eta": 0.4, "eta_b": eta_b, "T_max": T_max},
    "sustainable_scenario": {"mean_tau": 0.5, "income_growth_rate": 0.001, "redistribution_rate": 0.005, "F_asymptotic": 0.8, "F_baseline": F_baseline, "eta": 0.4, "eta_b": eta_b, "T_max": T_max},
    "equitable_scenario": {"mean_tau": 0.3, "income_growth_rate": 0.001, "redistribution_rate": 0.020, "F_asymptotic": 0.8, "F_baseline": F_baseline, "eta": 0.4, "eta_b": eta_b, "T_max": T_max},
    "industrial_scenario": {"mean_tau": 0.25, "income_growth_rate": 0.005, "redistribution_rate": -0.020, "F_asymptotic": 1.25, "F_baseline": F_baseline, "eta": 0.4, "eta_b": eta_b, "T_max": T_max},
}
################################################
output_variables = ["t_eval", "mean_soil_quality", "food_insecurity_index_annual", "food_price_index_annual",
                     "ssr_annual", "palma", "gini", "P_modified_annual", "annual_J_investment", "average_real_income",
                       "lam_series", "food_consumption", "annual_emissions", "income_x", "inflation_index",
                       "fertiliser_series", "annual_production"]
################################################
def prepare_source(params):
    source = SCENARIO_FILE.read_text(encoding="utf-8")
    for param, value in params.items():
        source = re.sub(
            rf"^{re.escape(param)}\s*=\s*float\([^\n]+\)$",
            f"{param} = float({float(value)})",
            source,
            flags=re.MULTILINE,
        )
    return source

def run_scenario(params, output_variables=output_variables):
    namespace = {"__name__": "scenario_analysis_runner", "__file__": str(SCENARIO_FILE)}
    source = prepare_source(params)
    os.chdir(TARGET_DIR)
    exec(compile(source, str(SCENARIO_FILE), "exec"), namespace)
    result = {}
    for param in params:
        result[param] = float(namespace[param])
    for var in output_variables:
        result[var] = np.array([float(v) for v in namespace[var]])
    return result

def run_named_scenario(name):
    params = SCENARIOS[name]
    return run_scenario(params, output_variables=output_variables)

################################################
baseline_results = run_named_scenario("baseline_scenario")
sustainable_results = run_named_scenario("sustainable_scenario")
industrial_results = run_named_scenario("industrial_scenario")
equitable_results = run_named_scenario("equitable_scenario")
time_array = baseline_results["t_eval"]
################################################
# Reinforce the working directory
os.chdir(TARGET_DIR)
# Plot mean soil quality index
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(time_array, baseline_results["mean_soil_quality"], linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(time_array, sustainable_results["mean_soil_quality"], linewidth=baseline_linewidth, color='green', label='Sustainable Scenario')
plt.plot(time_array, industrial_results["mean_soil_quality"], linewidth=baseline_linewidth, color='red', label='Industrial Scenario')
plt.plot(time_array, equitable_results["mean_soil_quality"], linewidth=baseline_linewidth, color='blue', label='Equitable Scenario')
plt.xlabel("Time [yr]")
plt.ylabel(r"Mean Soil Quality Index")

plt.xlim(1, T_max)
plt.ylim(0, max(baseline_results["mean_soil_quality"], sustainable_results["mean_soil_quality"], industrial_results["mean_soil_quality"], equitable_results["mean_soil_quality"]) * 1.1)
plt.tight_layout()
plt.savefig('mean_soil_quality_plot.eps', bbox_inches='tight')
plt.show()

# Generate figure only for legend
legend_handles = [
    mlines.Line2D([], [], color="gray", lw=1.7, label="Baseline"),
    mlines.Line2D([], [], color="green", lw=1.7, label="Sustainable"),
    mlines.Line2D([], [], color="red", lw=1.7, label="Industrial"),
    mlines.Line2D([], [], color="blue", lw=1.7, label="Equitable"),
]
fig, ax = plt.subplots(figsize=(4.5, 0.2), dpi=150)
ax.axis("off")
ax.legend(handles=legend_handles,loc="center",ncol=4,frameon=False,fontsize=8)
plt.tight_layout()
plt.savefig('legend_plot.eps', bbox_inches='tight')
plt.show()

# Plot fertiliser series
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(time_array, baseline_results["fertiliser_series"], linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(time_array, sustainable_results["fertiliser_series"], linewidth=baseline_linewidth, color='green', label='Sustainable Scenario')
plt.plot(time_array, industrial_results["fertiliser_series"], linewidth=baseline_linewidth, color='red', label='Industrial Scenario')
plt.plot(time_array, equitable_results["fertiliser_series"], linewidth=baseline_linewidth, color='blue', label='Equitable Scenario')
plt.xlabel("Time [yr]")
plt.ylabel(r"Fertiliser Application Rate [M/ha]")
plt.xlim(0, T_max)
plt.tight_layout()
#plt.savefig('fertiliser_series_plot.eps', bbox_inches='tight')
plt.show()

# Plot annual production
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(np.arange(int(T_max) + 1), baseline_results["annual_production"]/1e6, linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(np.arange(int(T_max) + 1), sustainable_results["annual_production"]/1e6, linewidth=baseline_linewidth, color='green', label='Sustainable Scenario')
plt.plot(np.arange(int(T_max) + 1), industrial_results["annual_production"]/1e6, linewidth=baseline_linewidth, color='red', label='Industrial Scenario')
plt.plot(np.arange(int(T_max) + 1), equitable_results["annual_production"]/1e6, linewidth=baseline_linewidth, color='blue', label='Equitable Scenario')
plt.xlabel("Time [yr]")
plt.ylabel(r"Domestic Production [Mt]")

plt.xlim(0, T_max)
plt.ylim(0, max(max(baseline_results["annual_production"][1:]/1e6), max(sustainable_results["annual_production"][1:]/1e6), max(industrial_results["annual_production"][1:]/1e6), max(equitable_results["annual_production"][1:]/1e6)) * 1.1)
plt.tight_layout()
plt.savefig('annual_production_plot.eps', bbox_inches='tight')
plt.show()

# Plot food supply 
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(np.arange(int(T_max) + 1), baseline_results["P_modified_annual"]/1e6, linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(np.arange(int(T_max) + 1), sustainable_results["P_modified_annual"]/1e6, linewidth=baseline_linewidth, color='green', label='Sustainable Scenario')
plt.plot(np.arange(int(T_max) + 1), industrial_results["P_modified_annual"]/1e6, linewidth=baseline_linewidth, color='red', label='Industrial Scenario')
plt.plot(np.arange(int(T_max) + 1), equitable_results["P_modified_annual"]/1e6, linewidth=baseline_linewidth, color='blue', label='Equitable Scenario')
plt.xlabel("Time [yr]")
plt.ylabel(r"Food Supply [Mt]")

plt.xlim(1, T_max)
plt.ylim(0, max(max(baseline_results["P_modified_annual"][1:]/1e6), max(sustainable_results["P_modified_annual"][1:]/1e6), max(industrial_results["P_modified_annual"][1:]/1e6), max(equitable_results["P_modified_annual"][1:]/1e6)) * 1.1)
plt.tight_layout()
plt.show()

# Plot food consumption
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(time_array, baseline_results["food_consumption"]/1e12, linewidth=baseline_linewidth, color='black', label='All Scenarios')
plt.xlabel("Time [yr]")
plt.ylabel(r"Food Demand [$10^{12}\times$cal/yr]")
plt.xlim(0, T_max)
plt.ylim(0, max(baseline_results["food_consumption"]/1e12) * 1.1)
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('food_consumption_plot.eps', bbox_inches='tight')
plt.show()

# Plot annual emissions 
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(np.arange(int(T_max) + 1), baseline_results["annual_emissions"]/1e6, linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(np.arange(int(T_max) + 1), sustainable_results["annual_emissions"]/1e6, linewidth=baseline_linewidth, color='green', label='Sustainable Scenario')
plt.plot(np.arange(int(T_max) + 1), industrial_results["annual_emissions"]/1e6, linewidth=baseline_linewidth, color='red', label='Industrial Scenario')
plt.plot(np.arange(int(T_max) + 1), equitable_results["annual_emissions"]/1e6, linewidth=baseline_linewidth, color='blue', label='Equitable Scenario')
plt.xlabel("Time [yr]")
plt.ylabel(r"Annual Emissions [MtCO$_2$e]")
plt.xlim(1, T_max)

maxval = max(max(baseline_results["annual_emissions"][1:]/1e6), max(sustainable_results["annual_emissions"][1:]/1e6), max(industrial_results["annual_emissions"][1:]/1e6), max(equitable_results["annual_emissions"][1:]/1e6))
plt.ylim(0, maxval * 1.1)
plt.tight_layout()
plt.savefig('annual_emissions_plot.eps', bbox_inches='tight')
plt.show()


# Plot reinvestment J(t)
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(np.arange(int(T_max) + 1), baseline_results["annual_J_investment"]/1e9, linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(np.arange(int(T_max) + 1), sustainable_results["annual_J_investment"]/1e9, linewidth=baseline_linewidth, color='green', label='Sustainable Scenario')
plt.plot(np.arange(int(T_max) + 1), industrial_results["annual_J_investment"]/1e9, linewidth=baseline_linewidth, color='red', label='Industrial Scenario')
plt.plot(np.arange(int(T_max) + 1), equitable_results["annual_J_investment"]/1e9, linewidth=baseline_linewidth, color='blue', label='Equitable Scenario')
plt.xlabel("Time [yr]")
plt.ylabel(r"Import investments [£ bn]")
plt.xlim(0, T_max)

plt.ylim(0, max(max(baseline_results["annual_J_investment"][1:]/1e9), max(sustainable_results["annual_J_investment"][1:]/1e9), max(industrial_results["annual_J_investment"][1:]/1e9), max(equitable_results["annual_J_investment"][1:]/1e9)) * 1.1)
plt.tight_layout()
plt.tight_layout()
plt.savefig('annual_J_investment_plot.eps', bbox_inches='tight')
plt.show()

# Plot self-sufficiency ratio
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(np.arange(int(T_max) + 1), baseline_results["ssr_annual"], linewidth=baseline_linewidth, color='gray', label='Total Production')
plt.plot(np.arange(int(T_max) + 1), sustainable_results["ssr_annual"], linewidth=baseline_linewidth, color='green', label='Sustainable Production')
plt.plot(np.arange(int(T_max) + 1), industrial_results["ssr_annual"], linewidth=baseline_linewidth, color='red', label='Industrial Production')
plt.plot(np.arange(int(T_max) + 1), equitable_results["ssr_annual"], linewidth=baseline_linewidth, color='blue', label='Equitable Production')
plt.xlabel("Time [yr]")
plt.ylabel(r"Self-Sufficiency Ratio [\%]")
plt.xlim(0, T_max)
plt.ylim(0, max(max(baseline_results["ssr_annual"]), max(sustainable_results["ssr_annual"]), max(industrial_results["ssr_annual"]), max(equitable_results["ssr_annual"])) * 1.1)
plt.tight_layout()
plt.savefig('self_sufficiency_ratio_plot.eps', bbox_inches='tight')
plt.show()

# Plot food price index
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(np.arange(int(T_max) + 1), baseline_results["food_price_index_annual"]/1e3, linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(np.arange(int(T_max) + 1), sustainable_results["food_price_index_annual"]/1e3, linewidth=baseline_linewidth, color='green', label='Sustainable Scenario')
plt.plot(np.arange(int(T_max) + 1), industrial_results["food_price_index_annual"]/1e3, linewidth=baseline_linewidth, color='red', label='Industrial Scenario')
plt.plot(np.arange(int(T_max) + 1), equitable_results["food_price_index_annual"]/1e3, linewidth=baseline_linewidth, color='blue', label='Equitable Scenario')

plt.xlabel("Time [yr]")
plt.ylabel(r"Food Price Index [£k/yr]")
plt.xlim(1, T_max)
plt.ylim(0, max(max(baseline_results["food_price_index_annual"]/1e3), max(sustainable_results["food_price_index_annual"]/1e3), max(industrial_results["food_price_index_annual"]/1e3), max(equitable_results["food_price_index_annual"]/1e3)) * 1.1)
plt.tight_layout()
plt.show()

# Plot food insecurity index
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(np.arange(int(T_max) + 1), baseline_results["food_insecurity_index_annual"], linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(np.arange(int(T_max) + 1), sustainable_results["food_insecurity_index_annual"], linewidth=baseline_linewidth, color='green', label='Sustainable Scenario')
plt.plot(np.arange(int(T_max) + 1), industrial_results["food_insecurity_index_annual"], linewidth=baseline_linewidth, color='red', label='Industrial Scenario')
plt.plot(np.arange(int(T_max) + 1), equitable_results["food_insecurity_index_annual"], linewidth=baseline_linewidth, color='blue', label='Equitable Scenario')
plt.xlabel("Time [yr]")
plt.ylabel(r"Food Insecurity Index [\%]")
plt.xlim(1, T_max)
#plt.ylim(0, max(max(baseline_results["food_insecurity_index_annual"]), max(sustainable_results["food_insecurity_index_annual"]), max(industrial_results["food_insecurity_index_annual"]), max(equitable_results["food_insecurity_index_annual"]), max(integrated_results["food_insecurity_index_annual"])) * 1.1)
plt.ylim(0, 40)
plt.tight_layout()
plt.savefig('food_insecurity_index_plot.eps', bbox_inches='tight')
plt.show()

# Plot Gini index
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(time_array, baseline_results["gini"], linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(time_array, sustainable_results["gini"], linewidth=baseline_linewidth, color='green', linestyle='--', label='Sustainable Scenario')
plt.plot(time_array, industrial_results["gini"], linewidth=baseline_linewidth, color='red', linestyle='-.', label='Industrial Scenario')
plt.plot(time_array, equitable_results["gini"], linewidth=baseline_linewidth, color='blue', linestyle=':', label='Equitable Scenario')
plt.xlabel("Time [yr]")
plt.ylabel(r"Gini Index")
plt.xlim(0, T_max)
plt.ylim(0, max(max(baseline_results["gini"]), max(sustainable_results["gini"]), max(industrial_results["gini"]), max(equitable_results["gini"])) * 1.1)
plt.tight_layout()
plt.savefig('gini_index_plot.eps', bbox_inches='tight') 
plt.show()

# PLot Palma ratio
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(time_array, baseline_results["palma"], linewidth=baseline_linewidth, color='gray', label='Baseline Scenario')
plt.plot(time_array, sustainable_results["palma"], linewidth=baseline_linewidth, color='green', linestyle='--', label='Sustainable Scenario')
plt.plot(time_array, industrial_results["palma"], linewidth=baseline_linewidth, color='red', linestyle='-.', label='Industrial Scenario')
plt.plot(time_array, equitable_results["palma"], linewidth=baseline_linewidth, color='blue', linestyle=':', label='Equitable Scenario')
plt.xlabel("Time [yr]")
plt.ylabel(r"Palma Ratio")
plt.xlim(0, T_max)
plt.ylim(0, max(max(baseline_results["palma"]), max(sustainable_results["palma"]), max(industrial_results["palma"]), max(equitable_results["palma"]), max(integrated_results["palma"])) * 1.1)
plt.tight_layout()
plt.savefig('palma_ratio_plot.eps', bbox_inches='tight')
plt.show()

# Calculate income pdf
baseline_alpha_par_0 = baseline_results["average_real_income"][0] * baseline_results["lam_series"][0]
baseline_scale_par_0 = 1/baseline_results["lam_series"][0]
baseline_income_pdf_0 = gamma.pdf(x=baseline_results["income_x"], a=baseline_alpha_par_0, scale=baseline_scale_par_0)
baseline_alpha_par_T = baseline_results["average_real_income"][-1] * baseline_results["lam_series"][-1]
baseline_scale_par_T = 1/baseline_results["lam_series"][-1]
baseline_income_pdf_T = gamma.pdf(x=baseline_results["income_x"], a=baseline_alpha_par_T, scale=baseline_scale_par_T)    

sustainable_alpha_par_T = sustainable_results["average_real_income"][-1] * sustainable_results["lam_series"][-1]
sustainable_scale_par_T = 1/sustainable_results["lam_series"][-1]
sustainable_income_pdf_T = gamma.pdf(x=sustainable_results["income_x"], a=sustainable_alpha_par_T, scale=sustainable_scale_par_T)    
industrial_alpha_par_T = industrial_results["average_real_income"][-1] * industrial_results["lam_series"][-1]
industrial_scale_par_T = 1/industrial_results["lam_series"][-1]
industrial_income_pdf_T = gamma.pdf(x=industrial_results["income_x"], a=industrial_alpha_par_T, scale=industrial_scale_par_T)

equitable_alpha_par_T = equitable_results["average_real_income"][-1] * equitable_results["lam_series"][-1]
equitable_scale_par_T = 1/equitable_results["lam_series"][-1]
equitable_income_pdf_T = gamma.pdf(x=equitable_results["income_x"], a=equitable_alpha_par_T, scale=equitable_scale_par_T)
# Plot income pdfs
plt.figure(figsize=(2.8, 2.0), dpi=150)
plt.plot(baseline_results["income_x"], baseline_income_pdf_0, linewidth=baseline_linewidth, color='black', linestyle='-', label='Baseline Scenario (t=0)')
plt.plot(baseline_results["income_x"], baseline_income_pdf_T, linewidth=baseline_linewidth, color='gray', linestyle='--', label='Baseline Scenario (t=T)')
plt.plot(sustainable_results["income_x"], sustainable_income_pdf_T, linewidth=baseline_linewidth, color='green', linestyle='--', label='Sustainable Scenario (t=T)')
plt.plot(industrial_results["income_x"], industrial_income_pdf_T, linewidth=baseline_linewidth, color='red', linestyle='--', label='Industrial Scenario (t=T)')
plt.plot(equitable_results["income_x"], equitable_income_pdf_T, linewidth=baseline_linewidth, color='blue', linestyle='--', label='Equitable Scenario (t=T)')

plt.xlabel("Real Income [£k/yr]")
plt.ylabel(r"Income PDF")
plt.xlim(0, baseline_results["income_x"][-1])
plt.ylim(0, max(max(baseline_income_pdf_0), max(baseline_income_pdf_T), max(sustainable_income_pdf_T), max(industrial_income_pdf_T), max(equitable_income_pdf_T)) * 1.1)
plt.tight_layout() 
plt.show() 
