import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from typing import Dict
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
################################################
#set the default font for all figures
font_families = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
#plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.family'] = 'helvet'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.usetex'] = True
# Global text size for figures
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
baseline_linewidth = 2.0
################################################
os.chdir('/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/paper_simulations/critical_threshold_plot/')  
###############################################
def g_fertiliser(F: float, zeta: float, D0: float) -> float:
    """g(F) = F^2 - 2*zeta*F + D0"""
    F = float(F)
    zeta = float(zeta)
    D0 = float(D0)
    return F * F - 2.0 * zeta * F + D0

def alpha_logistic(S: float, alpha_max: float, rho: float, S_T: float) -> float:
    z = np.clip(-rho * (S - S_T), -160, 160)
    return alpha_max / (1.0 + np.exp(z))

def dS_dt_RHS(S: float, Dt: float, alpha_max: float, rho: float, S_T: float) -> float:
    """Evaluate the RHS of dS/dt in terms of S and D(t)."""
    alpha_S = alpha_logistic(S, alpha_max, rho, S_T)
    rhs = (alpha_S - Dt) * S * (1.0 - S)
    return rhs

def analytical_solution(B_con: float, alpha_max: float, rho: float, S_T: float, F: float, zeta: float, D0: float) -> float:
    """Compute the analytical solution of dS/dt for constant B(t) = B_con."""
    D_con = g_fertiliser(F, zeta, D0) * B_con
    S_eq = -(1.0/rho) * (np.log(alpha_max - D_con) - np.log(D_con)) + S_T
    return S_eq

###############################################
# ---- Input parameters ----
high = 0.9
low = 0.1
F = 1.25
zeta = 1.0
D0 = 1.2
alpha_max = 0.25
rho = 50.0
S_T = 0.2
[S_min, S_max] = [0.0, 1.0]
Dt_min = g_fertiliser(F, zeta, D0) * low
Dt_max = g_fertiliser(F, zeta, D0) * high
###############################################
# Array of S values
S_values = np.linspace(S_min, S_max, 101)
Dt_values = np.linspace(Dt_min, Dt_max, 101)
# Create a meshgrid of S and Dt values
S_grid, Dt_grid = np.meshgrid(S_values, Dt_values)
# Evaluate the RHS of dS/dt for each combination of S and Dt
dS_dt_values = dS_dt_RHS(S_grid, Dt_grid, alpha_max, rho, S_T)

# Create a contour plot 
'''
plt.figure(figsize=(4, 2.5), dpi=180)
contour = plt.contourf(S_grid, Dt_grid, dS_dt_values, levels=50, cmap='RdBu_r')
plt.colorbar(contour)
plt.text(1.03, 1.02, r'$\frac{dS}{dt}$', transform=plt.gca().transAxes, fontsize=12, ha='center')
zero_contour = plt.contour(S_grid, Dt_grid, dS_dt_values, levels=[0], colors='black', linewidths=1.5)
plt.clabel(zero_contour, fmt={0: r'$dS/dt = 0$'}, inline=True, fontsize=8)
plt.xlabel(r'$S(t)$')
plt.ylabel(r'$D(t)$')
plt.xlim(S_min - 0.05, S_max+0.05)
plt.ylim(Dt_min, Dt_max)
plt.tight_layout()
plt.savefig("dS_dt_contour_plot.eps", bbox_inches='tight')
plt.show()
'''

# Evaluate dS/dt for Dt_min and Dt_max
dS_dt_min = dS_dt_RHS(S_values, Dt_min, alpha_max, rho, S_T) # \tau = 1
dS_dt_max = dS_dt_RHS(S_values, Dt_max, alpha_max, rho, S_T) # \tau = 0
# Find the root of dS/dt (0<S<1) for Dt_min and Dt_max
root_min = brentq(dS_dt_RHS, S_min + 1e-6, S_max - 1e-6, args=(Dt_min, alpha_max, rho, S_T))
root_max = brentq(dS_dt_RHS, S_min + 1e-6, S_max - 1e-6, args=(Dt_max, alpha_max, rho, S_T))

# Plot DS/dt vs S for different Dt_min and Dt_max values
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axvspan(root_min, root_max, color='lightgrey', label=r'Critical region', alpha=0.5)
plt.axvline(x=root_min, color='gray', linestyle='--', label = r'$C^-$', linewidth=1.0)
plt.axvline(x=root_max, color='gray', linestyle='--', label = r'$C^+$', linewidth=1.0)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
plt.axvline(x=S_T, color="darkorange", linestyle='--', linewidth=1.5)
plt.plot(S_values, dS_dt_min, color='deepskyblue', linewidth=1.5)
plt.plot(S_values, dS_dt_max, color='royalblue', linewidth=1.5)
plt.text(S_T - 0.072, 0.04, r'$S_T$', color='darkorange')
plt.text(0.47, 0.048, r'$\tau = 1$', color='deepskyblue') 
plt.text(0.7, 0.005, r'$\tau = 0$', color='royalblue') 
plt.xlabel(r'$S(t)$')
plt.ylabel(r'$\frac{dS}{dt}$')
plt.xlim(S_min, S_max)
plt.ylim(dS_dt_values.min() - 0.002, dS_dt_values.max() + 0.002)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("dS_dt_vs_S_plot.eps", bbox_inches='tight')
plt.savefig("dS_dt_vs_S_plot.pdf", bbox_inches='tight')
plt.show()
######################################
# Find bifurcation points of the autonomous system
bifurcation_tau0_lower = brentq(lambda F: g_fertiliser(F, zeta, D0) - alpha_max/high, 0.0, 1.0)
bifurcation_tau0_upper = brentq(lambda F: g_fertiliser(F, zeta, D0) - alpha_max/high, 1.0, 2.0)
bifurcation_tau1_upper = brentq(lambda F: g_fertiliser(F, zeta, D0) - alpha_max/low, 1.0, 3.0)
# Loop over values of F and find the critical region for each F
F_values_tau0 = np.linspace(bifurcation_tau0_lower, bifurcation_tau0_upper, 280)
critical_tau0 = np.zeros(len(F_values_tau0))
analytical_tau0 = np.zeros(len(F_values_tau0))
for i, F in enumerate(F_values_tau0):  
    Dt_max = g_fertiliser(F, zeta, D0) * high
    # Find the S-value root of dS/dt (0<S<1)
    root_max = brentq(dS_dt_RHS, S_min + 1e-6, S_max - 1e-6, args=(Dt_max, alpha_max, rho, S_T))
    critical_tau0[i] = root_max
    # This is just to check that the root found above agrees the analytical solution
    analytical_tau0[i] = analytical_solution(high, alpha_max, rho, S_T, F, zeta, D0)

# Correct the rounding error by changing the first and last values to 1.0, which is the correct value at the two bifurcation points
critical_tau0[0] = 1.0
critical_tau0[-1] = 1.0
analytical_tau0[0] = 1.0
analytical_tau0[-1] = 1.0

# Do the same for tau = 1, i.e. find the critical region for each F
F_values_tau1 = np.linspace(0, bifurcation_tau1_upper - 1e-15, 280) # remove 1e-15 correct the rounding error
critical_tau1 = np.zeros(len(F_values_tau1))
for i, F in enumerate(F_values_tau1):  
    Dt_min = g_fertiliser(F, zeta, D0) * low
    root_min = brentq(dS_dt_RHS, S_min + 1e-6, S_max - 1e-6, args=(Dt_min, alpha_max, rho, S_T))
    critical_tau1[i] = root_min

critical_tau1[-1] = 1.0

# Plot bifurcation of the autonomous system for tau = 0
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([0, bifurcation_tau0_lower], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)  
plt.plot([bifurcation_tau0_upper, 3], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)  
plt.plot([bifurcation_tau0_lower, bifurcation_tau0_upper], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)  
plt.plot(F_values_tau0, critical_tau0, color='royalblue', linewidth=baseline_linewidth, linestyle='--')
plt.plot(bifurcation_tau0_lower, 1.0, 'o', color='crimson', markersize=4)
plt.plot(bifurcation_tau0_upper, 1.0, 'o', color='crimson', markersize=4)
plt.text(0.55, 0.41, r'$S^u$', color='royalblue') 
#plt.text(1.3, 0.91, r'$T^+$', color='crimson', fontsize=10)
#plt.text(0.75, 0.91, r'$T^-$', color='crimson', fontsize=10)
plt.text(0.1, 0.91, r'$S^+$', color='black', fontsize=10)
plt.text(0.1, 0.03, r'$S^-$', color='black', fontsize=10)

plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(0, 3)
plt.tight_layout()
plt.savefig("tau_0_bifurcation.eps", bbox_inches='tight')
plt.savefig("tau_0_bifurcation.pdf", bbox_inches='tight')
plt.show()

# Plot bifurcation of the autonomous system for tau = 1
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([0, bifurcation_tau1_upper], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)  
plt.plot([bifurcation_tau1_upper, 3], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)  
plt.plot(F_values_tau1, critical_tau1, color='deepskyblue', linewidth=baseline_linewidth, linestyle='--')
plt.plot(bifurcation_tau1_upper, 1.0, 'o', color='crimson', markersize=4)
plt.text(1.5, 0.21, r'$S^u$', color='deepskyblue') 
plt.text(2.55, 0.91, r'$T^+$', color='crimson', fontsize=10)
plt.text(0.1, 0.91, r'$S^+$', color='black', fontsize=10)
plt.text(0.1, 0.03, r'$S^-$', color='black', fontsize=10)
plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(0, 3)
plt.tight_layout()
plt.savefig("tau_1_bifurcation.eps", bbox_inches='tight')
plt.savefig("tau_1_bifurcation.pdf", bbox_inches='tight')
plt.show()

# Plot the combined bifurcation diagram for both tau = 0 and tau = 1
plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=baseline_linewidth)
plt.plot([0, bifurcation_tau0_lower], [1, 1], color='orange', linestyle='-.', linewidth=baseline_linewidth)  
plt.plot([bifurcation_tau0_lower, bifurcation_tau0_upper], [1, 1], color='black', linestyle='-', linewidth=baseline_linewidth)  
plt.plot([bifurcation_tau0_upper, bifurcation_tau1_upper], [1, 1], color='orange', linestyle='-.', linewidth=baseline_linewidth)  
plt.plot([bifurcation_tau1_upper, 3], [1, 1], color='black', linestyle='--', linewidth=baseline_linewidth)  

plt.plot(F_values_tau0, critical_tau0, color='royalblue', linewidth=baseline_linewidth, linestyle='--', label=r'$\tau = 0$') 
plt.plot(F_values_tau1, critical_tau1, color='deepskyblue', linewidth=baseline_linewidth, linestyle='--', label=r'$\tau = 1$') 
plt.plot(bifurcation_tau0_lower, 1.0, 'o', color='crimson', markersize=4)
plt.plot(bifurcation_tau0_upper, 1.0, 'o', color='crimson', markersize=4)
plt.plot(bifurcation_tau1_upper, 1.0, 'o', color='crimson', markersize=4)
plt.text(0.37, 0.41, r'$S^u_{\tau = 0}$', color='royalblue') 
plt.text(2.2, 0.41, r'$S^u_{\tau = 1}$', color='deepskyblue') 
plt.text(0.1, 0.91, r'$S^+$', color='black', fontsize=10)
plt.text(0.1, 0.03, r'$S^-$', color='black', fontsize=10) 
plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(0, 3)
plt.tight_layout()
plt.savefig("bifurcation_combined_taus.pdf", bbox_inches='tight')
plt.savefig("bifurcation_combined_taus.eps", bbox_inches='tight')
plt.show()  




####################################
# This just to plot the midle equilibrium solution for the autonomous system
F_test = bifurcation_tau1_upper + 0.0000000000000002
# plot log(alpha_max - g(F)*0.1) against F
F_values = np.linspace(2.51657, F_test, 10000000)
log_values = np.zeros(len(F_values))
for i, F in enumerate(F_values):
    if g_fertiliser(F, zeta, D0) * low >= alpha_max:
        log_values[i] = np.nan
    else:
        log_values[i] = -(np.log(alpha_max - g_fertiliser(F, zeta, D0) * low) - np.log(g_fertiliser(F, zeta, D0) * low))/rho + S_T

plt.figure(figsize=(4, 2.5), dpi=180)
plt.plot(F_values, log_values, color='blue', linewidth=baseline_linewidth)
plt.xlabel(r'$F$')
plt.ylabel(r'$S$')
plt.xlim(2.51657, F_test)
plt.tight_layout()
plt.show()