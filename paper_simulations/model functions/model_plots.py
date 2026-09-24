# cd /Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/model_plots
import numpy as np
import matplotlib.pyplot as plt
import os
################################################
os.chdir('/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/paper_simulations/model functions')
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
###################################
def alpha(S, alpha_max=0.25, rho=50, S_T=0.2):
    return alpha_max / (1 + np.exp(-rho * (S - S_T)))

def plot_alpha_S_T(S_T_values, save_as=None):
    S_values = np.linspace(0, 1, 1000)
    plt.figure(figsize=(4, 2.5), dpi=150)
    colors = ['lightcoral', 'red', 'darkred']
    for S_T, color in zip(S_T_values, colors):
        alpha_values = alpha(S_values, S_T=S_T)
        plt.plot(S_values, alpha_values, label=f'$S_T = {S_T}$', linewidth=2, color=color)

    plt.text(0.02, 0.9, '(a)', transform=plt.gca().transAxes, fontsize=10)
    plt.xlabel(f'Soil Quality, $S$')
    plt.ylabel(f'Recovery Rate, $\\alpha(S)$ $[yr^{-1}]$')
    plt.xlim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()

def plot_alpha_rho(rho_values, save_as=None):
    S_values = np.linspace(0, 1, 1000)
    plt.figure(figsize=(4, 2.5), dpi=150)
    colors = ['lightblue', 'blue', 'darkblue']
    for rho, color in zip(rho_values, colors):
        alpha_values = alpha(S_values, rho=rho)
        plt.plot(S_values, alpha_values, label=f'$\\rho = {rho}$', linewidth=2, color=color)
    plt.text(0.02, 0.9, '(b)', transform=plt.gca().transAxes, fontsize=10)
    plt.xlabel(f'Soil Quality, $S$')
    plt.ylabel(f'Recovery Rate, $\\alpha(S)$ $[yr^{-1}]$')
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()


###################################
def binary_state(t, omega_values, phi_values, tau=1):
    N = len(omega_values)
    B_values = np.zeros((N, len(t)))
    for i in range(N):
        omega = omega_values[i]
        phi = phi_values[i]
        t_shifted = t - phi
        t_mod = t_shifted % omega
        B_values[i, :] = np.where((t_mod >= 0) & (t_mod < tau), 0, 1)

    return B_values

def plot_binary_state_(B_values, title, xlabel='Time', save_as=None):
    plt.figure(figsize=(6, 1.5), dpi=150)
    plt.imshow(B_values, cmap='YlGn', aspect='auto')
    plt.xlabel(xlabel)
    plt.ylabel('Land parcel')
    plt.title(title)
    plt.yticks(range(B_values.shape[0]), range(1, B_values.shape[0] + 1))
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()


###################################
def degredation_factor(F_array,zeta,D0):
    g_F = F_array**2 - 2*zeta*F_array + D0
    return g_F

def plot_degredation_factor(F_array, zeta, D0_values, save_as=None):
    plt.figure(figsize=(4, 2.5), dpi=150)
    colors = ['lightcoral', 'red', 'darkred']
    plt.text(-0.2, 28, '(a)', color='black')
    for D0, color in zip(D0_values, colors):
        g_F_values = degredation_factor(F_array, zeta, D0)
        plt.plot(F_array, g_F_values, label=f'$D_0 = {D0}$', linewidth=2, color=color)

    plt.xlabel(f'Fertiliser application rate, $F$ [M/(ha$\\cdot$yr)]')
    plt.ylabel(f'Degredation factor, $g(F)$ $[yr^{-1}]$')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()

###################################
def yield_factor(F, xi, psi):
    nonlinear_term = (-F**2 + xi*F) * np.exp(-psi*F)
    h_F = np.maximum(0, nonlinear_term)
    return h_F

def plot_yield_factor(F_array, xi_values, psi, save_as=None):
    """Plot yield_factor(F) for different values of xi"""
    plt.figure(figsize=(4, 2.5), dpi=150)
    colors = ['lightgreen', 'green', 'darkgreen']
    
    for xi, color in zip(xi_values, colors):
        h_values = yield_factor(F_array, xi, psi)
        plt.plot(F_array, h_values, label=f'$\\xi = {xi}$', linewidth=2, color=color)
    
    plt.text(-0.2, 9.2, '(b)', color='black')
    plt.xlabel(f'Fertiliser application rate, $F$ [M/(ha$\\cdot$yr)]')
    plt.ylabel(f'Yield factor, $h(F)$ [ton/(ha$\\cdot$yr)]')
    plt.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()

def plot_combined_factors(F_array, zeta, D0, xi, psi, save_as=None):
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(4, 2.5), dpi=150)
    # Calculate degradation factor
    g_F_values = degredation_factor(F_array, zeta, D0)
    color1 = 'black'
    ax1.set_xlabel('Fertiliser application rate, $F$ [M/(ha$\\cdot$yr)]')
    ax1.set_ylabel('Degradation factor, $g(F)$ $[yr^{-1}]$', color=color1)
    #ax1.axvline(x=1, color='gray', linestyle='--', linewidth=2.0)
    #ax1.axvline(x=1.61, color='pink', linestyle='--', linewidth=2.0)
    ax1.plot(F_array, g_F_values, color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    h_F_values = yield_factor(F_array, xi, psi)
    color2 = 'red'
    ax2.set_ylabel('Yield factor, $h(F)$ [ton/(ha$\\cdot$yr)]', color=color2)
    ax2.plot(F_array, h_F_values, color=color2, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    plt.show()


###################################
plot_alpha_S_T(S_T_values=[0.1, 0.2, 0.3], save_as='alpha_varied_ST.eps')
plot_alpha_rho(rho_values=[25, 50, 150], save_as='alpha_varied_rho.eps')
###################################
F_array = np.linspace(0, 6, 200)
zeta = 1
D0_values = [1.2, 3, 5]
plot_degredation_factor(F_array=F_array, zeta=zeta, D0_values=D0_values, save_as='degredation_factor.eps')
###################################
F_array = np.linspace(0, 6, 200)
xi_values = [7, 10, 15]
psi = 0.5
plot_yield_factor(F_array=F_array, xi_values=xi_values, psi=psi, save_as='yield_factor.eps')
combined_F_array = np.linspace(0, 8, 200)
plot_combined_factors(F_array=combined_F_array, zeta=1, D0=1.2, xi=10, psi=0.5, save_as='combined_factors.eps')
###################################
# First scenario
time_interval = np.linspace(0, 50, 51) # if one of the parameters of B(t) is not an integer, need to add more points to the time interval to capture the changes in B(t) accurately
omega_values=[3, 5]
phi_values=[2, 3]
last_year = time_interval[-1]
B_values = binary_state(t=time_interval, omega_values=omega_values, phi_values=phi_values, tau=1)
title = f'(a) $\\omega_1$={omega_values[0]}, $\\omega_2$={omega_values[1]}'
plot_binary_state_(B_values, title=title, xlabel='Time (years)', save_as='binary_state_scenario_1.eps')
# Second scenario
omega_values=[2, 5, 10]
phi_values=[1, 3, 5]
B_values = binary_state(t=time_interval, omega_values=omega_values, phi_values=phi_values, tau=1)
title = f'(b) $\\omega_1$={omega_values[0]}, $\\omega_2$={omega_values[1]}, $\\omega_3$={omega_values[2]}'
plot_binary_state_(B_values, title=title, xlabel='Time (years)', save_as='binary_state_scenario_2.eps')
###################################
# Plot one scenario with S_T =0.2 and rho = 50
alpha_max=0.25; rho=50; S_T=0.2
S_values = np.linspace(0, 1, 1000)
alpha_values = alpha(S_values, S_T=S_T)

plt.figure(figsize=(4, 2.5), dpi=180)
plt.axvline(x=S_T, color='royalblue', linestyle='--', linewidth=1.5)
plt.axhline(y=alpha_max, color='gray', linestyle='--', linewidth=1.5)
plt.plot(S_values, alpha_values, linewidth=2, color='black')
plt.text(S_T + 0.02, 0.05, r'$S_T$', color='royalblue', fontsize=10)
plt.text(0.02, alpha_max - 0.02, r'$\alpha_{max}$', color='gray', fontsize=10)
plt.xlabel(r'Soil Health, $S$')
plt.ylabel(r'Recovery Rate, $\alpha(S)$')
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('alpha_S_T_plot.eps', bbox_inches='tight')
plt.show()
###################################
# Plot one scenario with D0 = 1.2 and zeta = 1
F_array = np.linspace(0, 6, 200)
zeta = 1
D0 = 1.2
g_F_values = degredation_factor(F_array, zeta, D0)

plt.figure(figsize=(4, 2.5), dpi=180)
plt.axhline(y=D0, color='gray', linestyle='--', linewidth=1.5)
plt.axvline(x=zeta, color='royalblue', linestyle='--', linewidth=1.5)
plt.plot(F_array, g_F_values, linewidth=2, color='black')
plt.text(zeta + 0.08, 20, r'$\zeta$', color='royalblue', fontsize=10)
plt.text(4, D0 + 0.8, r'$D_0$', color='gray', fontsize=10)
plt.xlabel(r'Fertiliser Input, $F$')
plt.ylabel(r'Degredation Factor, $g(F)$')
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig('degredation_factor_plot.eps', bbox_inches='tight')
plt.show()

