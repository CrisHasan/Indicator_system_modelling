import numpy as np
import matplotlib.pyplot as plt
import os
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
###############################################
os.chdir('/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/paper_simulations/binary_matrix_plot/')  
N = 10          # number of land parcels
T = 30          # total years (j = 0,1,...,T-1)
tau = 0.2       # duration of cover-crop phase (years)
resolution = 100 # number of time samples per year

# Generate phi matrix 
phi = np.random.uniform(0, 1 - tau, size=(N, T))
phi = np.round(phi, 2)   # approximate to two decimal places

# Define time interval [0, T)
steps_per_year = resolution
total_steps = T * steps_per_year
t_array = np.linspace(0, T, total_steps, endpoint=False) 


# 3. Compute B_i(t) for each parcel
B = np.zeros((N, total_steps)) 
for i in range(N):
    for idx, t in enumerate(t_array):
        j = int(np.floor(t))          # integer year j
        phase = phi[i, j]             # phase for this parcel & year
        # Check if (fractional_part - phase) mod 1 lies in [0, tau)
        condition_val = t - j - phase
        if condition_val >= 0 and condition_val < tau:
            B[i, idx] = 0.1
        else:
            B[i, idx] = 0.9


'''Method 2:
        condition_val = (fractional_part - phase) % 1.0
        if condition_val < tau:
            B[i, idx] = 0.1
        else:
            B[i, idx] = 0.9
'''

# Plot phi as a heatmap
fig, ax = plt.subplots(figsize=(6, 2.5), dpi=180)
im = ax.imshow(phi, aspect='auto', cmap='viridis', vmin=0, vmax=1-tau, extent=[0, T, N, 0])
cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
#plt.imshow(phi, aspect='auto', cmap='viridis', vmin=0, vmax=1-tau, extent=[0, T, N, 0])
#plt.colorbar()
plt.xlabel(r'Year Index ($j$)')
plt.ylabel(r'Land Parcel Index ($i$)')
plt.text(0.02, 1.045, '(a) Phase Matrix', transform=plt.gca().transAxes, fontsize=10)
plt.text(1.0, 1.045, r'$\phi_{i,j}$', transform=plt.gca().transAxes, fontsize=10)
plt.yticks(np.arange(0, N) + 0.5, range(1, N + 1))
plt.xticks(np.arange(4, T, 5) + 0.5, np.arange(5, T+5, 5))
plt.tight_layout()
plt.savefig('phase_matrix_plot.eps', bbox_inches='tight')
plt.show()
    

# plot B as a heatmap with x-axis as time and y-axis as land parcel index
fig, ax = plt.subplots(figsize=(6, 2.5), dpi=180)
im = ax.imshow(B, aspect='auto', cmap=plt.get_cmap('YlGn').resampled(2), extent=[0, T, N, 0])
cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
cbar.set_ticks([])
plt.xlabel(r'Time ($t$)')
plt.ylabel(r'Land Parcel Index ($i$)')
plt.text(0.02, 1.045, '(b) Binary Mode', transform=plt.gca().transAxes, fontsize=10)
plt.text(1., 1.045, r'$\mathcal{B}_i(t)$', transform=plt.gca().transAxes, fontsize=10)
plt.text(1.05, 0.72, r'$0.9$', transform=plt.gca().transAxes, fontsize=10)
plt.text(1.05, 0.22, r'$0.1$', transform=plt.gca().transAxes, fontsize=10)
plt.yticks(np.arange(0, N) + 0.5, range(1, N + 1))
plt.tight_layout()
plt.savefig('binary_mode_plot.eps', bbox_inches='tight')
plt.show()  
