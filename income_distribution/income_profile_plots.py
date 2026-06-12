import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma
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
os.chdir('/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/income_distribution/')  
x = np.linspace(0, 200000, 1000)
#mu_param = 36425.84230537908
mu_param = 36000
#lambda_param = 7.9264511228399e-05
lambda_param = 8.0e-05
alpha_param = mu_param * lambda_param
mu_low = 25000
mu_high = 55000
pdf_fitted_0 = gamma.pdf(x, a=alpha_param, scale=1/lambda_param)
pdf_mu_low = gamma.pdf(x, a=mu_low*lambda_param, scale=1/lambda_param)
pdf_mu_high = gamma.pdf(x, a=mu_high*lambda_param, scale=1/lambda_param)

lambda_low = 3.0e-05
lambda_high = 4.0e-04
pdf_lambda_low = gamma.pdf(x, a=mu_param*lambda_low, scale=1/lambda_low)
pdf_lambda_high = gamma.pdf(x, a=mu_param*lambda_high, scale=1/lambda_high)

plt.figure(figsize=(4, 2.5), dpi=150)
plt.plot(x/1000, pdf_mu_low, color='#6693F5', linewidth=baseline_linewidth)
plt.fill_between(x/1000, 0, pdf_mu_low, label=r'$\mu = {}$'.format(mu_low), alpha=0.8, color='#6693F5') 
plt.plot(x/1000, pdf_fitted_0, color='#bd0d00', linewidth=baseline_linewidth)
plt.fill_between(x/1000, 0, pdf_fitted_0, label=r'$\mu = {}$'.format(mu_param), alpha=0.5, color='#bd0d00')
plt.plot(x/1000, pdf_mu_high, color='#fad46b', linewidth=baseline_linewidth)
plt.fill_between(x/1000, 0, pdf_mu_high, label=r'$\mu = {}$'.format(mu_high), alpha=0.3, color='#fad46b')
plt.xlabel('Equivalised Annual Household Income (£k)')
plt.xlim(0, 200)
plt.ylabel('Probability Density Function')
plt.title(r'(a) $\lambda = 8 \times 10^{-5}$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()  
plt.savefig('income_dist_different_mu.png', bbox_inches='tight')
plt.savefig('income_dist_different_mu.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(4, 2.5), dpi=150)
plt.plot(x/1000, pdf_lambda_low, color='#6693F5', linewidth=baseline_linewidth)
plt.fill_between(x/1000, 0, pdf_lambda_low, label=r'$\lambda = {}$'.format(lambda_low), alpha=0.8, color='#6693F5') 
plt.plot(x/1000, pdf_fitted_0, color='#bd0d00', linewidth=baseline_linewidth)
plt.fill_between(x/1000, 0, pdf_fitted_0, label=r'$\lambda = {}$'.format(lambda_param), alpha=0.5, color='#bd0d00')
plt.plot(x/1000, pdf_lambda_high, color='#fad46b', linewidth=baseline_linewidth)
plt.fill_between(x/1000, 0, pdf_lambda_high, label=r'$\lambda = {}$'.format(lambda_high), alpha=0.3, color='#fad46b')
plt.xlabel('Equivalised Annual Household Income (£k)')
plt.xlim(0, 200)
plt.ylabel('Probability Density Function')
plt.title(r'(b) $\mu = 36000$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()  
plt.savefig('income_dist_different_lambda.png', bbox_inches='tight')
plt.savefig('income_dist_different_lambda.pdf', bbox_inches='tight')
plt.show()  













