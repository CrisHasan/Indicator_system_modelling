import os
import pandas as pd
import matplotlib.pyplot as plt
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
baseline_linewidth = 2.5
###############################################
# Equivalised household income distribution for the UK, 2022
# Source: https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/adhocs/13182distributionofhouseholdincome
# Ideally get the data for Scotland, but only the graph is available: https://data.gov.scot/poverty/2024/
os.chdir('/Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/income_distribution/')  
raw_income_data = pd.read_excel('UK_income_dist_2022.xlsx', sheet_name="Average disposable income", usecols="A:B", skiprows=2, nrows=170)
# The first data point is too large and it skews the entire distribution; perhaps ana outlier. so we will set it to half of the second data point. 
raw_income_data.loc[0, 'Frequency'] = raw_income_data.loc[1, 'Frequency'] / 2
#gross_income_data = pd.read_excel('UK_income_dist_2022.xlsx', sheet_name="Average gross income", usecols="A:B", skiprows=2, nrows=170)
num_rows = raw_income_data.shape[0]
num_households = raw_income_data['Frequency'].sum()
step_size = 1000
filled_rows = []
# If a gap is expanded, the right endpoint frequency is split across the
# inserted points and the right endpoint itself.
overridden_frequency = {}
# Interpolate the data to fill in the gaps between income thresholds, by adding new rows with a step size of 1000, and splitting the frequency of the right endpoint across the new points and the right endpoint itself
for i in range(len(raw_income_data) - 1):
    left_x = raw_income_data.loc[i, 'Income threshold']
    left_y = overridden_frequency.get(i, raw_income_data.loc[i, 'Frequency'])
    right_x = raw_income_data.loc[i + 1, 'Income threshold']
    right_y = raw_income_data.loc[i + 1, 'Frequency']

    filled_rows.append({'Income threshold': left_x, 'Frequency': left_y})

    gap = right_x - left_x
    new_points = list(range(int(left_x + step_size), int(right_x), step_size)) if gap > step_size else []

    if new_points:
        split_value = right_y / (len(new_points) + 1)
        for x_new in new_points:
            filled_rows.append({'Income threshold': x_new, 'Frequency': split_value})
        overridden_frequency[i + 1] = split_value

# Keep the last original point.
last_idx = len(raw_income_data) - 1
filled_rows.append({
    'Income threshold': raw_income_data.loc[last_idx, 'Income threshold'],
    'Frequency': overridden_frequency.get(last_idx, raw_income_data.loc[last_idx, 'Frequency'])
})
income_data = pd.DataFrame(filled_rows)
num_rows = income_data.shape[0]
x = income_data['Income threshold']
# Add a culumn for probability density function (frequency / total households)
income_data['Probability Density'] = income_data['Frequency'] / num_households
# Calculate the cumulative distribution function (CDF) of the empirical data
income_data['Cumulative Density'] = income_data['Probability Density'].cumsum() 
###############################################
# Plot the histogram of the income distribution
plt.figure(figsize=(5, 3), dpi=150)
plt.hist(x, weights=income_data['Frequency'], bins=num_rows, color='lightgreen', alpha=0.7, edgecolor='black')
plt.plot(x, income_data['Probability Density']*num_households, color='blue') 
plt.xlabel('Equivalised Household Income (£)')
plt.ylabel('Frequency')
plt.title('Equivalised Household Income Distribution in the UK (2022)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###############################################
# Fit a gamma distribution using lambda and alpha parameters
# Calculate the expected value (mean) and variance of the income distribution
expected_value = (x * income_data['Probability Density']).sum()
variance = ((x - expected_value) ** 2 * income_data['Probability Density']).sum()
# To fit it with a gamma distribution we can approximate lambda as mean/variance, and alpha as mean*lambda
lambda_param = expected_value / variance
alpha_param = expected_value * lambda_param
# Plot the fitted gamma distribution
pdf_fitted = gamma.pdf(x, a=alpha_param, scale=1/lambda_param)
# Estimate the scale factor in order to compare the two distributions
bin_width = x[1] - x[0]
plt.figure(figsize=(4, 2.5), dpi=150)
plt.hist(x/1000, weights=income_data['Probability Density']/bin_width, bins=num_rows, color='lightblue', alpha=0.7, edgecolor='lightblue', label='Observed')
plt.plot(x/1000, pdf_fitted, label='Gamma fit', color='red', linestyle='--', linewidth = baseline_linewidth)
plt.xlabel('Equivalised Annual Household Income (£k)')
plt.ylabel('Probability Density Function')
plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=10)
plt.xlim(0, 200)
#plt.title('Fitted Gamma Distribution vs Empirical PDF of Equivalised Household Income in the UK (2022)')
plt.legend()
plt.tight_layout()
plt.savefig('fitted_gamma_distribution.eps', bbox_inches='tight')
plt.show()
#############################
# Plot the the fitted gamma distribution rescaled by the number of households, to compare it with the histogram
# Discritize the fitted PDF by multiplying it by the total number of households and the bin width (x[1] - x[0]) to get the expected frequency in each bin, which can be compared to the histogram frequencies
pdf_fitted_discrete_frequencies = pdf_fitted * num_households * bin_width
plt.figure(figsize=(4, 2.5), dpi=150)
plt.hist(x/1000, weights=income_data['Frequency'], bins=num_rows, color='lightgreen', alpha=0.7, edgecolor='black', label='Observed Histogram')
plt.plot(x/1000, pdf_fitted_discrete_frequencies, label='Gamma fit', color='red', linestyle='--', linewidth=baseline_linewidth)
plt.xlabel('Equivalised Annual Household Income (£k)')
plt.ylabel('Frequency')
plt.title('Fitted Gamma Distribution vs Observed Histogram of Equivalised Household Income in the UK (2022)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###############################################
# Validation
# Calculate the cumulative distribution function (CDF) of the fitted gamma distribution
cdf_fitted = gamma.cdf(x, a=alpha_param, scale=1/lambda_param)
# Compute the Kolmogorov-Smirnov statistic to assess the goodness of fit
ks_statistic = max(abs(income_data['Cumulative Density'] - cdf_fitted))
# Plot the empirical CDF and the fitted gamma CDF
plt.figure(figsize=(4, 2.5), dpi=150)
plt.plot(x/1000, income_data['Cumulative Density'], label='Observed', color='blue', linewidth = baseline_linewidth)
plt.plot(x/1000, cdf_fitted, label='Gamma fit', color='red', linestyle='--', linewidth=baseline_linewidth)
plt.xlabel('Equivalised Annual Household Income (£k)')
plt.ylabel('Cumulative Distribution Function')
plt.title('Kolmogorov-Smirnov Statistic: {:.4f}'.format(ks_statistic))
plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=10)
plt.legend()
#plt.grid(True, alpha=0.3)
plt.xlim(0, 200)
plt.tight_layout()
plt.savefig('fitted_gamma_cdf.eps', bbox_inches='tight')
plt.show()