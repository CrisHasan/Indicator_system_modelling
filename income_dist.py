import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
# Equivalised household income distribution for the UK, 2022
# Source: https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/adhocs/13182distributionofhouseholdincome
# Ideally get the data for Scotland, but only the graph is available: https://data.gov.scot/poverty/2024/
income_data = pd.read_excel('UK_income_dist_2022.xlsx', sheet_name="Average disposable income", usecols="A:B", skiprows=2, nrows=170)
#gross_income_data = pd.read_excel('UK_income_dist_2022.xlsx', sheet_name="Average gross income", usecols="A:B", skiprows=2, nrows=170)
num_rows = income_data.shape[0]
num_households = income_data['Frequency'].sum()
# Add a culumn for probability density function (frequency / total households)
income_data['Probability Density'] = income_data['Frequency'] / num_households
# Calculate the cumulative distribution function (CDF) of the empirical data
income_data['Cumulative Density'] = income_data['Probability Density'].cumsum() 
###############################################
# Plot the histogram of the income distribution
plt.figure(figsize=(12, 8))
#plt.hist(income_data['Income threshold'], weights=income_data['Frequency'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
plt.hist(income_data['Income threshold'], weights=income_data['Frequency'], bins=num_rows, color='lightgreen', alpha=0.7, edgecolor='black')
plt.xlabel('Equivalised Household Income (£)')
plt.ylabel('Frequency')
plt.title('Equivalised Household Income Distribution in the UK (2022)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###############################################
# Plot the probability density function
plt.figure(figsize=(12, 8))
plt.plot(income_data['Income threshold'], income_data['Probability Density'], color='blue') 
plt.xlabel('Equivalised Household Income (£)')
plt.ylabel('Probability Density Function')
plt.title('Probability Density Function of Equivalised Household Income in the UK (2022)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###############################################
# Calculate the expected value (mean) and variance of the income distribution
expected_value = (income_data['Income threshold'] * income_data['Probability Density']).sum()
variance = ((income_data['Income threshold'] - expected_value) ** 2 * income_data['Probability Density']).sum()
# To fit it with a gamma distribution we can approximate lambda as mean/variance, and alpha as mean*lambda
lambda_param = expected_value / variance
alpha_param = expected_value * lambda_param
# Plot the fitted gamma distribution
x = income_data['Income threshold']
pdf_fitted = gamma.pdf(x, a=alpha_param, scale=1/lambda_param)
plt.figure(figsize=(12, 8))
plt.plot(x, income_data['Probability Density']/1100, label='Empirical PDF (rescaled)', color='blue')
plt.plot(x, pdf_fitted, label='Fitted Gamma PDF', color='red', linestyle='--')
plt.xlabel('Equivalised Household Income (£)')
plt.ylabel('Probability Density Function')
plt.title('Fitted Gamma Distribution vs Empirical PDF of Equivalised Household Income in the UK (2022)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###############################################
"""Proper regression of alpha and lambda"""


###############################################
# Calculate the cumulative distribution function (CDF) of the fitted gamma distribution
cdf_fitted = gamma.cdf(x, a=alpha_param, scale=1/lambda_param)
# Compute the Kolmogorov-Smirnov statistic to assess the goodness of fit
ks_statistic = max(abs(income_data['Cumulative Density'] - cdf_fitted))
# Plot the empirical CDF and the fitted gamma CDF
plt.figure(figsize=(12, 8))
plt.plot(x, income_data['Cumulative Density'], label='Empirical CDF', color='blue')
plt.plot(x, cdf_fitted, label='Fitted Gamma CDF', color='red', linestyle='--')
plt.xlabel('Equivalised Household Income (£)')
plt.ylabel('Cumulative Distribution Function')
plt.title('Fitted Gamma CDF vs Empirical CDF. Kolmogorov-Smirnov Statistic: {:.4f}'.format(ks_statistic))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()