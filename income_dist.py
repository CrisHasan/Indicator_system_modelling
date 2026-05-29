import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
# Equivalised household income distribution for the UK, 2022
# Source: https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/adhocs/13182distributionofhouseholdincome
# Ideally get the data for Scotland, but only the graph is available: https://data.gov.scot/poverty/2024/
raw_income_data = pd.read_excel('UK_income_dist_2022.xlsx', sheet_name="Average disposable income", usecols="A:B", skiprows=2, nrows=170)
#gross_income_data = pd.read_excel('UK_income_dist_2022.xlsx', sheet_name="Average gross income", usecols="A:B", skiprows=2, nrows=170)
num_rows = raw_income_data.shape[0]
num_households = raw_income_data['Frequency'].sum()
step_size = 1000
filled_rows = []
# If a gap is expanded, the right endpoint frequency is split across the
# inserted points and the right endpoint itself.
overridden_frequency = {}
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
# Add a culumn for probability mass function (frequency / total households)
income_data['Probability Mass'] = income_data['Frequency'] / num_households
# Calculate the cumulative distribution function (CDF) of the empirical data
income_data['Cumulative Density'] = income_data['Probability Mass'].cumsum() 
###############################################
# Plot the histogram of the income distribution
plt.figure(figsize=(12, 8))
plt.hist(x, weights=income_data['Frequency'], bins=num_rows, color='lightgreen', alpha=0.7, edgecolor='black')
plt.plot(x, income_data['Probability Mass']*num_households, color='blue') 
plt.xlabel('Equivalised Household Income (£)')
plt.ylabel('Frequency')
plt.title('Equivalised Household Income Distribution in the UK (2022)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###############################################
# Plot the probability mass function
plt.figure(figsize=(12, 8))
plt.plot(x, income_data['Probability Mass'], color='blue') 
plt.xlabel('Equivalised Household Income (£)')
plt.ylabel('Probability Mass Function')
plt.title('Probability Mass Function of Equivalised Household Income in the UK (2022)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
###############################################
# Fit a gamma distribution using lambda and alpha parameters
# Calculate the expected value (mean) and variance of the income distribution
expected_value = (x * income_data['Probability Mass']).sum()
variance = ((x - expected_value) ** 2 * income_data['Probability Mass']).sum()
# To fit it with a gamma distribution we can approximate lambda as mean/variance, and alpha as mean*lambda
lambda_param = expected_value / variance
alpha_param = expected_value * lambda_param
# Plot the fitted gamma distribution
pdf_fitted = gamma.pdf(x, a=alpha_param, scale=1/lambda_param)
# Estimate the scale factor in order to compare a fitted PDF (continuous) with the empirical PDF (discrete)
scale_estimate = income_data['Probability Mass'].max() / pdf_fitted.max()
plt.figure(figsize=(12, 8))
plt.plot(x, income_data['Probability Mass']/scale_estimate, label='Empirical PMF (rescaled)', color='blue')
plt.plot(x, pdf_fitted, label='Fitted Gamma PDF', color='red', linestyle='--')
plt.xlabel('Equivalised Household Income (£)')
plt.ylabel('Probability Mass Function')
plt.title('Fitted Gamma Distribution vs (Rescaled) Empirical PMF of Equivalised Household Income in the UK (2022)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#############################
# Plot the the fitted gamma distribution rescaled by the number of households, to compare it with the histogram
# Discritize the fitted PDF by multiplying it by the total number of households and the bin width (x[1] - x[0]) to get the expected frequency in each bin, which can be compared to the histogram frequencies
bin_width = x[1] - x[0]
pdf_fitted_discrete_frequencies = pdf_fitted * num_households * bin_width
plt.figure(figsize=(12, 8))
plt.hist(x, weights=income_data['Frequency'], bins=num_rows, color='lightgreen', alpha=0.7, edgecolor='black', label='Empirical Histogram')
plt.plot(x, pdf_fitted_discrete_frequencies, label='Fitted Gamma Distribution', color='red', linestyle='--')
plt.xlabel('Equivalised Household Income (£)')
plt.ylabel('Frequency')
plt.title('Fitted Gamma Distribution vs Empirical Histogram of Equivalised Household Income in the UK (2022)')
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