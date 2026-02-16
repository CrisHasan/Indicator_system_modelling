import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
# Equivalised household income distribution for the UK, 2022
# Source: https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/adhocs/13182distributionofhouseholdincome
income_data = pd.read_excel('UK_income_dist_2022.xlsx', sheet_name="Average disposable income", usecols="A:B", skiprows=2, nrows=170)
income_values = income_data['Income threshold']
income_values = np.insert(income_values.values, 0, 0)
# Define an x range with a fine resolution
x = np.linspace(0, income_values[-1], 10000)

def dmu_dt(mu):
    return delta_mu * mu

def dlambda_dt(lambda_,mu):
    return delta_lambda * (lambda_ - 1/mu)

def gamma_params(t, y):
    mu, lambda_ = y
    dmu = dmu_dt(mu)
    dlambda = dlambda_dt(lambda_, mu)
    return [dmu, dlambda] 

def cdf_truncated(pdf_dist):
    # Calculate the cumulative distribution function (CDF)
    cdf_values = np.cumsum(pdf_dist)
    cdf_values /= cdf_values[-1]  # Normalize to ensure the last value is 1
    return cdf_values

def gamma_percentile(percentile, alpha, lambda_):
    """
    Calculate the percentile of a gamma distribution
    """
    return gamma.ppf(percentile/100, a=alpha, scale=1/lambda_)

def lorenz_curve(pdf_dist, x):
    # CDF gives the cumulative share of population
    cdf_values = cdf_truncated(pdf_dist)
    # Calculate the (weighted) cumulative share of income
    cumulative_income = np.cumsum(pdf_dist * x)
    # Calculate the normalised cumulative share of income (between 0 and 1)
    cumulative_income /= cumulative_income[-1]  
    return cdf_values, cumulative_income

def gini_coefficient(pdf_dist, x):
    cdf_values, cumulative_income = lorenz_curve(pdf_dist, x)
    # Calculate the area under the Lorenz curve using the trapezoidal rule
    area_under_lorenz = np.trapz(cumulative_income, cdf_values)
    # The Gini coefficient is 1 - 2 * area under the Lorenz curve (A/(A+B) = 2A = 1 - 2B)
    gini = 1 - 2 * area_under_lorenz
    return gini

def palma_gamma_distribution(alpha, lambda_, x):
    """
    Calculate the Palma ratio for a gamma distribution in continuous form.
    That is, we seek an accurate value of the Palma ratio outside the discrete x values.
    """
    bottom_40_threshold = gamma_percentile(40, alpha, lambda_)
    top_10_threshold = gamma_percentile(90, alpha, lambda_)
    pdf_dist_bottom_40 = gamma.pdf(x[x <= bottom_40_threshold], a=alpha, scale=1/lambda_)
    pdf_dist_top_10 = gamma.pdf(x[x >= top_10_threshold], a=alpha, scale=1/lambda_)
    x_bottom_40 = x[x <= bottom_40_threshold]
    x_top_10 = x[x >= top_10_threshold]
    bottom_40_income = np.trapz(pdf_dist_bottom_40 * x_bottom_40, x_bottom_40)
    top_10_income = np.trapz(pdf_dist_top_10 * x_top_10, x_top_10)
    palma_ratio = top_10_income / bottom_40_income
    return palma_ratio


# Define parameters and initial conditions for the ODEs
delta_mu = 0.0
delta_lambda = 0.05
m_0 = 36203.15
lambda_0 = 7.786169e-05
num_years = 50
# Run the ODEs 
t_span = (0, num_years)
t_eval = np.linspace(0, num_years, int(num_years/0.1) + 1)
y0 = [m_0, lambda_0]
solution = solve_ivp(gamma_params, t_span, y0, t_eval=t_eval)
mu_values = solution.y[0]
lambda_values = solution.y[1]

palma_ratios = []
gini_coefficients = []
twentieth_percentile_incomes = []
# Gamma density function
for t in range(len(solution.t)):
    mu_t = mu_values[t]
    lambda_t = lambda_values[t]
    alpha_t = mu_t * lambda_t
    pdf_fitted = gamma.pdf(x, a=alpha_t, scale=1/lambda_t)
    cdf_values = cdf_truncated(pdf_fitted)
    palma_ratios.append(palma_gamma_distribution(alpha_t, lambda_t, x))
    gini_coefficients.append(gini_coefficient(pdf_fitted, x))
    twentieth_percentile_incomes.append(gamma_percentile(20, alpha_t, lambda_t))
    # store the pdf values for plotting later
    if t == 0:
        pdf_values = pdf_fitted
    else:
        pdf_values = np.vstack((pdf_values, pdf_fitted))


# Plot the gamma distribution at t = 0, 25, 50
plt.figure(figsize=(6,4))
plt.plot(x, income_data['Probability Density'], label='Empirical PDF', color='blue')
plt.plot(x, pdf_values[0], label='Fitted Gamma PDF at t=0', color='lightcoral', linestyle='--')
plt.plot(x, pdf_values[int(len(solution.t)/2)], label='Fitted Gamma PDF at t=25', color='red', linestyle='--')
plt.plot(x, pdf_values[-1], label='Fitted Gamma PDF at t=50', color='darkred', linestyle='--')
plt.xlabel('Equivalised Real Household Income (£)')
plt.ylabel('Income Density Function')
plt.title('Fitted Gamma Distribution at Different Time Points')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()  

# Plot the evolution of the parameters over time
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(solution.t, mu_values, color='blue')
plt.xlabel('Time (years)')
plt.ylabel(r'$\mu$ (Average real income)')
plt.title(r'Evolution of $\mu$ over time')
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(solution.t, lambda_values, color='red')
plt.xlabel('Time (years)')
plt.ylabel(r'$\lambda$ (redistribution factor)')
plt.title(r'Evolution of $\lambda$ over time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()  

# plot the evolution of the Palma ratio and Gini coefficient over time
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(solution.t, palma_ratios, color='green')
plt.xlabel('Time (years)')
plt.ylabel('Palma Ratio')
plt.title('Evolution of the Palma Ratio over time')
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(solution.t, gini_coefficients, color='orange')
plt.xlabel('Time (years)')
plt.ylabel('Gini Coefficient')
plt.title('Evolution of the Gini Coefficient over time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# plot the evolution of the 20th percentile income over time
plt.figure(figsize=(6,4))
plt.plot(solution.t, twentieth_percentile_incomes, color='purple')
plt.xlabel('Time (years)')
plt.ylabel('20th Percentile Income (£)')
plt.title('Evolution of the 20th Percentile Income over time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



########################################################
'''This section is for visualisation purposes only, to see how the fitted gamma distribution changes with different values of lambda (redistribution factor)'''
lambda_test = [1/m_0, 1.2/m_0, 1.5/m_0, 2.5/m_0]
color_test = ['lightcoral', 'red', 'darkred', 'darkblue']
plt.figure(figsize=(12, 8))
# vertical line for line of (relative) poverty threshold
plt.axvline(x=18000, color='black', linestyle='--', label='Poverty Threshold (£18,000)')
for lambda_, color in zip(lambda_test, color_test):
    alpha_test = m_0 * lambda_
    pdf_test = gamma.pdf(x, a=alpha_test, scale=1/lambda_)
    plt.plot(x, pdf_test, label=f'Gamma PDF with lambda={lambda_}', color=color)

plt.xlabel('Equivalised Real Household Income (£)')
plt.ylabel('Income Density Function')
plt.title('Fitted Gamma Distribution with Different Lambda Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# plot the cdf of the fitted gamma distribution with different lambda values
plt.figure(figsize=(12, 8))
plt.axvline(x=18000, color='black', linestyle='--', label='Poverty Threshold (£18,000)')
for lambda_, color in zip(lambda_test, color_test):
    alpha_test = m_0 * lambda_
    pdf_test = gamma.pdf(x, a=alpha_test, scale=1/lambda_)
    cdf_test = cdf_truncated(pdf_test)
    plt.plot(x, cdf_test, label=f'Gamma CDF with lambda={lambda_}', color=color)
plt.xlabel('Equivalised Real Household Income (£)')
plt.ylabel('Cumulative Distribution Function')
plt.title('Fitted Gamma CDF with Different Lambda Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#plot the Lorenz curve for the fitted gamma distribution with different lambda values
plt.figure(figsize=(12, 8))
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Line of Equality')
for lambda_, color in zip(lambda_test, color_test):
    alpha_test = m_0 * lambda_
    pdf_test = gamma.pdf(x, a=alpha_test, scale=1/lambda_)
    cdf_test, cumulative_income_test = lorenz_curve(pdf_test, x)
    plt.plot(cdf_test, cumulative_income_test, label=f'Lorenz Curve with lambda={lambda_}', color=color)
plt.xlabel('Cumulative Share of Population')
plt.ylabel('Cumulative Share of Income')
plt.title('Lorenz Curve for Fitted Gamma Distribution with Different Lambda Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()