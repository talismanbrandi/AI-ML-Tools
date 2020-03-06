import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import shap

today = '3/4/20'

# Read the data
data_confirmed_raw = pd.read_csv('data/time_series_19-covid-Confirmed.csv')
data_deaths_raw = pd.read_csv('data/time_series_19-covid-Deaths.csv')
data_recovered_raw = pd.read_csv('data/time_series_19-covid-recovered.csv')

# Slect the current data
data_confirmed = data_confirmed_raw[['Country/Region', today]]
data_deaths = data_deaths_raw[['Country/Region', today]]
data_recovered = data_recovered_raw[['Country/Region', today]]

# Isolate data for plotting
countries = data_confirmed[['Country/Region']].values
confirmed_current = data_confirmed[[today]].values
deaths_current = data_deaths[[today]].values
recovered_current = data_recovered[[today]].values

# define mortality rates
mortality = deaths_current/(deaths_current+recovered_current)*100.
mortality = np.nan_to_num(mortality)
affected = deaths_current+recovered_current

# The actual average global mortality rate
print(np.average(mortality, weights=affected))

# plot some figures
# plt.figure()
# plt.xscale('log')
# plt.xlim(0.9, 40000)
# plt.ylabel('mortality rate [%]')
# plt.xlabel('total deaths + total recovered')
# plt.title('Mortality Rates for CovID-19')
# plt.scatter(affected, mortality, zorder=99)
# plt.grid(linestyle='--')
# plt.show()

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.9, 90000)
plt.ylim(0.08, 110)
plt.ylabel('mortality rate [%]')
plt.xlabel('total confirmed')
plt.title('Mortality Rates for CovID-19')
plt.scatter(confirmed_current, mortality, zorder=99)
plt.grid(linestyle='--', which='both', linewidth=0.5, alpha=0.65)
plt.show()

plt.figure()
fig, ax = plt.subplots()
plt.yscale('log')
plt.ylabel('Number of regions')
plt.xlabel('mortality rate [%]')
plt.title('Mortality Rates distribution for CovID-19')
plt.hist(mortality, bins=50, edgecolor='black', linewidth=1, zorder=99)
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.grid(linestyle='--', which='both', axis='y', linewidth=0.5, alpha=0.65)
plt.show()
