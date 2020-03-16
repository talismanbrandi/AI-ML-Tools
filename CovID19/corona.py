import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import shap

# Read the data
data_confirmed_raw = pd.read_csv('data/time_series_19-covid-Confirmed.csv').drop('Province/State', axis=1)
data_deaths_raw = pd.read_csv('data/time_series_19-covid-Deaths.csv').drop('Province/State', axis=1)
data_recovered_raw = pd.read_csv('data/time_series_19-covid-recovered.csv').drop('Province/State', axis=1)

# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols=[]
    labels=[]
    for l in lst:
        if l >= 10:
            cols.append('#8b0100')
            labels.append('A')
        elif l >= 1 :
            cols.append('#cc5100')
            labels.append('B')
        else:
            cols.append('#00888b')
            labels.append('C')
    return cols, labels


# Set up the plot
bbox_props = dict(boxstyle="round,pad=2", alpha=0)
plt.figure()
fig, ax = plt.subplots(figsize=(15,10))
plt.xscale('log')
plt.yscale('log')
plt.xlim(2, 200000)
plt.ylim(1, 110)
plt.ylabel('mortality rate [%]', fontsize=16)
plt.xlabel('total confirmed', fontsize=16)
plt.title('Mortality Rates for CovID-19', fontsize=24)
plt.tick_params(labelsize=16)
plt.grid(linestyle='--', which='both', linewidth=0.5, alpha=0.65)


def series(plt, today, annotate=False):
    # Slect the current data
    data_confirmed = data_confirmed_raw[['Country/Region', today]].groupby('Country/Region', as_index=False).sum()
    data_deaths = data_deaths_raw[['Country/Region', today]].groupby('Country/Region', as_index=False).sum()
    data_recovered = data_recovered_raw[['Country/Region', today]].groupby('Country/Region', as_index=False).sum()

    countries = data_confirmed[['Country/Region']].values

    # Isolate data for plotting
    confirmed_current = data_confirmed[[today]].values
    deaths_current = data_deaths[[today]].values
    recovered_current = data_recovered[[today]].values

    # define mortality rates
    mortality = deaths_current/(deaths_current+recovered_current)*100.
    mortality = np.nan_to_num(mortality)
    affected = deaths_current+recovered_current

    # The actual average global mortality rate
    print(np.average(mortality, weights=affected))

    colors, labels = pltcolor(mortality)
    plt.scatter(confirmed_current, mortality, zorder=99, color=colors, s=60)
    if annotate:
        for i, txt in enumerate(countries):
            # plt.text(confirmed_current[i], mortality[i], txt[0], fontsize=14)
            ax.annotate(txt[0], xy=(confirmed_current[i], mortality[i]), xycoords='data', horizontalalignment='center',
                        verticalalignment='top', fontsize=14, bbox=bbox_props)


today = ['3/15/20']

for i, date in enumerate(today):
    if i == 0: series(plt=plt, today=date, annotate=True)
    else: series(plt=plt, today=date, annotate=False)
plt.savefig('aa.png')
plt.show(block=False)


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

# plt.figure()
# fig, ax = plt.subplots()
# plt.yscale('log')
# plt.ylabel('Number of regions')
# plt.xlabel('mortality rate [%]')
# plt.title('Mortality Rates distribution for CovID-19')
# plt.hist(mortality, bins=50, edgecolor='black', linewidth=1, zorder=99)
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# plt.grid(linestyle='--', which='both', axis='y', linewidth=0.5, alpha=0.65)
# plt.show()
