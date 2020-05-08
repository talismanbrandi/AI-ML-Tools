import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as mtick
import datetime as dt
import shap
date = dt.date.today

# Read the data
data_confirmed_raw = pd.read_csv('data/time_series_covid19_confirmed_global.csv').drop('Province/State', axis=1)
data_deaths_raw = pd.read_csv('data/time_series_covid19_deaths_global.csv').drop('Province/State', axis=1)
data_recovered_raw = pd.read_csv('data/time_series_covid19_recovered_global.csv').drop('Province/State', axis=1)

# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols=[]
    labels=[]
    for l in lst:
        if l >= 10:
            cols.append('#8b0100')
            labels.append('A')
        elif l >= 5:
            cols.append('#cc5100')
            labels.append('C')
        else:
            cols.append('#00888b')
            labels.append('B')
    return cols, labels


# Set up the plot
now = dt.datetime.now()
date = now.strftime(" (%d.%m.%Y)")

bbox_props = dict(boxstyle="round4,pad=0.5", fc="w", ec="#808080", lw=5, alpha=0.5)
plt.figure()
fig, ax = plt.subplots(figsize=(60, 30))
ax.loglog()
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlim(300, 800000)
plt.ylim(0.8, 120)
plt.ylabel(r'running mortality rate $ \left[\frac{deaths}{deaths + recovered}\right]$', fontsize=46, fontweight='bold')
plt.xlabel('total confirmed', fontsize=46, fontweight='bold')
plt.title('Running Mortality Rates for CovID-19' + date, fontsize=54, fontweight='bold', y=1.01)
plt.tick_params(labelsize=46)
plt.grid(linestyle='--', which='both', linewidth=1, alpha=0.65)


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
    mortality = []
    affected = []
    affected_countries = []
    confirmed = []
    k = 0
    for i, j in zip(deaths_current, recovered_current):
        if i + j >= 500:
            num = i + j
            affected.append(num)
            mortality.append(i/num*100.)
            affected_countries.append(countries[k])
            confirmed.append(confirmed_current[k])
        k += 1

    # The actual average global mortality rate
    ave = np.average(mortality, weights=affected)
    print(ave)

    colors, labels = pltcolor(mortality)
    plt.scatter(confirmed, mortality, zorder=99, color=colors, s=3500, alpha=0.6, marker='P')
    if annotate:
        for i, txt in enumerate(affected_countries):
            # plt.text(confirmed_current[i], mortality[i], txt[0], fontsize=14)
            ax.annotate(txt[0], xy=(confirmed[i], mortality[i]), xycoords='data', horizontalalignment='center',
                        verticalalignment='center', fontsize=36, fontweight='bold', zorder=100)
        for i, num in enumerate(mortality):
            # plt.text(confirmed_current[i], mortality[i], txt[0], fontsize=14)
            ax.annotate("{:.1f}%".format(num[0]), xy=(confirmed[i], num*np.exp(0.08)), xycoords='data', horizontalalignment='center',
                        verticalalignment='bottom', fontsize=26, fontweight='bold', zorder=100)
        ax.annotate("global average = {:.2f}%".format(ave), xy=(0.8, 0.1), xycoords='axes fraction', horizontalalignment='center',
                    verticalalignment='top', fontsize=36, fontweight='bold', bbox=bbox_props, zorder=100)
        ax.annotate(r"Only countries with (deaths + recovered) $\geq$ 500", xy=(0.8, 0.05), xycoords='axes fraction', horizontalalignment='center',
                    verticalalignment='top', fontsize=26, fontweight='normal', zorder=100)


today = ['4/18/20']

for i, date in enumerate(today):
    if i == 0: series(plt=plt, today=date, annotate=True)
    else: series(plt=plt, today=date, annotate=False)
plt.savefig('aa.png')
plt.show(block=False)

