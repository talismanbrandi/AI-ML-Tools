import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, FuncFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as mtick
import datetime as dt
from matplotlib.lines import Line2D
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
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.yaxis.set_minor_formatter(mtick.PercentFormatter())
plt.xlim(900, 1100000)
plt.ylim(0.91, 30)
plt.ylabel(r'Case Fatality Rate $ \left[\frac{deaths(t)}{confirmed (t-10)}\right]$', fontsize=44, fontweight='bold')
plt.xlabel('total confirmed', fontsize=44, fontweight='bold')
plt.title('Case Fatality Rates for CovID-19' + date, fontsize=54, fontweight='bold', y=1.01)
plt.tick_params(labelsize=46, which='both')
plt.grid(linestyle='--', which='both', linewidth=1, alpha=0.65)


def series(plt, today, reference, annotate=False):
    # Select the current data
    data_confirmed = data_confirmed_raw[['Country/Region', today]].groupby('Country/Region', as_index=False).sum()
    data_confirmed_reference = data_confirmed_raw[['Country/Region', reference]].groupby('Country/Region', as_index=False).sum()
    data_deaths = data_deaths_raw[['Country/Region', today]].groupby('Country/Region', as_index=False).sum()
    data_recovered = data_recovered_raw[['Country/Region', today]].groupby('Country/Region', as_index=False).sum()

    countries = data_confirmed[['Country/Region']].values

    # Isolate data for plotting
    confirmed_current = data_confirmed[[today]].values
    confirmed_reference = data_confirmed_reference[[reference]].values
    deaths_current = data_deaths[[today]].values
    recovered_current = data_recovered[[today]].values

    # define mortality rates
    mortality = []
    affected = []
    affected_countries = []
    confirmed = []
    mortality_reference = []
    k = 0
    for i, j in zip(deaths_current, recovered_current):
        if i + j >= 1000:
            num = i + j
            affected.append(num)
            mortality.append(i/num*100.)
            mortality_reference.append(i/confirmed_reference[k]*100.)
            affected_countries.append(countries[k])
            confirmed.append(confirmed_current[k])
        k += 1

    # The actual average global mortality rate
    ave = np.average(mortality, weights=affected)
    ave_reference = np.average(mortality_reference, weights=affected)
    print(ave, ave_reference)

    colors, labels = pltcolor(mortality_reference)
    plt.scatter(confirmed, mortality_reference, zorder=99, color=colors, s=3500, alpha=0.6, marker='P')
    if annotate:
        for i, txt in enumerate(affected_countries):
            # plt.text(confirmed_current[i], mortality[i], txt[0], fontsize=14)
            ax.annotate(txt[0], xy=(confirmed[i], mortality_reference[i]), xycoords='data', horizontalalignment='center',
                        verticalalignment='center', fontsize=36, fontweight='bold', zorder=100)
        for i, num in enumerate(mortality_reference):
            # plt.text(confirmed_current[i], mortality[i], txt[0], fontsize=14)
            ax.annotate("{:.1f}%".format(num[0]), xy=(confirmed[i], num*np.exp(0.08)), xycoords='data', horizontalalignment='center',
                        verticalalignment='bottom', fontsize=26, fontweight='bold', zorder=100)
        ax.annotate("global average = {:.2f}%".format(ave_reference), xy=(0.8, 0.1), xycoords='axes fraction', horizontalalignment='center',
                    verticalalignment='top', fontsize=46, fontweight='bold', bbox=bbox_props, zorder=100)
        ax.annotate(r"Only countries with (deaths + recovered) $\geq$ 1000", xy=(0.8, 0.05), xycoords='axes fraction', horizontalalignment='center',
                    verticalalignment='top', fontsize=26, fontweight='normal', zorder=100)
    redcross = Line2D([0], [0], color='#8b0100', linewidth=1.5, linestyle='', label=r'$\bf \geq 10\%$', marker='+',
                     markersize=50, markeredgewidth=15, alpha=0.6)
    orangecross = Line2D([0], [0], color='#cc5100', linewidth=1.5, linestyle='', label=r'$\bf \geq 5\%$', marker='+',
                        markersize=50, markeredgewidth=15, alpha=0.6)
    greencross = Line2D([0], [0], color='#00888b', linewidth=1.5, linestyle='', label=r'$\bf < 5\%$', marker='+',
                        markersize=50, markeredgewidth=15, alpha=0.6)
    ax.legend(handles=[redcross, orangecross, greencross], loc='upper right', ncol=1, fontsize=46)


today = ['4/28/20']
reference = '4/18/20'

for i, date in enumerate(today):
    if i == 0: series(plt=plt, today=date, reference=reference, annotate=True)
    else: series(plt=plt, today=date, reference=reference, annotate=False)
plt.savefig('aa.png')
plt.show(block=False)

