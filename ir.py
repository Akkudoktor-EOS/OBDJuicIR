"""
Copyright (c) 2024 Dr. Andreas Schmitz - info@akkudoktor.net - akkudoktor.net
Alle Rechte vorbehalten.

Die Erlaubnis wird hiermit kostenlos erteilt, eine Kopie dieser Software und der zugehörigen 
Dokumentationsdateien (die "Software") zu verwenden, zu kopieren, zu verändern, zusammenzuführen, 
zu veröffentlichen, zu vertreiben und Unterlizenzierungen der Software zu gestatten, und Personen, 
denen die Software zur Verfügung gestellt wird, diese zu gestatten, unter den folgenden Bedingungen:

DIE KOMMERZIELLE NUTZUNG DIESER SOFTWARE IST NUR MIT AUSDRÜCKLICHER SCHRIFTLICHER GENEHMIGUNG DES COPYRIGHT-INHABERS ERLAUBT.

Die obige Urheberrechtsanzeige und diese Genehmigungsanzeige sind in allen Kopien oder wesentlichen 
Teilen der Software beizubehalten.

DIE SOFTWARE WIRD OHNE JEGLICHE AUSDRÜCKLICHE ODER IMPLIZIERTE GARANTIE BEREITGESTELLT, EINSCHLIESSLICH 
DER GARANTIE DER MARKTFÄHIGKEIT, DER EIGNUNG FÜR EINEN BESTIMMTEN ZWECK UND DER NICHTVERLETZUNG. IN 
KEINEM FALL SIND DIE AUTOREN ODER URHEBERRECHTSINHABER FÜR JEGLICHE ANSPRÜCHE, SCHÄDEN ODER SONSTIGE 
HAFTUNG VERANTWORTLICH, OB IN EINER VERTRAGSKLAGE, EINER UNERLAUBTEN HANDLUNG ODER ANDERWEITIG, DIE 
SICH AUS DER SOFTWARE ODER DER NUTZUNG ODER ANDEREN GESCHÄFTEN IN DER SOFTWARE ERGEBEN.

Dieses Projekt verwendet die folgenden Bibliotheken:

1. NumPy (Lizenz: BSD License)
   https://numpy.org/license.html

2. pandas (Lizenz: BSD License)
   https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html#license

3. seaborn (Lizenz: BSD License)
   https://seaborn.pydata.org/

4. Matplotlib (Lizenz: PSF License)
   https://matplotlib.org/stable/users/license.html
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import seaborn as sns
import argparse

# Argumentparser für Shell-Parameter
parser = argparse.ArgumentParser(description='Process battery data from Car Scanner and calculate internal resistance statistics.')
parser.add_argument('--data_path', type=str, required=True, help='CSV Dateiname')
parser.add_argument('--threshold', type=float, required=True, help='Nominaler/gesunder Akku Widerstand in mOhm')
parser.add_argument('--cable_bms_ir', type=float, default=0.2, help='Kabel und BMS Widerstand (default: 0.2 mOhm)')
parser.add_argument('--cells', type=float, default=96, required=True, help='Anzahl Zellen des Akkus')

args = parser.parse_args()

# Daten laden
data_path = args.data_path
df = pd.read_csv(data_path)

# IR < als für die Wahrscheinlichkeitsberechnung
threshold = args.threshold
cable_bms_ir = args.cable_bms_ir



# Daten laden
#data_path = '2024-05-08_cleaned_battery_data_wide_soc.csv'  # Pfad zur CSV-Datei anpassen
df = pd.read_csv(data_path)


# Wahrscheinlichkeit
prob_threshold = 0.9

# Nominaler innerer Widerstand einer neuen Batterie
nominal_ir = threshold*1.1


threshold = threshold 

# Filtern von Datenpunkten mit negativem Strom
resting_current_threshold = 0.4
resting_data = df[np.abs(df['Battery current']) <= resting_current_threshold]
discharging_resting_data = resting_data[resting_data['Battery current'] <= 0]



# Interpolation der Ruhespannung mit Savitzky-Golay-Filter
soc_vs_resting_voltage = resting_data.groupby('Battery State of Charge')['Battery voltage'].mean()
soc_vs_resting_voltage_smoothed = savgol_filter(soc_vs_resting_voltage, window_length=11, polyorder=2)
soc_vs_resting_voltage_interp = np.interp(df['Battery State of Charge'], soc_vs_resting_voltage.index, soc_vs_resting_voltage_smoothed)





# # Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Zeit (Sekunden)')
ax1.set_ylabel('Spannung (V)', color=color)
ax1.plot(df['SECONDS'], df['Battery voltage'], color=color, label='Spannung')

ax1.plot(df['SECONDS'], soc_vs_resting_voltage_interp, color="red", label='Interpolierte Ruhespannung')


#ax1.scatter(discharging_resting_data['SECONDS'], discharging_resting_data['Battery voltage'], color='red', label='Verwendete Ruhespannungspunkte', alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')
ax1.grid(True)

# Zweite Achse für Batteriestrom
# ax2 = ax1.twinx()
# color = 'tab:orange'
# ax2.set_ylabel('Batteriestrom (A)', color=color)
# ax2.scatter(discharging_resting_data['SECONDS'], discharging_resting_data['Battery current'], color=color, label='Zuordnungsströme', alpha=0.7)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.legend(loc='upper right')

fig.tight_layout()
plt.title('Interpolierte Ruhespannung und zugehörige Ströme (nur Entladung) über der Zeit')
plt.show()






# Innenwiderstand Berechnung mit verbesserter Ruhespannung
load_data = df[abs(df['Battery current']) > 40]  # Betrachtung von Entladedaten
load_data['Resting voltage'] = np.interp(load_data['Battery State of Charge'], soc_vs_resting_voltage.index, soc_vs_resting_voltage_smoothed)
load_data['Internal resistance'] = (load_data['Resting voltage'] - load_data['Battery voltage']) / load_data['Battery current']
load_data['Internal resistance (mOhm)'] = np.abs(load_data['Internal resistance']) / args.cells * 1000  # Umrechnung in mOhm und Normalisierung auf die Zellenzahl
load_data['Internal resistance (mOhm)'] -=cable_bms_ir


# Gruppieren nach Entladerate
load_data['Discharge rate'] = np.abs(load_data['Battery current'])
discharge_rate_bins = pd.cut(load_data['Discharge rate'], bins=10)
load_data['Discharge rate bin'] = discharge_rate_bins

# Wahrscheinlichkeit, dass der Widerstand kleiner als 0.92 mOhm ist, für jede Entladerate
probabilities = load_data.groupby('Discharge rate bin')['Internal resistance (mOhm)'].apply(lambda x: np.sum(x < threshold) / len(x))

# Plot der inneren Widerstände über der Zeit
plt.figure(figsize=(12, 8))
plt.plot(load_data['SECONDS'], load_data['Internal resistance (mOhm)'], marker='o', linestyle='-', alpha=0.6)
plt.xlabel('Time (s)')
plt.ylabel('Internal resistance (mOhm)')
plt.title('Internal Resistance over Time')
plt.grid(True)
plt.show()




# Berechnung des Prozentsatzes der Widerstände über dem nominalen Wert für verschiedene Entladeraten
exceeding_percentages = {}

for name, group in load_data.groupby('Discharge rate bin'):
    sorted_resistances = np.sort(group['Internal resistance (mOhm)'])
    threshold_index = int(len(sorted_resistances) * prob_threshold)
    threshold_value = sorted_resistances[threshold_index]
    exceeding_percentage = np.sum(sorted_resistances > nominal_ir) / len(sorted_resistances) * 100
    exceeding_percentages[name] = exceeding_percentage

# Violinplot erstellen
plt.figure(figsize=(12, 8))
sns.violinplot(x='Discharge rate bin', y='Internal resistance (mOhm)', data=load_data, inner='quartile')

# Wahrscheinlichkeiten als Text hinzufügen
for i, (bin_name, percentage) in enumerate(exceeding_percentages.items()):
    plt.text(i, 0.95 * plt.ylim()[1], f'{percentage:.2f}%', ha='center', color='white', fontsize=12, 
             bbox=dict(facecolor='black', alpha=0.5))

plt.xticks(rotation=20, fontsize=14)
plt.yticks(fontsize=14)


plt.xlabel('Discharge rate (A)')
plt.ylabel('Internal resistance (mOhm)')
plt.title('Internal Resistances pro Entladestrom \n Wahrscheinlichkeit gemessener IR > 10% Anstieg')
plt.grid(True)
plt.show()












# Violinplot erstellen
plt.figure(figsize=(12, 8))
sns.violinplot(x='Discharge rate bin', y='Internal resistance (mOhm)', data=load_data, inner='quartile')

# Wahrscheinlichkeiten als Text hinzufügen
for i, bin_name in enumerate(probabilities.index.categories):
    prob = probabilities[bin_name] * 100  # Convert to percentage
    plt.text(i, 0.95 * plt.ylim()[1], f'{prob:.2f}%', ha='center', color='white', fontsize=12, 
             bbox=dict(facecolor='black', alpha=0.5))

plt.xticks(rotation=45)
plt.xlabel('Discharge rate (A)')
plt.ylabel('Internal resistance (mOhm)')
plt.title('Violin plot of Internal Resistances by Discharge Rate\n with Probabilities < '+str(threshold)+' mOhm')
plt.grid(True)
plt.show()


# Berechnung der Varianz der internen Widerstände für verschiedene Entladeraten
variance_by_current = load_data.groupby('Discharge rate')['Internal resistance (mOhm)'].var()

plt.figure(figsize=(12, 8))
plt.plot(variance_by_current.index, variance_by_current, marker='o', linestyle='-', color='b')
plt.xlabel('Discharge rate (A)')
plt.ylabel('Variance of Internal resistance (mOhm^2)')
plt.title('Variance of Internal Resistance vs. Discharge Rate')
plt.grid(True)
plt.show()



print(f"Die Wahrscheinlichkeit, dass der Innenwiderstand kleiner als {threshold} mOhm ist:")
print(probabilities)
