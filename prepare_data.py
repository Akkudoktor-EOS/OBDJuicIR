import pandas as pd

# Lade die ursprüngliche Datei
file_path = '2024-05-08 16-00-40.csv'
data = pd.read_csv(file_path, delimiter=';')

# Entferne die überflüssige Spalte
data_clean = data.drop(columns=['Unnamed: 4'])

# Definiere die gewünschten PIDs
selected_pids = [
    'Battery voltage', 'Battery current', 'State of health', 'HV EV Battery Power', 
    'Cell temperatures max', 'Cell temperatures min', 'Battery State of Charge'
]

# Filtere die gewünschten PIDs
data_filtered = data_clean[data_clean['PID'].isin(selected_pids)]

# Wandle die Daten in ein breites Format um
data_wide = data_filtered.pivot_table(index='SECONDS', columns='PID', values='VALUE', aggfunc='first')

# Konvertiere den Index zurück in eine Spalte
data_wide.reset_index(inplace=True)

# Konvertiere die SECONDS-Spalte zu numerischen Werten
data_wide['SECONDS'] = pd.to_numeric(data_wide['SECONDS'], errors='coerce')

# Filtere die Daten ab Sekunde 58.500
data_wide_filtered = data_wide[data_wide['SECONDS'] >= 58500]

# Interpoliere die Daten
data_interpolated = data_wide_filtered.interpolate(method='linear', limit_direction='both')

# Berechne die mittlere Temperatur
data_interpolated['Average temperature'] = data_interpolated[['Cell temperatures max', 'Cell temperatures min']].mean(axis=1)

# Speichere die bereinigten Daten im breiten Format
output_path = 'test_data.csv'
data_interpolated.to_csv(output_path, index=False)

# Ausgabe des Speicherpfads (optional)
print(f'Die bereinigte CSV-Datei wurde gespeichert unter: {output_path}')
