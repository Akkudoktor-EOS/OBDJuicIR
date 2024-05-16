import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
file_path = '2024-05-08_cleaned_battery_data_wide_soc.csv'
bms_data = pd.read_csv(file_path)

#bms_data = bms_data.sample(n=2000, random_state=42)

bms_data.sort_values(by="SECONDS",inplace=True)

# Filter the data to include only discharge cycles (negative power)
discharge_data = bms_data[bms_data['HV EV Battery Power'] < 0]
discharge_data['Time Interval'] = discharge_data['SECONDS'].diff().dropna() / 3600  # Convert seconds to hours
discharge_data['Energy Discharged (kWh)'] = discharge_data['HV EV Battery Power'] * discharge_data['Time Interval']
discharge_data['Cumulative Energy Old (kWh)'] = discharge_data['Energy Discharged (kWh)'].cumsum()

# Angenommen, dein DataFrame `discharge_data` enthält bereits die Spalten 'SECONDS' und 'HV EV Battery Power'
time_seconds = discharge_data['SECONDS']  # Zeitpunkte in Sekunden
power_in_kw = discharge_data['HV EV Battery Power']  # Leistung in kW

# Berechnen der kumulativen Energie mit der Trapezregel
# Da np.trapz die Integration über das gesamte Array berechnet, speichern wir die Zwischenergebnisse nach jedem Schritt
cumulative_energy_kwh = np.array([np.trapz(power_in_kw[:i+1], x=time_seconds[:i+1]) for i in range(len(time_seconds))]) / 3600

# Hinzufügen der berechneten kumulativen Energie zum DataFrame
discharge_data['Cumulative Energy (kWh)'] = cumulative_energy_kwh

print(discharge_data)

# Prepare data for linear regression
regression_data = discharge_data[['Cumulative Energy (kWh)', 'Battery State of Charge']].dropna()
X = regression_data[['Cumulative Energy (kWh)']]  # Features (cumulative discharged energy)
y = regression_data['Battery State of Charge']  # Target variable (SOC)

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict SOC at 0 kWh and full depletion using nominal full capacity
soc_at_zero_kwh = model.predict([[0]])
soc_at_full_depletion = model.predict([[-32.6]])  # Nominal full capacity

# Calculate the estimated full capacity based on the regression
slope = model.coef_[0]
intercept = model.intercept_
estimated_capacity_kwh = -intercept / slope  # Where SOC would reach zero

# Extend the kWh values to project down to SoC of 0
extended_kwh_values = np.linspace(X.min(), estimated_capacity_kwh, 300)
predicted_soc_extended = model.predict(extended_kwh_values)






# Plot the actual data and the extended regression line
plt.figure(figsize=(12, 7))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(extended_kwh_values, predicted_soc_extended, color='red', label='Extended Regression Line to SoC 0%')
plt.title('Extended Linear Regression of SoC vs. Cumulative Discharged Energy')
plt.xlabel('Cumulative Discharged Energy (kWh)')
plt.ylabel('State of Charge (%)')
plt.axhline(y=0, color='green', linestyle='--', label='SoC 0% Line')
plt.legend()
plt.grid(True)
plt.show()



# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct, WhiteKernel
# regression_data = discharge_data[['Cumulative Energy (kWh)', 'Battery State of Charge']].dropna()


# # Daten vorbereiten
# X = regression_data[['Cumulative Energy (kWh)']].values
# y = regression_data['Battery State of Charge'].values

# kernel = WhiteKernel(1.0, (1e-7, 1e3)) +  C(1.0)*DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2))  #RBF(length_scale=np.ones(1), length_scale_bounds=((1e-7, 1e3))) +
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True)
# print(X.shape)
# # Modell trainieren
# gpr.fit(X, y)

# # Vorhersagen und Unsicherheiten für den Bereich berechnen
# x_values = np.linspace(0, -50, 300).reshape(-1, 1)
# y_pred, sigma = gpr.predict(x_values, return_std=True)

# # Plot erstellen
# plt.figure(figsize=(12, 7))
# plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
# plt.plot(x_values, y_pred, color='red', label='GPR Regression Line')
# plt.fill_between(x_values.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, color='red', alpha=0.2, label='95% Confidence Interval')
# plt.title('Gaussian Process Regression of SoC vs. Cumulative Discharged Energy')
# plt.xlabel('Cumulative Discharged Energy (kWh)')
# plt.ylabel('State of Charge (%)')
# plt.legend()
# plt.grid(True)
# plt.show()




















from sklearn.metrics import mean_squared_error
from math import sqrt

# Berechnung der Residuen und der Standardabweichung
print(X.values)
y_pred = model.predict(X.values)
residuals = y - y_pred
std_dev = sqrt(mean_squared_error(y, y_pred))

# Konfidenzintervall um die Vorhersagen (95% Konfidenzintervall annehmen)
# Das 95% Konfidenzintervall wird oft mit 1.96 Standardabweichungen von der Vorhersage berechnet
ci_upper = predicted_soc_extended + 1.96 * std_dev
ci_lower = predicted_soc_extended - 1.96 * std_dev

print(ci_lower)
print(extended_kwh_values.reshape(-1))
# Plot die aktualisierten Daten und die Regressionslinie mit Konfidenzintervallen
plt.figure(figsize=(12, 7))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(extended_kwh_values, predicted_soc_extended, color='red', label='Regression Line to SoC 0%')
plt.fill_between(extended_kwh_values.reshape(-1), ci_lower, ci_upper, color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('Extended Linear Regression of SoC vs. Cumulative Discharged Energy with Confidence Interval')
plt.xlabel('Cumulative Discharged Energy (kWh)')
plt.ylabel('State of Charge (%)')
plt.axhline(y=0, color='green', linestyle='--', label='SoC 0% Line')
plt.legend()
plt.grid(True)
plt.show()
