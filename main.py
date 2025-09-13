import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation

print("Loading GCB2022v27_MtCO2_flat.csv...")
df = pd.read_csv("GCB2022v27_MtCO2_flat.csv")
global_emissions = df.groupby("Year", as_index=False)["Total"].sum()
global_emissions["global_gt"] = global_emissions["Total"] / 1000
global_emissions = global_emissions[["Year", "global_gt"]]
global_emissions.rename(columns={"Year": "year"}, inplace=True)
years = global_emissions["year"].values.reshape(-1, 1)
emissions = global_emissions["global_gt"].values

linear_model = LinearRegression().fit(years, emissions)
pred_linear = linear_model.predict(years)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(years)
poly_model = LinearRegression().fit(X_poly, emissions)
pred_poly = poly_model.predict(X_poly)

def exp_func(x, a, b): return np.exp(a + b*x)
years_norm = years.flatten() - years.min()
popt_exp, _ = curve_fit(exp_func, years_norm, emissions, maxfev=10000)
pred_exp = exp_func(years_norm, *popt_exp)

def logistic(x, K, r, x0): return K / (1 + np.exp(-r * (x - x0)))
popt_log, _ = curve_fit(logistic, years.flatten(), emissions, p0=[600, 0.03, 2050], maxfev=10000)
pred_logistic = logistic(years.flatten(), *popt_log)

scaler_x, scaler_y = StandardScaler(), StandardScaler()
X_scaled = scaler_x.fit_transform(years)
y_scaled = scaler_y.fit_transform(emissions.reshape(-1, 1))
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(1,64), nn.Linear(64,64), nn.Linear(64,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
for epoch in range(2000):
    optimizer.zero_grad()
    loss = criterion(net(X_tensor), y_tensor)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    y_pred_nn = net(X_tensor).numpy()
pred_nn = scaler_y.inverse_transform(y_pred_nn)

def rmse(actual, pred): return np.sqrt(mean_squared_error(actual, pred))
results = {
    "Linear": rmse(emissions, pred_linear),
    "Polynomial": rmse(emissions, pred_poly),
    "Exponential": rmse(emissions, pred_exp),
    "Logistic": rmse(emissions, pred_logistic),
    "NeuralNet": rmse(emissions, pred_nn.flatten())
}
print("\nRMSE Comparison:")
print(results)

future_years = np.arange(max(years)+1, 2201).reshape(-1,1)
proj_linear = linear_model.predict(future_years)
proj_poly = poly_model.predict(poly.transform(future_years))
proj_exp = exp_func(future_years.flatten() - years.min(), *popt_exp)
proj_logistic = logistic(future_years.flatten(), *popt_log)
with torch.no_grad():
    future_scaled = scaler_x.transform(future_years)
    proj_nn = net(torch.tensor(future_scaled, dtype=torch.float32)).numpy()
proj_nn = scaler_y.inverse_transform(proj_nn)

thresholds = [350, 400, 450]
threshold_hits = {}
for t in thresholds:
    years_hit = future_years.flatten()[np.where(proj_logistic >= t)[0]]
    threshold_hits[t] = int(years_hit[0]) if len(years_hit) > 0 else None
print("\nThreshold Crossing Years (Logistic Model):")
print(threshold_hits)

dc_proj = pd.DataFrame({"year":[2022,2026,2030],"dc_percent":[0.03,0.06,0.12]})
dc_model = LinearRegression().fit(dc_proj[["year"]], dc_proj["dc_percent"])
dc_pred = dc_model.predict(future_years)
global_proj = proj_logistic
dc_gt = global_proj * dc_pred
dc_half_gt = global_proj * (dc_pred*0.5)

plt.style.use("dark_background")
plt.figure(figsize=(12,7))
plt.scatter(years, emissions, color="white", label="Observed")
plt.plot(years, pred_linear, label="Linear")
plt.plot(years, pred_poly, label="Polynomial")
plt.plot(years, pred_exp, label="Exponential")
plt.plot(years, pred_logistic, label="Logistic")
plt.plot(years, pred_nn, label="Neural Net", linestyle="--")
plt.title("Global CO₂ Emissions Models")
plt.xlabel("Year"); plt.ylabel("CO₂ Emissions (Gt/year)")
plt.legend()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(12,7))
plt.plot(future_years, global_proj, label="Global Emissions (Logistic)", color="white")
plt.plot(future_years, dc_gt, label="Data Centers", linestyle="--", color="red")
plt.plot(future_years, dc_half_gt, label="Data Centers (Half Growth)", linestyle="-.", color="blue")
plt.title("Global vs. Data Center Emissions Projection")
plt.xlabel("Year"); plt.ylabel("CO₂ Emissions (Gt/year)")
plt.legend()
plt.savefig("dc_scenario.png", dpi=300)
plt.show()

print("Loading CAMX.csv...")
camx = pd.read_csv("CAMX.csv")
co2_daynight = camx["CO2 (lbs/MWh)"].iloc[2:].astype(float).values
hours = np.arange(1, len(co2_daynight)+1)

fig, ax = plt.subplots(figsize=(10,6))
line1, = ax.plot([], [], color="red", label="Original")
line2, = ax.plot([], [], color="cyan", label="Adjusted (Day/Night Savings)")
ax.set_xlim(0, 24)
ax.set_ylim(co2_daynight.min()*0.9, co2_daynight.max()*1.1)
ax.set_xlabel("Hour of Day"); ax.set_ylabel("CO₂ (lbs/MWh)")
ax.set_title("Day/Night CO₂ Emissions")
ax.legend()

def update(frame):
    xdata = hours[:frame]
    y1 = co2_daynight[:frame]
    y2 = y1 * 0.95
    line1.set_data(xdata, y1)
    line2.set_data(xdata, y2)
    return line1, line2

ani = FuncAnimation(fig, update, frames=len(hours), interval=100, blit=True)
ani.save("daynight_animation.gif", writer="pillow")
plt.show()

plt.figure(figsize=(12,7))
plt.plot(hours, co2_daynight, label="Original", color="red")
plt.plot(hours, co2_daynight*0.95, label="Adjusted (Day/Night Savings)", color="cyan")
plt.title("Day vs Night CO₂ Savings - California (CAMX)")
plt.xlabel("Hour of Day"); plt.ylabel("CO₂ (lbs/MWh)")
plt.legend()
plt.savefig("daynight_projection.png", dpi=300)
plt.show()



