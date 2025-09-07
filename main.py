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

# -----------------------------
# 1. Load Global Emissions Data
# -----------------------------
print("Loading GCB2022v27_MtCO2_flat.csv...")

# Read country-level CO₂ data
df = pd.read_csv("GCB2022v27_MtCO2_flat.csv")

# Group by year and sum across all countries
global_emissions = df.groupby("Year", as_index=False)["Total"].sum()

# Convert MtCO₂ → GtCO₂
global_emissions["global_gt"] = global_emissions["Total"] / 1000

# Keep only needed columns
global_emissions = global_emissions[["Year", "global_gt"]]
global_emissions.rename(columns={"Year": "year"}, inplace=True)

# Assign variables for modeling
years = global_emissions["year"].values.reshape(-1, 1)
emissions = global_emissions["global_gt"].values

print("Dataset ready! Years:", years.min(), "to", years.max())
print("Sample:\n", global_emissions.head())

# -----------------------------
# 2. Baseline Models
# -----------------------------
# Linear
linear_model = LinearRegression().fit(years, emissions)
pred_linear = linear_model.predict(years)

# Polynomial (Quadratic)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(years)
poly_model = LinearRegression().fit(X_poly, emissions)
pred_poly = poly_model.predict(X_poly)

# Exponential (normalize years to avoid overflow)
def exp_func(x, a, b): return np.exp(a + b*x)
years_norm = years.flatten() - years.min()
popt_exp, _ = curve_fit(exp_func, years_norm, emissions)
pred_exp = exp_func(years_norm, *popt_exp)

# Logistic
def logistic(x, K, r, x0): return K / (1 + np.exp(-r * (x - x0)))
popt_log, _ = curve_fit(logistic, years.flatten(), emissions, p0=[600, 0.03, 2050])
pred_logistic = logistic(years.flatten(), *popt_log)

# -----------------------------
# 3. AI Model (PyTorch NN)
# -----------------------------
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

for epoch in range(3000):
    optimizer.zero_grad()
    loss = criterion(net(X_tensor), y_tensor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_pred_nn = net(X_tensor).numpy()
pred_nn = scaler_y.inverse_transform(y_pred_nn)

# -----------------------------
# 4. Evaluation
# -----------------------------
def rmse(actual, pred): return np.sqrt(mean_squared_error(actual, pred))

results = {
    "Linear": rmse(emissions, pred_linear),
    "Polynomial": rmse(emissions, pred_poly),
    "Exponential": rmse(emissions, pred_exp),
    "Logistic": rmse(emissions, pred_logistic),
    "NeuralNet": rmse(emissions, pred_nn.flatten())
}

print("\nRMSE Comparison:")
for k,v in results.items():
    print(f"{k}: {v:.3f}")

# -----------------------------
# 5. Future Projections (to 2200)
# -----------------------------
future_years = np.arange(max(years)+1, 2201).reshape(-1,1)

proj_linear = linear_model.predict(future_years)
proj_poly = poly_model.predict(poly.transform(future_years))
proj_exp = exp_func(future_years.flatten() - years.min(), *popt_exp)
proj_logistic = logistic(future_years.flatten(), *popt_log)

with torch.no_grad():
    future_scaled = scaler_x.transform(future_years)
    proj_nn = net(torch.tensor(future_scaled, dtype=torch.float32)).numpy()
proj_nn = scaler_y.inverse_transform(proj_nn)

# -----------------------------
# 6. Threshold Detection
# -----------------------------
thresholds = [350, 400, 450]
threshold_hits = {}
for t in thresholds:
    years_hit = future_years.flatten()[np.where(proj_logistic >= t)[0]]
    threshold_hits[t] = int(years_hit[0]) if len(years_hit) > 0 else None

print("\nThreshold Crossing Years (Logistic Model):")
print(threshold_hits)

# -----------------------------
# 7. Data Center Emissions Scenario
# -----------------------------
dc_proj = pd.DataFrame({"year":[2022,2026,2030],"dc_percent":[0.03,0.06,0.12]})
dc_model = LinearRegression().fit(dc_proj[["year"]], dc_proj["dc_percent"])
dc_pred = dc_model.predict(future_years)

global_proj = proj_logistic
dc_gt = global_proj * dc_pred   # base scenario
dc_half_gt = global_proj * (dc_pred*0.5)  # slower growth

# -----------------------------
# 8. Plots
# -----------------------------
plt.figure(figsize=(12,7))
plt.scatter(years, emissions, color="black", label="Observed")
plt.plot(years, pred_linear, label="Linear")
plt.plot(years, pred_poly, label="Polynomial")
plt.plot(years, pred_exp, label="Exponential")
plt.plot(years, pred_logistic, label="Logistic")
plt.plot(years, pred_nn, label="Neural Net", linestyle="--")
plt.title("Global CO₂ Emissions Models")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (Gt/year)")
plt.legend()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(12,7))
plt.plot(future_years, global_proj, label="Global Emissions (Logistic)", color="black")
plt.plot(future_years, dc_gt, label="Data Centers", linestyle="dashed", color="red")
plt.plot(future_years, dc_half_gt, label="Data Centers (Half Growth)", linestyle="dotdash", color="blue")
plt.title("Global vs. Data Center Emissions Projection")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (Gt/year)")
plt.legend()
plt.savefig("dc_scenario.png", dpi=300)
plt.show()
