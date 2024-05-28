import pandas as pd
import numpy as np
from functions import *
import matplotlib.pyplot as plt

################## Simulation Part ##################
def simulate_VAR2(A1, A2, Sigma_u, T):
  """
  Simulates a VAR(2) process.
  
  Parameters:
  A1 (pd.DataFrame): Coefficient matrix for lag 1.
  A2 (pd.DataFrame): Coefficient matrix for lag 2.
  Sigma_u (pd.DataFrame): Covariance matrix for residuals.
  T (int): Number of time periods to simulate.
  
  Returns:
  np.ndarray: Simulated time series data starting from the 51st observation.
  """
  K = A1.shape[0]  # Number of variables in the VAR process
  y = np.zeros((T + 2 + 50, K))  # Matrix to store simulated observations

  # Generating random multivariate normal innovations
  U = np.random.multivariate_normal(mean=np.zeros(K), cov=Sigma_u, size=T + 2 + 50)

  # Simulating a VAR(2) process
  for t in range(2, T + 2 + 50):
      y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + U[t]

  return y[50:]  # Return simulated data starting from the 51st observation


# Define DataFrames representing coefficients and covariance matrix
A1 = pd.DataFrame([[0.4, 0.25], [0, 0.5]])  # Coefficients for lag 1
A2 = pd.DataFrame([[0.2, 0.4], [0, 0]])    # Coefficients for lag 2
Sigma_u = pd.DataFrame([[1, 0.5], [0.5, 1]])  # Covariance matrix for residuals

# Simulate VAR(2) process and estimate its coefficients and residuals
df = pd.DataFrame(simulate_VAR2(A1, A2, Sigma_u, T=100))  # Generate simulated VAR data
B, Sigma_u, T_ratio, Z, U = Estimate_VAR_Model(df.T, p=2, intercept=True)  # Estimate VAR model

# Compute impulse responses
thetas = Get_Impulse_Responses(B, Sigma_u, h=16)  # Compute impulse responses for 16 periods

# Plotting the orthogonalized IRFs in a 2x2 panel
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

for i in range(2):
  for j in range(2):
    irf_values = [x[i, j] for x in thetas]  # Extract values for the plot
    axes[i, j].plot(irf_values)
    axes[i, j].set_title(f'Variable {i+1} to shock {j+1}')

plt.tight_layout()
plt.show()




################## Real data part - quarterly Euro area time series ##################

# Load the CSV file and select specific columns (YER, ITR, LTN, STN)
df = pd.read_csv('data/awm19up18.csv')[['YER', 'ITR', 'LTN', 'STN']]
# Calculate the logarithmic differences for GDP growth rates, scale by 400
df['lgdp'] = np.log(df['YER']).diff() * 400
# Calculate the logarithmic differences for investment growth rates, scale by 400
df['invest'] = np.log(df['ITR']).diff() * 400
# Compute the differences for the long-term interest rate (R)
df['R'] = df['LTN'].diff()
# Compute the differences for the short-term interest rate (r)
df['r'] = df['STN'].diff()
# Remove the first row (NaN resulting from differences) and reset the index
df = df[['lgdp', 'invest', 'R','r']].iloc[1:].reset_index(drop=True)

# Check Granger causality
Check_Granger_Causality(df, [0, 0, 1, 1], p=4)

# Estimate VAR model
B, Sigma_u, T_ratio, Z, U = Estimate_VAR_Model(df.T, p=2, intercept=True)

# Compute impulse responses
thetas = Get_Impulse_Responses(B, Sigma_u, h=16)  # Compute impulse responses for 16 periods

# Plotting the orthogonalized IRFs in a 2x2 panel
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

for i in range(2):
  for j in range(2):
    irf_values = [x[i, j] for x in thetas]  # Extract values for the plot
    axes[i, j].plot(irf_values)
    axes[i, j].set_title(f'Variable {i+1} to shock {j+1}')

plt.tight_layout()
plt.show()