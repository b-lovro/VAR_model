# VAR Model Analysis

This repository contains Python functions for estimating and analyzing Vector Autoregression (VAR) models. VAR is a statistical model used to capture the linear interdependencies among multiple time series. Each variable in a VAR model is a linear function of past values of itself and past values of all other variables in the model.

## Available Functions

### `simulate_VAR2`

**Description**: Simulates a VAR(2) process, which is a VAR model with two lags.

**Parameters**:
- `A1` (pd.DataFrame): Coefficient matrix for lag 1.
- `A2` (pd.DataFrame): Coefficient matrix for lag 2.
- `Sigma_u` (pd.DataFrame): Covariance matrix for residuals.
- `T` (int): Number of time periods to simulate.

**Returns**: `np.ndarray` - Simulated time series data starting from the 51st observation.

### `Estimate_VAR_Model`

**Description**: Estimates a VAR model given a DataFrame of time series data.

**Parameters**:
- `df` (pd.DataFrame): DataFrame containing the time series data.
- `p` (int): Number of lags to include in the model (default is 2).
- `intercept` (bool): Whether to include an intercept term in the model (default is True).

**Returns**: `tuple` - Contains the estimated coefficients (`B`), residual covariance matrix (`Sigma_u`), T-ratios (`T_ratio`), matrix of lagged values (`Z`), and matrix of residuals (`U`).

### `Check_Granger_Causality`

**Description**: Checks for Granger causality in a VAR model.

**Parameters**:
- `df` (pd.DataFrame): DataFrame containing the time series data.
- `dummy_vector` (list): List indicating the variables to test for Granger causality.
- `p` (int): Number of lags to include in the VAR model (default is 4).

**Returns**: `tuple` - Contains the Wald statistic value, p-value for the Wald statistic, F statistic value, and p-value for the F statistic.

### `Get_Impulse_Responses`

**Description**: Computes impulse response functions for a VAR model.

**Parameters**:
- `B` (pd.DataFrame): Estimated coefficients of the VAR model.
- `sigma_u` (pd.DataFrame): Estimated residual covariance matrix.
- `h` (int): Number of periods to compute the impulse responses for (default is 16).

**Returns**: `list` - A list of impulse response matrices for each period up to `h`.

