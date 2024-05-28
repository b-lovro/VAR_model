import pandas as pd
import numpy as np
from scipy import stats
import math
import scipy

NA=np.array(np.nan)

def Estimate_VAR_Model(df,p=2,intercept=True):


    # Get the number of columns (T) and rows (K+p) in the DataFrame 'df'.
    T = len(df.columns)
    K = len(df.index)


    ######   Get matrix Z   ####### 
    # Initialize an empty list called 'matrix' to store the calculated values of Z.
    matrix = []
    # Loop through columns starting from the 'p'th column to the last column.
    for i in range(p, T):
        # Create an empty list 'temp_1' for the current column 'i'.
        if intercept:
            temp_1 = [1]  # Include an intercept term (1).
        else:
            temp_1 = []  # No intercept term.

        # Loop through rows from 0 to 'K'.
        for k in range(0, K):
            # Create an empty list 'temp_2' for the current row 'k'.
            temp_2 = []

            # Loop through the 'p' lag values for the current column.
            for j in range(1, p + 1):

                # Append the lagged values from the DataFrame to 'temp_2'.
                temp_2.append(df.iloc[k, i - j])

            # Concatenate 'temp_2' to 'temp_1'.
            temp_1 = temp_1 + temp_2

        # Append 'temp_1' to the 'matrix'.
        matrix.append(temp_1)

    # Create a DataFrame 'Z' from the 'matrix' and transpose it.
    Z = pd.DataFrame(matrix)
    Z = Z.T

    ######   Get matrix Y   ####### 


    Y = df.T
    # Drop rows indexed from 0 to 'p' (exclusive) to remove unnecessary rows
    Y = Y.drop(np.arange(0, p, 1))
    Y = Y.T

    ######   Get matrix B ... YZ'*(ZZ')^-1  (sum1)*(sum2) ####### 


    # Calculate the dot product of Y and the transpose of matrix Z
    sum1 = np.dot(Y, Z.T)
    # Calculate the dot product of matrix Z and the transpose of Z
    sum2 = np.dot(Z, Z.T)
    # Calculate matrix B as the result of multiplying 'sum1' by the inverse of 'sum2'
    B = np.dot(sum1, np.linalg.inv(sum2))
    # Convert the result 'B' to a DataFrame
    B = pd.DataFrame(B)


    ########## Estimate of the residual covariance matrix (sigma_u) #############


    # Calculate the residuals by subtracting the product of 'B' and 'Z' from 'Y'
    U = Y - np.dot(B, Z)
    # Calculate the estimate of the residual covariance matrix 'sigma_u'
    sigma_u = (1 / (T - p - K * p - 1)) * np.dot(U, U.T)
    # Convert 'sigma_u' to a DataFrame
    Sigma_u = pd.DataFrame(sigma_u)


    ########## T- rations ##############
    # Calculate the Kronecker product of the inverse of the dot product of 'Z' and its transpose
    # and the 'sigma_u' matrix
    t = np.kron(np.linalg.inv(np.dot(Z, Z.T)), sigma_u)
    # Convert the 'B' DataFrame to a NumPy array and flatten it in column-major order
    B_vec = B.to_numpy().flatten('F')
    # Calculate the diagonal of the 't' matrix
    t_diag = np.diag(t)

    # Initialize an empty list to store T-ratios
    t_rati_list = []
    # Calculate the T-ratios for each element in 'B_vec' and append them to the list
    for i in range(len(B_vec)):
        t_rati_list.append((B_vec[i]) / (math.sqrt(t_diag[i])))

    # Convert the list of T-ratios to a NumPy array and reshape it to a Kx7 matrix
    t_rati_list = np.array(t_rati_list)
    t_rati = t_rati_list.reshape(K, len(B.columns))
    # Convert the result to a DataFrame
    T_ratio = pd.DataFrame(t_rati)
    return B,Sigma_u,T_ratio,Z,U


def Check_Granger_Causality(df,dummy_vector,p=4):
  """
    Checks for Granger causality in a VAR model.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    dummy_vector (list): List indicating the variables to test for Granger causality.
    p (int): Number of lags to include in the VAR model (default is 4).
    
    Returns:
    tuple: Contains the following elements:
        - wald_statistic (float): Wald statistic value.
        - wald_p (float): P-value for the Wald statistic.
        - f_statistic (float): F statistic value.
        - f_p (float): P-value for the F statistic.
  """
    
  B,Sigma_u,T_ratio,Z,U = Estimate_VAR_Model(df.T,p=p,intercept=True)

  vec_B=B.to_numpy().flatten('F')
  K=len(dummy_vector)
  vec_ones=[0]*K
  dummy_reversed = [0 if i == 1 else 1 for i in dummy_vector]

  for i in dummy_vector:
      if i == 0:
          vec_ones.extend([0] * (p * K))
      else:
          vec_ones.extend(dummy_reversed * p)
      
  temp = []
  zeros_list = [0] * len(vec_ones)
  for i, val in enumerate(vec_ones):
      if val == 1:
          zeros_list[i] = 1
          temp.append(zeros_list.copy())
          zeros_list[i] = 0
  
  C = pd.DataFrame(temp) 

  ############ getting wald and f stat ###########

  ZZ_t_inverse=np.linalg.inv(Z @ Z.T)
  wald_statistic = ((C @ vec_B).T) @ (np.linalg.inv((C @ (np.kron(ZZ_t_inverse,Sigma_u)) @ C.T))) @ (C @ vec_B)
  f_statistic=wald_statistic/(C.shape[0])

  wald_p=1 - stats.chi2.cdf(wald_statistic, C.shape[0])
  f_p=1 - stats.f.cdf(f_statistic, C.shape[0], df.shape[0]*K-K**2*p-K)


  print('\nWald statistic:')
  print(wald_statistic)
  print('Wald statistic p-value:')
  print(wald_p)

  print('\nF statistic:')
  print(f_statistic)
  print('F statistic p-value:')
  print(f_p)

  return wald_statistic, wald_p, f_statistic, f_p

def Get_Impulse_Responses(B, sigma_u, h=16):
  """
  Computes impulse response functions for a VAR model.
  
  Parameters:
  B (pd.DataFrame): Estimated coefficients of the VAR model.
  sigma_u (pd.DataFrame): Estimated residual covariance matrix.
  h (int): Number of periods to compute the impulse responses for (default is 16).
  
  Returns:
  list: A list of impulse response matrices for each period up to 'h'.
  """
  # Calculate the number of lags based on matrix dimensions
  p = int(((B.shape[1] - 1) / sigma_u.shape[1]))
  
  # Extract VAR coefficients excluding intercept
  df = B.iloc[:, 1:]
  
  # Cholesky decomposition of the residual covariance matrix
  P = scipy.linalg.cholesky(sigma_u, lower=True)
  
  # Get the number of equations in the VAR model
  K = B.shape[0]
  
  # Store coefficient matrices for each lag
  all_a = []
  for i in range(p):
      all_a.append(df.iloc[:, i::p].to_numpy())
  
  # Initialize Phi_0 as identity matrix
  Phis = [np.eye(K)]
  
  # Compute Phi_i for each period up to 'h'
  for i in range(1, h+1):
      # Calculate Phi_i using VAR coefficients and lagged matrices
      Phi_i = sum(Phis[i - j] @ all_a[j - 1] for j in range(1, min(i, p) + 1))
      Phis.append(Phi_i)
  
  # Compute impulse responses for each period by multiplying Phi_i with the Cholesky decomposition
  thetas = []
  for i in range(0, h, 1):
      thetas.append(Phis[i] @ P)
  
  # Return computed impulse response matrices and coefficient matrices
  return thetas