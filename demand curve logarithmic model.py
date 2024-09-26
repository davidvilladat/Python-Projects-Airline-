# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:46:51 2024

@author: crist
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from scipy.optimize import minimize
import pandas as pd


#data sample from GRULAD POS AO 
data = {
    "Año y mes": [
        "ene 2023", "feb 2023", "mar 2023", "abr 2023", "may 2023", "jun 2023", "jul 2023", 
        "ago 2023", "sept 2023", "oct 2023", "nov 2023", "dic 2023", "ene 2024", 
        "feb 2024", "mar 2024", "abr 2024", "may 2024", "jun 2024", "jul 2024", "ago 2024"
    ],
    "PAX": [
        2107, 1829, 1788, 1597, 1701, 1641, 1469, 1881, 2026, 2186, 2404, 1817, 
        2079, 2332, 3691, 2513, 2720, 2421, 3019, 3076
    ],
    "Avg Fare": [
        795.5028524, 786.8948442, 896.0003915, 915.2512085, 1007.818389, 1098.670768, 
        872.9608475, 825.3341206, 768.7544689, 807.2493073, 831.5379311, 671.2448545, 
        696.2084116, 730.2622329, 570.694745, 635.882265, 712.1753952, 689.9307539, 
        726.0415518, 608.224423
    ]
}



# Convert to DataFrame
df = pd.DataFrame(data)

# IQR method to remove outliers
Q1_fare, Q3_fare = df['Avg Fare'].quantile([0.25, 0.75])
IQR_fare = Q3_fare - Q1_fare

Q1_pax, Q3_pax = df['PAX'].quantile([0.25, 0.75])
IQR_pax = Q3_pax - Q1_pax

# Filter the data
df_filtered = df[~((df['Avg Fare'] < (Q1_fare - 1.5 * IQR_fare)) | (df['Avg Fare'] > (Q3_fare + 1.5 * IQR_fare)))]
df_filtered = df_filtered[~((df_filtered['PAX'] < (Q1_pax - 1.5 * IQR_pax)) | (df_filtered['PAX'] > (Q3_pax + 1.5 * IQR_pax)))]






# Prepare for regression
X = np.array(df_filtered['Avg Fare']).reshape(-1, 1)
y = np.array(df_filtered['PAX'])





# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Apply logarithmic transformation on the Avg Fare (log model)
X_log = FunctionTransformer(np.log, validate=True).transform(X)
model_log = LinearRegression().fit(X_log, y)

# Coefficients for log-linear equation
a, b = model_log.coef_[0], model_log.intercept_

# Define the log-linear demand and revenue functions
def demand_function(fare):
    return max(0, a * np.log(fare) + b) if fare > 0 else 0

def revenue_function(fare):
    return fare * demand_function(fare)

# Finding the optimal fare that maximizes revenue
result = minimize(lambda fare: -revenue_function(fare), x0=[100], bounds=[(1, None)])  
optimal_fare = result.x[0]
optimal_revenue = revenue_function(optimal_fare)

# Generate fare values and corresponding revenues for plotting
fare_values = np.linspace(0, 2000,50)
revenues = [revenue_function(fare) for fare in fare_values]



# Scatter plot of the filtered data with outliers removed
plt.figure(figsize=(10,6))
plt.scatter(df_filtered['Avg Fare'], df_filtered['PAX'], color='blue', label='Filtered Data (No Outliers)')
plt.plot(df_filtered['Avg Fare'], model.predict(np.array(df_filtered['Avg Fare']).reshape(-1, 1)), 
         color='red', label='Fitted Line')
plt.xlabel('Avg Fare (USD)')
plt.ylabel('PAX (Demand)')
plt.title('Demand Curve after Outlier Removal')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Revenue vs Fare with dotted line at the maximum
plt.figure(figsize=(10,6))
plt.plot(fare_values, revenues, label='Revenue Curve (Logarithmic Model)', color='blue')
plt.axvline(optimal_fare, color='green', linestyle='--', label=f'Optimal Fare = {optimal_fare:.2f} USD')
plt.axhline(optimal_revenue, color='red', linestyle='--', label=f'Max Revenue = {optimal_revenue:.2f} USD')
plt.xlabel('Average Fare (USD)')
plt.ylabel('Revenue (USD)')
plt.title('Revenue vs Fare (Logarithmic Demand Model)')
plt.grid(True)
plt.legend()
plt.show()

# Print optimal fare and maximum revenue
print(f"Optimal fare for maximum revenue: {optimal_fare:.2f} USD")
print(f"Maximum revenue at optimal fare: {optimal_revenue:.2f} USD")






    
    
    

