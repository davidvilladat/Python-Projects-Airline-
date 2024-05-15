# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:29:54 2024

@author: crist
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 3)
y = X @ np.array([5, 3, -2]) + np.random.randn(100)

# Set up different values of lambda (alpha in Ridge function)
lambdas = [0.01, 0.1, 1, 10, 100]
coefficients = []

for l in lambdas:
    model = Ridge(alpha=l)
    model.fit(X, y)
    coefficients.append(model.coef_)

# Plotting the effect of different lambdas on coefficients
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(lambdas, [coef[i] for coef in coefficients], marker='o', label=f'Coefficient {i+1}')

plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Coefficient Magnitude')
plt.title('Effect of Lambda on Coefficients in Ridge Regression')
plt.legend()
plt.show()
