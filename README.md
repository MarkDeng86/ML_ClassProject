# House Price Prediction Project

## Project Overview
This project implements various machine learning models to predict house prices based on a comprehensive dataset containing 81 features, including both quantitative (36) and categorical (43) variables. The goal is to compare different regression techniques and evaluate their performance in predicting house prices.

## Dataset Description
- Training data: 1460 instances
- Features: 81 total attributes
  - 36 quantitative features (e.g., square footage, number of rooms)
  - 43 categorical features (e.g., neighborhood, house style)
  - ID and Sale Price columns

## Methodology

### Data Preprocessing
1. Handling missing values
   - Numerical features: Filled with 0
   - Categorical features: Filled with 'None'

2. Feature Engineering
   - Ordinal encoding for features with natural ordering
   - One-hot encoding for remaining categorical variables
   - Feature scaling using StandardScaler
   - Dimensionality reduction using PCA (100 components)

### Models Implemented

1. **Ridge Regression**
   - Tested alpha values from 0.25 to 2.0
   - Best performance achieved with alpha = 2.0
   - Validation score: 0.868 (R² score)

2. **Lasso Regression**
   - Tested alpha values from 0.25 to 2.0
   - Included tolerance parameter of 0.0925
   - Validation score: 0.836 (R² score)

3. **Neural Network (MLPRegressor)**
   - Architecture: (200, 100, 50, 30) neurons
   - ReLU activation function
   - Learning rate: 0.001
   - Momentum: 0.9
   - Validation score: 0.841 (R² score)

4. **Binary Classification (SVM)**
   - Price threshold: $200,000
   - Linear kernel
   - Tested C values from 0.4 to 1.4
   - Best accuracy: 93.15% (C = 0.6)

## Results Summary
1. 1. Polynomial transformation with Lasso regularization (α=1000) performed best among regression models with 91.45% accuracy
2. Neural networks achieved comparable performance to linear models, with best accuracy of 89.26% (50,30 layer structure)
3. For binary classification, both linear and sigmoid kernel SVMs achieved highest accuracy of 92.47%

## Requirements
- Python 3.x
- scikit-learn
- numpy
- pandas
- matplotlib

## Model Performance Comparison
| Model | Best Validation Score |
|-------|-------------------|
| Linear Regression (no regularization) | 87.74% |
| Linear Regression with Lasso (α=10) | 89.04% |
| Polynomial (X²) with Lasso (α=1000) | 91.45% |
| Neural Network (50,30) with ReLU | 89.26% |
| SVM Linear/Sigmoid Kernel (Binary) | 92.47% |

Note: Regression models use R² score, while SVM uses accuracy score.
