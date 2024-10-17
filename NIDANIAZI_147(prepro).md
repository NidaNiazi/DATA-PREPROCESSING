# Required Libraries for Data Analysis and Preprocessing
# Pandas
Purpose: Used for data manipulation and analysis. It provides data structures like DataFrames to handle structured data efficiently.
# NumPy
Purpose: Provides support for numerical operations and array handling. It is often used in conjunction with Pandas for efficient data manipulation.
# Matplotlib
Purpose: A plotting library for creating static, interactive, and animated visualizations in Python.
# Seaborn
Purpose: Built on top of Matplotlib, it provides a high-level interface for drawing attractive statistical graphics, making it easier to visualize data.
# SciPy
Purpose: Contains modules for optimization, integration, interpolation, eigenvalue problems, and other scientific computations. It is useful for advanced mathematical operations, such as Z-score calculations.
# Scikit-learn
Purpose: A powerful library for machine learning that provides simple and efficient tools for data mining and data analysis. It includes various algorithms and preprocessing utilities.




```python
#this is the way to import your libraries if not previously installed
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels

```


```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Step 1: Load the data
data = pd.read_csv('data.csv')  # Replace 'data.csv' with your dataset

# Step 2: Inspect the data
print("Shape of the dataset:", data.shape)
print("\nFirst five rows of the dataset:\n", data.head())
print("\nData types of each column:\n", data.info())
print("\nNumber of duplicate rows:", data.duplicated().sum())

# Step 3: Summary statistics
print("\nSummary statistics:\n", data.describe())


```

# Steps Before Preprocessing:
Data Collection
Gather data from different sources like CSV, Excel files, databases, APIs, or web scraping.


```python
import pandas as pd

# Load data from a CSV file
data = pd.read_csv('data.csv')

# Load data from an Excel file
data = pd.read_excel('data.xlsx')

# Load data from an SQL database
import sqlite3
conn = sqlite3.connect('database.db')
data = pd.read_sql_query("SELECT * FROM table_name", conn)

```

# Inspect the Data

Check the shape of the data to understand how many rows and columns you are dealing with.
Preview the data to get a sense of its structure.
Understand column names, types of features (categorical or numerical), and inspect any glaring issues.


```python
# Check the shape of the data (rows, columns)
print(data.shape)

# Preview the first few rows of the data
print(data.head())

# Check data types of each column
print(data.info())

# Check for duplicate rows
print(data.duplicated().sum())

```




```python

```

# 1. Handling Missing Values
Imputation: Replace missing values with the mean, median, or mode, or use more complex techniques like KNN imputation.


```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
dataset['column_name'] = imputer.fit_transform(dataset[['column_name']])

```


```python
# Dropping: Remove rows or columns with missing values
dataset.dropna(inplace=True)
```

# 2. Handling Categorical Variables
One-Hot Encoding: Convert categorical values into binary columns


```python
dataset = pd.get_dummies(dataset, columns=['category_column'])
# Label Encoding: Convert categories into numerical labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataset['category_column'] = label_encoder.fit_transform(dataset['category_column'])

```

# 3. Scaling Features
Standardization: Rescale the data so that it has a mean of 0 and standard deviation of 1


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset[['column1', 'column2']] = scaler.fit_transform(dataset[['column1', 'column2']])

```


```python
#Min-Max Scaling: Scale values to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset[['column1', 'column2']] = scaler.fit_transform(dataset[['column1', 'column2']])

```

# 4. Handling Outliers
Capping/Flooring: Replace extreme values with a fixed value, like 1st and 99th percentiles


```python
dataset['column'] = dataset['column'].clip(lower=dataset['column'].quantile(0.01), upper=dataset['column'].quantile(0.99))

```


```python
#Z-score method: Remove data points that are far from the mean.
from scipy import stats
dataset = dataset[(np.abs(stats.zscore(dataset['column'])) < 3)]
```

# 5. Feature Engineering
Feature Creation: Create new features based on existing ones.
python


```python
dataset['new_feature'] = dataset['column1'] / dataset['column2']

```

# 6. Dimensionality Reduction
PCA (Principal Component Analysis): Reduce the number of features while preserving variance.



```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(dataset)

```

# 7. Data Splitting
Train-Test Split: Split data into training and testing sets


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

# 8. Encoding Dates/Time
Extracting Date Components: If you have a date column, extract the year, month, or day.



```python
dataset['year'] = dataset['date_column'].dt.year
dataset['month'] = dataset['date_column'].dt.month

```

# 9. Dealing with Imbalanced Data
Oversampling: Use techniques like SMOTE to generate more samples from the minority class


```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

```


```python
#Undersampling: Randomly sample from the majority class to balance the data.

from imblearn.under_sampling import RandomUnderSampler
under_sampler = RandomUnderSampler()
X_resampled, y_resampled = under_sampler.fit_resample(X, y)
```

# FOLLOWING IS THE PREPROCESSING TECHNIQUES AS ONE BIG CODE FOR BETTER UNDERSTANDING


```python
# Step 1: Install Required Libraries
# You can uncomment the following line to install the libraries if you haven't already
# !pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Step 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE

# Step 3: Load the Data
data = pd.read_csv('data.csv')  # Replace with your dataset file
print("Initial data shape:", data.shape)

# Step 4: Create Dependent and Independent Variable Factors
# Assuming 'target' is the name of your target variable
X = data.drop('target', axis=1)  # Independent variables
y = data['target']  # Dependent variable
print("Independent variables shape:", X.shape)
print("Dependent variable shape:", y.shape)

# Step 5: Handle Missing Values
# Using SimpleImputer to fill missing values
imputer = SimpleImputer(strategy='mean')  # Replace with 'median' or 'most_frequent' as needed
X_imputed = imputer.fit_transform(X)
print("Missing values handled. Imputed data shape:", X_imputed.shape)

# Step 6: Data Encoding (One Hot Encoding)
X_encoded = pd.get_dummies(X_imputed, drop_first=True)
print("Data after One Hot Encoding shape:", X_encoded.shape)

# Step 7: Splitting Data into Train and Test Set
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Step 8: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling applied.")

# Step 9: Outlier Detection (Using Z-score)
z_scores = np.abs(stats.zscore(X_train_scaled))
outliers = np.where(z_scores > 3)[0]  # Identify outliers
print(f"Indices of outliers in the training set: {outliers}")

# Optional: Remove outliers from training data
# X_train_scaled = np.delete(X_train_scaled, outliers, axis=0)
# y_train = np.delete(y_train.values, outliers, axis=0)

# Step 10: Convert Ordinal Data into Numeric Values
ordinal_columns = ['ordinal_column']  # Replace with actual ordinal column names
ordinal_encoder = OrdinalEncoder()
X_train_scaled[:, [X_train.columns.get_loc(col) for col in ordinal_columns]] = ordinal_encoder.fit_transform(
    X_train_scaled[:, [X_train.columns.get_loc(col) for col in ordinal_columns]]
)
X_test_scaled[:, [X_test.columns.get_loc(col) for col in ordinal_columns]] = ordinal_encoder.transform(
    X_test_scaled[:, [X_test.columns.get_loc(col) for col in ordinal_columns]]
)

# Step 11: Data Binning (if applicable)
# Example: Creating bins for a continuous variable
# X_train_scaled['binned_column'] = pd.cut(X_train_scaled['continuous_column'], bins=5, labels=False)  # Replace with your column
# X_test_scaled['binned_column'] = pd.cut(X_test_scaled['continuous_column'], bins=5, labels=False)  # Replace with your column

# Step 12: Managing Under and Over Sampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print("Resampled training set shape:", X_train_resampled.shape)

# Final Overview
print("\nData preprocessing completed successfully.")
print("Final training data shape:", X_train_resampled.shape)
print("Final test data shape:", X_test_scaled.shape)

```

# Difference between EDA and Pre-Processing :
# EDA:

Visualization: Create charts (histograms, box plots, scatter plots) to explore data distribution and relationships between features.
Summary Statistics: Calculate mean, median, standard deviation, correlations to summarize the data.
Outlier Detection: Detect potential anomalies in the data through visual or statistical means.
Distribution Analysis: Study the distribution of numerical variables (e.g., normal, skewed).
# Preprocessing:

Handling Missing Data: Imputation (mean, median, mode) or removing missing entries.
Encoding Categorical Data: One-hot encoding, label encoding to convert categorical variables into numerical ones.
Feature Scaling: Normalization or standardization to bring all features to a common scale.
Handling Outliers: Removing or capping extreme values.
Data Transformation: Log transformations, binning, or polynomial features to transform data for better performance.


```python


```
