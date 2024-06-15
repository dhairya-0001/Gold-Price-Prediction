# Gold-Price-Prediction

### Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
```
- `numpy` and `pandas`: Used for data manipulation and analysis.
- `matplotlib.pyplot` and `seaborn`: Used for data visualization.
- `sklearn.model_selection.train_test_split`: Used to split the dataset into training and testing sets.
- `sklearn.ensemble.RandomForestRegressor`: Used to build and train the Random Forest regression model.
- `sklearn.metrics`: Used for evaluating the model's performance.

### Data Collection and Processing
```python
# Load the CSV data into a pandas dataframe
gold_data = pd.read_csv("/content/gld_price_data.csv")
```
- Loads the gold price dataset from a CSV file into a pandas DataFrame.

```python
# Print the first five rows of the dataframe
print(gold_data.head())
```
- Displays the first five rows of the dataset to give an overview of the data.

```python
# Print the last five rows of the dataframe
print(gold_data.tail())
```
- Displays the last five rows of the dataset.

```python
# Print the shape of the dataframe (number of rows and columns)
print(gold_data.shape)
```
- Prints the shape of the DataFrame, showing the number of rows and columns.

```python
# Get more information about the dataframe
print(gold_data.info())
```
- Provides a concise summary of the DataFrame, including data types and non-null counts.

```python
# Check the number of missing values
print(gold_data.isnull().sum())
```
- Checks for any missing values in each column of the DataFrame.

```python
# Get the statistical measures of the data
print(gold_data.describe())
```
- Displays basic statistical measures (like mean, standard deviation, min, max, etc.) for numerical columns.

### Correlation Analysis
```python
# Correlation: Positive Correlation and Negative Correlation
numeric_data = gold_data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()

print("\nCorrelation matrix:")
print(correlation)
```
- Selects numerical columns from the DataFrame and calculates the correlation matrix to understand the relationships between variables.

```python
# Constructing a heatmap to understand the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
```
- Visualizes the correlation matrix using a heatmap. This helps to quickly identify which variables are positively or negatively correlated with each other.

```python
# Correlation values of the GLD
print(correlation['GLD'])
```
- Prints the correlation values of the `GLD` column with other numerical columns, showing how each variable is related to the gold price.

### Distribution of Gold Price
```python
# Checking the distribution of the gold price
sns.displot(gold_data['GLD'], color='gold')
```
- Plots the distribution of the gold price to understand its distribution pattern.

### Splitting the Features and Target
```python
# Splitting the Features and Target
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

print(X)
print(Y)
```
- Separates the features (independent variables) and the target (dependent variable) in the dataset. `X` contains all the features except 'Date' and 'GLD', while `Y` contains the 'GLD' column.

### Splitting Data into Training and Testing Sets
```python
# Splitting data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```
- Splits the dataset into training and testing sets. 80% of the data is used for training, and 20% is used for testing. `random_state=2` ensures reproducibility of the split.

### Model Training: Random Forest Regressor
```python
# Model Training: Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)

# Training the model
regressor.fit(X_train, Y_train)
```
- Initializes a Random Forest Regressor with 100 trees (`n_estimators=100`).
- Trains the model on the training data (`X_train` and `Y_train`).

### Model Evaluation
```python
# Prediction on Test Data
test_data_prediction = regressor.predict(X_test)

print(test_data_prediction)
```
- Uses the trained model to make predictions on the test data (`X_test`).
- Prints the predicted values for the test data.

```python
# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error: ", error_score)
```
- Calculates and prints the R-squared error, which indicates how well the model's predictions match the actual values. An R-squared value closer to 1 indicates a better fit.

### Visualization
```python
# Compare the actual values and predicted values in a plot
Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Values vs Predicted Values')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
```
- Converts `Y_test` to a list for plotting.
- Plots the actual gold prices (`Y_test`) in blue and the predicted gold prices (`test_data_prediction`) in green.
- Adds a title, axis labels, and a legend to the plot.
- Displays the plot to visually compare the actual and predicted values.
