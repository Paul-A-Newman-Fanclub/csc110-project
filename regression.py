"""
CSC110 Final Project: Multivariate Regression Analysis and Model Generation
"""
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import math
import plotly.express as px

# MODEL 1

# Load the data, then isolate variables of interest, remove observations with missing values
dataset = pandas.read_csv('owid-co2-data.csv', sep=',')
dataset = dataset[['gdp', 'co2', 'consumption_co2']]
dataset = dataset.dropna()

# Plot data to roughly visualize relationship
fig = px.scatter_3d(dataset, x='co2', y='consumption_co2', z='gdp',
                    title="3D Scatterplot of Co2 Emissions, Co2 Consumption, and GDP",
                    labels={
                        "co2": "CO2 Emissions (Million of Tonnes)",
                        "consumption_co2": "CO2 Consumption (Million of Tonnes)",
                        "gdp": "GDP (USD)"
                    }
                    )
fig.show()

# Divide dataset into target (response variable) and features (predictors)
gdp = dataset[['gdp']].copy()
co2 = dataset[['co2', 'consumption_co2']].copy()

# Transform target and features into numpy.array (appropriate data type for scikit regression)
co2 = co2.to_numpy()
gdp = gdp.to_numpy()
gdp = np.array(gdp).squeeze()

# Initialize Model 1
l_reg = linear_model.LinearRegression()

np.random.seed(6969)
# Split data into training and testing (test size of 20%)
co2_train, co2_test, gdp_train, gdp_test = train_test_split(co2, gdp, test_size=0.2)

# Train the model using training data, use model to predict based on testing data
co2_model = l_reg.fit(co2_train, gdp_train)
predictions = co2_model.predict(co2_test)

# Calculate accuracy
rmse = metrics.mean_squared_error(y_true=gdp_test, y_pred=predictions, squared=False)

# Print results of the regression analysis
print('Model 1: Co2 Emissions and Co2 Consumption')
print("Coefficient of Determination (r^2):", l_reg.score(co2_train, gdp_train))
print("Coefficient of Correlation (r):", math.sqrt(l_reg.score(co2_train, gdp_train)))
b_1, b_2 = l_reg.coef_
intercept = l_reg.intercept_
print(f'Linear Model: gdp = {intercept} + {b_1} * co2_emissions + {b_2} * co2_consumption')
print(f'Accuracy: {rmse} USD\n')

# Plot the model in 3D space
dataset_train = pandas.DataFrame()
dataset_train.insert(0, 'Co2 Emissions (Millions of Tonnes)', co2_train[:, 0])
dataset_train.insert(1, 'Co2 Consumption (Millions of Tonnes)', co2_train[:, 1])
dataset_train.insert(2, 'GDP (USD)', gdp_train)

x1 = np.linspace(0, 10000, 100).reshape(100, 1)
x2 = np.linspace(0, 10000, 100).reshape(100, 1)
points = np.hstack((x1, x2))
x3 = co2_model.predict(points)

df = pandas.DataFrame()
df.insert(loc=0, column='Co2 Emissions (Millions of Tonnes)', value=x1.reshape(100, ))
df.insert(loc=1, column='Co2 Consumption (Millions of Tonnes)', value=x2.reshape(100, ))
df.insert(loc=2, column='GDP (USD)', value=list(x3))
fig = px.scatter_3d(dataset_train, x='Co2 Emissions (Millions of Tonnes)',
                    y='Co2 Consumption (Millions of Tonnes)', z="GDP (USD)",
                    title="Linear Model of Co2 Emissions, Co2 Consumption, and GDP")

fig.add_traces([go.Scatter3d(x=df['Co2 Emissions (Millions of Tonnes)'],
                             y=df['Co2 Consumption (Millions of Tonnes)'],
                             z=df['GDP (USD)'], mode='lines')])
fig.show()

# MODEL 2

# Load the data, then isolate variables of interest, remove observations with missing values
dat = pandas.read_csv('owid-co2-data.csv', sep=',')
dat = dat[['gdp', 'co2', 'consumption_co2', 'methane', 'nitrous_oxide']]
dat = dat.dropna()

# Divide dataset into target (response variable) and features (predictors)
gdp = dat[['gdp']].copy()
ghg = dat[['co2', 'consumption_co2', 'methane', 'nitrous_oxide']].copy()

# Transform target and features into numpy.array (appropriate data type for scikit regression)
ghg = ghg.to_numpy()
gdp = gdp.to_numpy()
gdp = np.array(gdp).squeeze()

# Initialize Model 2
lin_reg = linear_model.LinearRegression()

np.random.seed(6969)
# Split data into training and testing (test size of 20%)
ghg_train, ghg_test, gdp_train, gdp_test = train_test_split(ghg, gdp, test_size=0.2)

# Train the model using training data, use model to predict based on testing data
ghg_model = l_reg.fit(ghg_train, gdp_train)
predictions = ghg_model.predict(ghg_test)

# Calculate accuracy
rmse = metrics.mean_squared_error(y_true=gdp_test, y_pred=predictions, squared=False)

# Print results of the regression analysis
print('Model 2: Co2 Emissions, Co2 Consumption, Methane and Nitrous Oxide Emissions')
print("Coefficient of Determination (r^2):", l_reg.score(ghg_train, gdp_train))
print("Coefficient of Correlation (r):", math.sqrt(l_reg.score(ghg_train, gdp_train)))
b_1, b_2, b_3, b_4 = l_reg.coef_
intercept = l_reg.intercept_
print(f'Linear Model: gdp = {intercept} + {b_1} * co2 + {b_2} * consumption_co2'
      f' + {b_3} * methane + {b_4} * nitrous_oxide')
print(f'Accuracy: {rmse} USD')
