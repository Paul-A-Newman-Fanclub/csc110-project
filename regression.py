# import necessary modules
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import math
import plotly.express as px


# load the data, then isolate variables of interest, remove observations with missing values
dataset = pandas.read_csv('data/owid-co2-data.csv', sep=',')
dataset = dataset[['gdp', 'co2', 'consumption_co2']]
dataset = dataset.dropna()


# plot data, to visualize relationship, roughly
fig = px.scatter_3d(dataset, x='co2', y='consumption_co2', z='gdp',
                    title="3D Scatterplot of Co2 Emissions, Co2 Consumption, and GDP",
                    labels={
                        "co2": "CO2 Emissions (Million of Tonnes)",
                        "consumption_co2": "CO2 Consumption (Million of Tonnes)",
                        "gdp": "GDP (USD)"
                    }
                    )
fig.show()


# Divide dataset into target(response variable) and features(predictors)
gdp = dataset[['gdp']].copy()
co2 = dataset[['co2', 'consumption_co2']].copy()


# transform target and features into numpy.array - appropriate data type for scikit regression
co2 = co2.to_numpy()
gdp = gdp.to_numpy()
gdp = np.array(gdp).squeeze()


# Algorithm
l_reg = linear_model.LinearRegression()


np.random.seed(6969)
# Split data into training and testing (test size of 20%)
co2_train, co2_test, gdp_train, gdp_test = train_test_split(co2, gdp, test_size=0.2)


# train the model
co2_model = l_reg.fit(co2_train, gdp_train)
predictions = co2_model.predict(co2_test)
r2 = l_reg.score(co2_train, gdp_train)


# Calculate accuracy
rmse = metrics.mean_squared_error(y_true=gdp_test, y_pred=predictions, squared=False)


# Relevant text output
print("Coefficient of Determination (r^2):", l_reg.score(co2_train, gdp_train))
print("Coefficient of Correlation(r): ", math.sqrt(l_reg.score(co2_train, gdp_train)))
b_1, b_2 = l_reg.coef_
intercept = l_reg.intercept_
print(f'linear model: gdp = {intercept} + {b_1} * co2_emissions + {b_2} * co2_consumption')
print(f'Accuracy: {rmse}')
