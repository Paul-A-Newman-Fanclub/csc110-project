# import necessary modules
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import math
import plotly.express as px


# load the data, then isolate variables of interest, remove observations with missing values
dataset = pandas.read_csv('owid-co2-data.csv', sep=',')
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


# Model 1
l_reg = linear_model.LinearRegression()


np.random.seed(6969)
# Split data into training and testing (test size of 20%)
co2_train, co2_test, gdp_train, gdp_test = train_test_split(co2, gdp, test_size=0.2)


# train the model
co2_model = l_reg.fit(co2_train, gdp_train)
predictions = co2_model.predict(co2_test)


# Calculate accuracy
rmse = metrics.mean_squared_error(y_true=gdp_test, y_pred=predictions, squared=False)


# Relevant text output
print('Model 1: Co2 Emissions and Co2 Consumption')
print("Coefficient of Determination (r^2):", l_reg.score(co2_train, gdp_train))
print("Coefficient of Correlation(r): ", math.sqrt(l_reg.score(co2_train, gdp_train)))
b_1, b_2 = l_reg.coef_
intercept = l_reg.intercept_
print(f'linear model: gdp = {intercept} + {b_1} * co2_emissions + {b_2} * co2_consumption')
print(f'Accuracy: {rmse} USD\n')


# load the data, then isolate variables of interest, remove observations with missing values
dat = pandas.read_csv('owid-co2-data.csv', sep=',')
dat = dat[['gdp', 'co2', 'consumption_co2', 'methane', 'nitrous_oxide']]
dat = dat.dropna()


# Divide dataset into target(response variable) and features(predictors)
gdp = dat[['gdp']].copy()
ghg = dat[['co2', 'consumption_co2', 'methane', 'nitrous_oxide']].copy()


# transform target and features into numpy.array - appropriate data type for scikit regression
ghg = ghg.to_numpy()
gdp = gdp.to_numpy()
gdp = np.array(gdp).squeeze()


# Model 2
lin_reg = linear_model.LinearRegression()


np.random.seed(6969)
# Split data into training and testing (test size of 20%)
ghg_train, ghg_test, gdp_train, gdp_test = train_test_split(ghg, gdp, test_size=0.2)


# train the model
ghg_model = l_reg.fit(ghg_train, gdp_train)
predictions = ghg_model.predict(ghg_test)

# Calculate accuracy
rmse = metrics.mean_squared_error(y_true=gdp_test, y_pred=predictions, squared=False)


# Relevant text output
print('Model 2: Co2 Emissions, Co2 Consumption, Methane and Nitrous Oxide Emissions')
print("Coefficient of Determination (r^2):", l_reg.score(ghg_train, gdp_train))
print("Coefficient of Correlation(r): ", math.sqrt(l_reg.score(ghg_train, gdp_train)))
b_1, b_2, b_3, b_4 = l_reg.coef_
intercept = l_reg.intercept_
print(f'linear model: gdp = {intercept} + {b_1} * co2 + {b_2} * consumption_co2'
      f' + {b_3} * methane + {b_4} * nitrous_oxide')
print(f'Accuracy: {rmse} USD')


# Interactive Function that predicts gdp
def gdp_predictor(co2: float, co2_consumption: float, meth: float, nitrous: float, model: int) -> float:
    """
    Using one of two the models trained in this module, returns prediction of the gdp of a given country
    based on the co2 emissions, co2 consumption, methane emissions, and nitrous emissions.
    The parameter model, specifies which model to use to make prediction. A value of 1 corresponds
    to the first module and a value of two corresponds to the second.

    Preconditions:
     - model == 1 or model == 2
    """
    if model == 1:
        return 62758394187.130005 - 2714676506.8076687 * co2 + 4921820701.91739 * co2_consumption
    else:
        return 39526250426.829346 - 3104942924.878715 * co2 + 5192264291.57594 * co2_consumption + 198442893.4650092\
               * meth + 1877327020.449066 * nitrous

