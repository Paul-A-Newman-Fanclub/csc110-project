"""
Regression Analysis Cs110 final project

Description
===============================
This module reads data from the Green House Gas Emissions Csv file,
it creates two linear regression model trained to predict GDP. The first uses
co2 emission and co2 consumption as predictors. The second uses
CO2 consumption and CO2 emissions, N2O, and methane emissions.
The module visualizes the the first model but not the second as it has too many dimensions.

Copyright and Usage Information
===============================
This file is Copyright (c) 2020 Michael Umeh, Daniel Lazaro, Matthew Parvaneh,
Tobey Brizeula.
"""
import math
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import plotly.express as px
from plotly import graph_objs as go


def plot_scatter(dataframe: pandas.DataFrame) -> None:
    """
    Displays 3d scatter plot given pandas dataframe with 3 columns consisting of
    numerical observations

    Parameter descriptions:
    - dataframe: is a dataframe consisting of 3 columns corresponding to the 3
        variables needed to train model 1

    Preconditions:
    - dataframe != pandas.DataFrame()
    """
    fig = px.scatter_3d(dataframe, x='co2', y='consumption_co2', z='gdp',
                        title="3D Scatterplot of Co2 Emissions, Co2 Consumption, and GDP",
                        labels={
                            "co2": "CO2 Emissions (Million of Tonnes)",
                            "consumption_co2": "CO2 Consumption (Million of Tonnes)",
                            "gdp": "GDP (USD)"
                        }
                        )
    fig.show()


def regression1_text_info(co2: np.ndarray, co2_t: np.ndarray, gdp: np.ndarray,
                          gdp_t: np.ndarray) -> None:
    """ Prints relevant regression information for the first model

    Parameter descriptions:
    - co2: corresponds to c02 emission and consumption variables in training dataframe
    - co2_t: corresponds to c02 emission and consumption variables in testing dataframe
    - gdp: corresponds to gdp variable in training dataframe
    - gdp_t: corresponds to gdp variable in testing dataframe

    Preconditions:
    - co2.shape == (2545, 2)
    - gdp.shape == (2545,)
    - co2_t.shape == (637, 2)
    - gdp_t.shape == (637,)
    """
    # Make predicitions
    predictions = co2_model.predict(co2_t)

    # Calculate accuracy
    rmse = metrics.mean_squared_error(y_true=gdp_t, y_pred=predictions, squared=False)

    # Relevant text output
    print('\nModel 1: Co2 Emissions and Co2 Consumption')
    print("Coefficient of Determination (r^2):", l_reg.score(co2, gdp))
    print("Coefficient of Correlation(r): ", math.sqrt(l_reg.score(co2, gdp)))
    b_1, b_2 = l_reg.coef_
    intercept = l_reg.intercept_
    print(f'linear model: gdp = {intercept} + {b_1} * co2_emissions + {b_2} * co2_consumption')
    print(f'Accuracy: {rmse} USD\n')


def plot_3d(co2: np.ndarray, gdp: np.ndarray) -> None:
    """
    Plot the first model in 3d space, along with scatter points of the training dataset,
    in order to visualize relationship

    Parameter descriptions:
    - co2: corresponds to c02 emission and consumption variables in training dataframe
    - gdp: corresponds to gdp variable in training dataframe

    Preconditions:
    - co2.shape == (2545, 2)
    - gdp.shape == (2545,)
    """

    # Create dataframe with scatter plot points
    dataset_train = pandas.DataFrame()
    dataset_train.insert(0, 'Co2 Emissions (Millions of Tonnes)', co2[:, 0])
    dataset_train.insert(1, 'Co2 Consumption (Millions of Tonnes)', co2[:, 1])
    dataset_train.insert(2, 'GDP (USD)', gdp)

    # Generate points to plot line
    x1 = np.linspace(0, 10000, 100).reshape(100, 1)
    x2 = np.linspace(0, 10000, 100).reshape(100, 1)
    points = np.hstack((x1, x2))
    x3 = co2_model.predict(points)

    # Create dataframe to to plot line
    df = pandas.DataFrame()
    df.insert(loc=0, column='Co2 Emissions (Millions of Tonnes)', value=x1.reshape(100, ))
    df.insert(loc=1, column='Co2 Consumption (Millions of Tonnes)', value=x2.reshape(100, ))
    df.insert(loc=2, column='GDP (USD)', value=list(x3))

    # Plot Scatter Plot Points
    fig = px.scatter_3d(dataset_train, x='Co2 Emissions (Millions of Tonnes)',
                        y='Co2 Consumption (Millions of Tonnes)', z="GDP (USD)",
                        title="Linear Model of Co2 Emissions, Co2 Consumption, and GDP")
    # Plot line
    fig.add_traces([go.Scatter3d(x=df['Co2 Emissions (Millions of Tonnes)'],
                                 y=df['Co2 Consumption (Millions of Tonnes)'],
                                 z=df['GDP (USD)'], mode='lines')])
    # Display Graph
    fig.show()


def regression2_text_info(ghg: np.ndarray, ghg_t: np.ndarray,
                          gdp: np.ndarray, gdp_t: np.ndarray) -> None:
    """ Prints relevant regression information for the second model

    Parameter descriptions:
    - ghg: corresponds to greenhouse gas variables in training dataframe
    - ghg_t: corresponds to greenhouse gas variables in testing dataframe
    - gdp: corresponds to gdp variable in training dataframe for model 2
    - gdp_t: corresponds to gdp variable in testing dataframe for model 2

    Preconditions:
    - ghg.shape == (2502, 4)
    - gdp.shape == (2502,)
    - ghg_t.shape == (626, 4)
    - gdp_t.shape == (626,)
    """
    predictions = ghg_model.predict(ghg_t)

    # Calculate accuracy
    rmse = metrics.mean_squared_error(y_true=gdp_t, y_pred=predictions, squared=False)

    # Relevant text output
    print('Model 2: Co2 Emissions, Co2 Consumption, Methane and Nitrous Oxide Emissions')
    print("Coefficient of Determination (r^2):", l_reg.score(ghg, gdp))
    print("Coefficient of Correlation(r): ", math.sqrt(l_reg.score(ghg, gdp)))
    b_1, b_2, b_3, b_4 = l_reg.coef_
    intercept = l_reg.intercept_
    print(f'linear model: gdp = {intercept} + {b_1} * co2 + {b_2} * consumption_co2'
          f' + {b_3} * methane + {b_4} * nitrous_oxide')
    print(f'Accuracy: {rmse} USD\n')


if __name__ == '__main__':
    # load the data, then isolate variables of interest, remove observations with missing values
    dataset = pandas.read_csv('owid-co2-data.csv', sep=',')
    dataset = dataset[['gdp', 'co2', 'consumption_co2']]
    dataset = dataset.dropna()

    # plot data, to visualize relationship, roughly
    plot_scatter(dataset)

    # Divide dataset into target(response variable) and features(predictors)
    gdp1 = dataset[['gdp']].copy()
    co21 = dataset[['co2', 'consumption_co2']].copy()

    # transform target and features into numpy.array - appropriate data type for scikit regression
    co21 = co21.to_numpy()
    gdp1 = gdp1.to_numpy()
    gdp1 = np.array(gdp1).squeeze()

    # Model 1
    l_reg = linear_model.LinearRegression()

    np.random.seed(6969)
    # Split data into training and testing (test size of 20%)
    co2_train, co2_test, gdp_train, gdp_test = train_test_split(co21, gdp1, test_size=0.2)

    # train the model
    co2_model = l_reg.fit(co2_train, gdp_train)

    # Outputs relevant regression analysis information
    regression1_text_info(co2_train, co2_test, gdp_train, gdp_test)

    # Plot the model in 3d space
    plot_3d(co2_train, gdp_train)

    # Model 2
    # load the data, then isolate variables of interest, remove observations with missing values
    dat = pandas.read_csv('owid-co2-data.csv', sep=',')
    dat = dat[['gdp', 'co2', 'consumption_co2', 'methane', 'nitrous_oxide']]
    dat = dat.dropna()

    # Divide dataset into target(response variable) and features(predictors)
    gdp1 = dat[['gdp']].copy()
    ghg1 = dat[['co2', 'consumption_co2', 'methane', 'nitrous_oxide']].copy()

    # transform target and features into numpy.array - appropriate data type for scikit regression
    ghg1 = ghg1.to_numpy()
    gdp1 = gdp1.to_numpy()
    gdp1 = np.array(gdp1).squeeze()

    # Model 2
    lin_reg = linear_model.LinearRegression()

    np.random.seed(6969)
    # Split data into training and testing (test size of 20%)
    ghg_train, ghg_test, gdp_train, gdp_test = train_test_split(ghg1, gdp1, test_size=0.2)

    # train the model
    ghg_model = l_reg.fit(ghg_train, gdp_train)

    # Print relevant text output
    regression2_text_info(ghg_train, ghg_test, gdp_train, gdp_test)

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['math', 'sklearn', 'sklearn.model_selection',
                          'numpy', 'pandas', 'plotly', 'plotly.express', 'python_ta.contracts'],
        'allowed-io': ['regression2_text_info', 'regression1_text_info'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
