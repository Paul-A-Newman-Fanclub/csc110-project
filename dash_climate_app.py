"""
CSC110 Final Project
"""
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from typing import Tuple


# Creating the Dash App for our map
map_app = dash.Dash(__name__)

# Reading in the csv file using pandas
df = pd.read_csv('owid-co2-data.csv')

# Importing and tidying/preprocessing our climate data
df2 = df[df['iso_code'].notnull()]
df2 = df2[df2['iso_code'] != "OWID_WRL"]
df2 = df2[['country', 'year', 'co2', 'gdp']]

# Establishing the years to be visualized in both maps
years_co2 = {year: str(year) for year in range(1990, 2019)}
years_gdp = {year: str(year) for year in range(1990, 2016)}

# Setting up the web app layout
map_app.layout = html.Div(id='main', style={'font-family': 'Helvetica',
                                            'backgroundColor': '#2E3440',
                                            'padding-top': '25px',
                                            'padding-bottom': '25px',
                                            'padding-left': '75px',
                                            'padding-right': '75px'}, children=[

    # Page Title
    html.H1(children="CSC110 Final Project: Climate Change Dashboard",
            style={'text-align': 'center', 'font-family': 'Helvetica', 'color': '#FFFFFF'}),

    # Adding a space between title and subsequent heading
    html.Br(),

    # Sub-heading
    html.H2(children='CO2 Emissions across the World (by year)',
            style={'font-family': 'Helvetica', 'color': '#FFFFFF'}),

    # Creating a Graph Dash Core Component to hold CO2 map in this section in the main div
    dcc.Graph(id='co2_emission_map'),

    # Creating a Graph Dash Core Component to allow user to change years on map
    dcc.Slider(id='choose_yr_co2',
               min=1990,
               max=2018,
               marks=years_co2,
               value=1990,
               included=False,
               updatemode='drag'
               ),

    # Adding a space between slider and next graph
    html.Br(),

    # Sub-heading
    html.H2(children='GDP in US$ (by year)',
            style={'font': 'Helvetica', 'color': '#FFFFFF'}),

    # Creating another Graph core component to hold the gdp map
    dcc.Graph(id='gdp_map'),

    # Creating another slider exclusive to the gdp map
    dcc.Slider(id='choose_yr_gdp',
               min=1990,
               max=2016,
               marks=years_gdp,
               value=1990,
               included=False,
               updatemode='drag'
               ),
])


# Using Dash Callback decorator to account for real-time interactions
@map_app.callback(
    # Establishing where the two outputs of this callback function go
    [Output(component_id='co2_emission_map', component_property='figure'),
     Output(component_id='gdp_map', component_property='figure')],
    # Establishing where the inputs for this callback function come from
    [Input(component_id='choose_yr_co2', component_property='value'),
     Input(component_id='choose_yr_gdp', component_property='value')]
)
def change_graph_year(co2_year: int, gdp_year: int) -> Tuple[Figure, Figure]:
    """
    Dash Callback Function. Takes in the currently selected 'year' marks on
    the sliders of both the co2 and gdp maps as argument, outputting the corresponding
    map for each year.
    """
    # Making a copy of the original dataframe
    df_copy = df2.copy()

    # Extracting only the observations corresponding to each year
    df_co2 = df_copy[df_copy['year'] == co2_year]
    df_gdp = df_copy[df_copy['year'] == gdp_year]

    # Creating the CO2 emission choropleth map (with plotly express)
    fig_co2 = px.choropleth(
        data_frame=df_co2,
        scope='world',
        projection='kavrayskiy7',
        locationmode='country names',
        locations='country',
        color_continuous_scale=px.colors.sequential.Sunset,
        color='co2',
        labels={'co2': 'CO2 Emissions (million tonnes)'},
        hover_data=['country', 'co2'],
        height=600
    )

    # Optional stylistic changes for CO2 map
    fig_co2.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#2E3440', subunitcolor='rgba(0,0,0,0)',
                 showframe=False),
        plot_bgcolor='#2E3440',
        paper_bgcolor='#2E3440',
        font_color="White",
        font_family="Helvetica",
    )

    # Creating the GDP choropleth map
    fig_gdp = px.choropleth(
        data_frame=df_gdp,
        scope='world',
        projection='kavrayskiy7',
        locationmode='country names',
        locations='country',
        color_continuous_scale=px.colors.sequential.Blugrn,
        color='gdp',
        labels={'gdp': 'GDP (US$)'},
        hover_data=['country', 'gdp'],
        height=600
    )

    # Optional stylistic changes for GDP map
    fig_gdp.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#2E3440', subunitcolor='rgba(0,0,0,0)',
                 showframe=False),
        plot_bgcolor='#2E3440',
        paper_bgcolor='#2E3440',
        font_color="White",
        font_family="Helvetica",
    )

    # Return fig objects, to be passed into both dcc.Graph components
    return fig_co2, fig_gdp


if __name__ == '__main__':
    map_app.run_server(debug=False)
