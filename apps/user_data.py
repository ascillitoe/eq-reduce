import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State, ALL
from flask_caching import Cache
import plotly.graph_objs as go

import os
import numpy as np
import pyvista as pv
import equadratures as eq
from joblib import Parallel, delayed, cpu_count
from utils import deform_airfoil, eval_poly, standardise, get_samples_constraining_active_coordinates, get_airfoils, airfoil_mask

from app import app

ncores = cpu_count()

###################################################################
# Collapsable more info card
###################################################################
info_text = dcc.Markdown('''
Coming soon!
''')
info = dbc.Card(
    [
        dbc.CardHeader(dbc.Form(
            [
                dbc.Label(dcc.Markdown('**More Information**'),className="mr-3"),
                dbc.Button(
                    "Expand",
                    color="primary",
                    id="data-info-button",
                    className="py-0"
                ),
            ],inline=True,
        )),
        dbc.Collapse(
            dbc.CardBody(info_text),
            id="data-info-collapse",
        ),
    ], style={'margin-top':'10px'} 
)

###################################################################
# The overall app layout
###################################################################
layout = dbc.Container(
    [
    html.H2("Data-Driven Dimension Reduction"),
    dcc.Markdown('''
    This app embeds computes dimension reducing subspaces. Upload your own data, or choose from an example dataset.

    **Coming soon!**
    '''),
#    ***Scroll to the bottom of the page for more information!***
#    '''),
    info,
#    dcc.Store(id='airfoil-data'),
#    tooltips
    ],
    fluid = True
)


###################################################################
# Other callbacks
###################################################################
# More info collapsable
@app.callback(
    Output("data-info-collapse", "is_open"),
    [Input("data-info-button", "n_clicks")],
    [State("data-info-collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open