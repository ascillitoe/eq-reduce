import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import flowfield, user_data
from navbar import navbar
from utils import convert_latex

homepage = dbc.Container(
    [
        html.H2(dcc.Markdown("Dimension Reduction with *equadratures*")),
        html.H4('Active Subspaces'),
        html.H4('Variable Projection'),
        html.H4('References')
    ], fluid=True
)

footer = html.Div(
        [
            html.P('Made by Ashley Scillitoe'),
#            html.A(html.P('ascillitoe.com'),href='https://ascillitoe.com'),
            html.P(html.A('ascillitoe.com',href='https://ascillitoe.com')),
            html.P('Copyright © 2021')
        ]
    ,className='footer'
)

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        navbar,
        html.Div(homepage,id="page-content"),
        footer,
    ],
    style={'padding-top': '70px'}
)

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return homepage
    if pathname == '/flowfield':
        return flowfield.layout
    elif pathname == '/datadriven':
        return user_data.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)
