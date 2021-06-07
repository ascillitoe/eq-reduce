import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from time import sleep

from app import app

layout = html.Div([
    html.H3('App 2'),
    html.Div(id='d1')
])

@app.callback(Output('d1','children'),[Input('dd','value')])
def return_2(value):
    print('Sleeping')
    sleep(2)
    print("App 2 Clicked with value {}".format(value))
    return html.H1("App 2 Clicked with value {}".format(value))
