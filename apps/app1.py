import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

layout = html.Div([
    html.H3('App 1'),
    html.Div(id='d2')
])

@app.callback(Output('d2','children'),[Input('dd','value')])
def return_1(value):
    print("App 1 Clicked with value {}".format(value))
    return html.H1("App 1 Clicked with value {}".format(value))
