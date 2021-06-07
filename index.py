import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import flowfield, app2
from navbar import navbar


homepage = dbc.Container(
    [
         html.H1("Dimension reduction with equadratures")
    ]
)


app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        navbar,
        html.Div(homepage,id="page-content")#, style=CONTENT_STYLE)
    ]
)

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return homepage
    if pathname == '/flowfield':
        return flowfield.layout
    elif pathname == '/datadriven':
        return app2.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)
