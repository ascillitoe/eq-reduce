import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import flowfield, user_data
from navbar import navbar


homepage = dbc.Container(
    [
        html.H2(dcc.Markdown("Dimension Reduction with *equadratures*")),

        html.P(children='Delicious \(\pi\) is inline with my goals.'),
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

#dbc.Container(
#    dbc.Row(
#        dbc.Col(
#            html.P(
#                [
#                    html.Span('Your Name', className='mr-2'), 
#                    html.A(html.I(className='fas fa-envelope-square mr-1'), href='mailto:<you>@<provider>.com'), 
#                    html.A(html.I(className='fab fa-github-square mr-1'), href='https://github.com/<you>/<repo>'), 
#                    html.A(html.I(className='fab fa-linkedin mr-1'), href='https://www.linkedin.com/in/<you>/'), 
#                    html.A(html.I(className='fab fa-twitter-square mr-1'), href='https://twitter.com/<you>'), 
#                ], 
#                className='footer'
#            )
#        )
#    )
#)

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        navbar,
        html.Div(homepage,id="page-content"),
        footer
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
