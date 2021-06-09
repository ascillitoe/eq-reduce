import dash
import dash_bootstrap_components as dbc
import os

app = dash.Dash(__name__, suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.SPACELAB, 'https://codepen.io/chriddyp/pen/bWLwgP.css'],
        external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" ])
app.title = "Dimension reduction with equadratures"

server = app.server
server.secret_key = os.environ.get('secret_key', 'secret')
