from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html

from app import app

EQ_LOGO = "https://equadratures.org/_static/logo_new.png"

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=EQ_LOGO, height="35px")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://equadratures.org/",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Introduction", href="/")),
                        dbc.NavItem(dbc.NavLink("Flowfield Estimation", href="/flowfield")),
                        dbc.NavItem(dbc.NavLink("User Data", href="/datadriven"))
                    ], className="ml-auto", navbar=True
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ]
    ),
    color="light",
    dark=False,
    className="mb-3",
)

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
