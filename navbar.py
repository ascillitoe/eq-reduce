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
                        dbc.Col(html.Img(src=EQ_LOGO, height="40px")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://equadratures.org/",
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Introduction", href="/",active='exact',className='px-3')),
                    dbc.NavItem(dbc.NavLink("Flowfield Estimation", href="/flowfield", active='exact',className='px-3')),
                    dbc.NavItem(dbc.NavLink("My Data", href="/datadriven",active='exact',className='px-3'))
                ], className="ml-auto", navbar=True, fill=True
            ),
        ], fluid=True
    ),
    color="white",
    dark=False,
    className="mb-0",
    fixed='top'
)
