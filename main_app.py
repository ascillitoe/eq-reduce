import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State, ALL
from flask_caching import Cache
import plotly.graph_objs as go

import pickle
import numpy as np
import pyvista as pv
import equadratures as eq
from joblib import Parallel, delayed, cpu_count
from utils import deform_airfoil, eval_poly, standardise

ncores = cpu_count()

###################################################################
# Load data
###################################################################
# Load baseline aerofoil
base_airfoil = pv.read('surface_base.vtk').points

# Load baseline mesh (and subsample)
basegrid = pv.read('basegrid.vtk')
xskip = 3
yskip = 2
nx,ny,_ = basegrid.dimensions
npts = int(nx*ny)
idx = np.arange(npts).reshape(nx,ny,order='F')[::yskip,::xskip].T
points = basegrid.points_matrix[:,:,0,:2].transpose([1,0,2])
pts = idx.flatten()
x = points[:,0,0][::xskip]
y = points[0,:,1][::yskip]
assert len(x)*len(y)==len(pts)
ypred = np.empty(len(pts))

# Load poly coeffs etc
coeffs = np.load('coeffs.npy')
lowers = np.load('lowers.npy')
uppers = np.load('uppers.npy')
W = np.load('W.npy')
var_name = ['Cp','nut/nu','u/U','v/U']
var = 0 # TODO add this as option

# Load training data to plot on summary plots
X = np.load('X.npy')
X = standardise(X)
Y = np.load('Y.npy')[pts,:,:]

###################################################################
# Overall app
###################################################################
# Define app
app = dash.Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.SPACELAB, 'https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Rapid flowfield estimation with polynomial ridges"
cache = Cache(app.server, config={"CACHE_TYPE": "SimpleCache"})

# Interface to define deformed airfoil via Hicks-Henne bumps 
define_bumps = html.Div([
    dbc.Row([
        dbc.Col(dbc.Button("Add Bump", id="add-bump", color="primary", n_clicks=0),width=2),
        dbc.Col(dbc.Button("Compute Flowfield", id="compute-flowfield", color="primary"),width=2),
        dbc.Col(dbc.Spinner(html.Div(id='flowfield-finished'),color="primary"),width=1)
    ],justify="start",align="center"),
    html.Div(id='slider-container', children=[]),
])

# The overall app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [   
                dbc.Col(define_bumps,width=4),
                dbc.Col(dcc.Graph(id="airfoil-plot"), width=8)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="summary-plot"),width=4),
                dbc.Col(dbc.FormGroup(
                    [
                    daq.ToggleSwitch(id='toggle-points', value=False,label="Toggle approximation points"),
#                    dbc.Checklist(options=[{"label": "Toggle approximation points", "value": True}],value=[],id="toggle-points",switch=True),
                    dcc.Graph(id="flowfield-plot",style={'height': '65vh'})
                    ]),
                width=8)
            ]
        ),
    # dcc.Store inside the app that stores the intermediate value
    dcc.Store(id='airfoil-data')
    ],
    fluid = True
)

###################################################################
# Function to compute flowfield
###################################################################
# This function is moderately time consuming so memoize it
@cache.memoize(timeout=600)
def compute_flowfield(design_vec):
    ypred = np.array(Parallel(n_jobs=ncores,verbose=1,)(delayed(eval_poly)(design_vec,lowers[pt,var],uppers[pt,var],coeffs[pt,var,:],W[pt,var,:,:]) for pt in pts))
    return ypred

###################################################################
# Utilities to specify bumps and plot airfoils
###################################################################
# callback to define bump properties (new one each time add bump is pressed)
@app.callback(
    Output('slider-container', 'children'),
    Input('add-bump', 'n_clicks'),
    State('slider-container', 'children'))
def define_bump(n_clicks, children):
    new_bump = dbc.Row(
        [
            dbc.Col(
                dbc.Form([
                    dbc.Label('Surface',html_for='select-surface'),
                    dbc.Select(
                        id={
                            'type':'select-surface',
                            'index': n_clicks
                            },
                        options=[
                            {"label": "Suction", "value": "s"},
                            {"label": "Pressure", "value": "p"},
                        ],
                        placeholder='Suction',
                        value='s'
                    )
                ])
                ,width=2
            ),

            dbc.Col(
                dbc.Form([
                    dbc.Label('Bump location (x/c)',html_for='slider-x'),
                    dcc.Slider(
                        id={
                            'type':'slider-x',
                            'index': n_clicks
                            },
                        min=0.05,
                        max=0.9,
                        step=0.03541666666,
                        value=0.5104166666666666,
                        marks={
                            0.05: {'label': '0.05'},
                            0.9: {'label': '0.9'}
                            },
                        tooltip = { 'always_visible': True, 'placement': 'bottom' }
                    )
                ])
                ,width=4
            ),

            dbc.Col( 
                dbc.Form([
                    dbc.Label('Bump amplitude',html_for='slider-amp'),
                    dcc.Slider(
                        id={
                            'type':'slider-amp',
                            'index': n_clicks
                            },
                        min=-0.005,
                        max=0.01,
                        step=0.0005,
                        value=0.0,
                        marks={
                            -0.005: {'label': '-0.005'},
                            0: {},
                            0.01: {'label': '0.01'}
                            },
                        tooltip = { 'always_visible': True, 'placement': 'bottom' }
                    )
                ])
                ,width=4
            )
        ]
    )
    children.append(new_bump)
    return children

###################################################################
# Aerofoil plot
###################################################################
# Create initial baseline airfoil fig
def create_airfoil_plot():
    layout = {"xaxis": {"title": 'x'}, "yaxis": {"title": 'y'}}
    data = go.Scatter(x=base_airfoil[:,0],y=base_airfoil[:,1],mode='lines',name='NACA0012',line_width=4,line_color='black')
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    return fig

# callback to create aerofoil plots
@app.callback(
    Output("airfoil-plot", "figure"),
    Output('airfoil-data', 'data'),
    Input({'type': 'slider-x'  , 'index': ALL}, 'value'),
    Input({'type': 'slider-amp', 'index': ALL}, 'value'),
    Input({'type': 'select-surface', 'index': ALL}, 'value'))
def make_graph(xs,amps,surfs):
    fig = create_airfoil_plot()
    deformed_airfoil, design_vec = deform_airfoil(base_airfoil,xs,amps,surfs)
    fig.add_trace(go.Scatter(x=deformed_airfoil[:,0],y=deformed_airfoil[:,1],mode='lines',name='Deformed',line_width=4,line_color='blue'))
    return fig,{'design-vec':design_vec,'airfoil-x':deformed_airfoil[:,0].tolist(),'airfoil-y':deformed_airfoil[:,1].tolist()}

###################################################################
# Flowfield plot
###################################################################
# callback to create flowfield plot
@app.callback(
    Output("flowfield-plot", "figure"),
    Output("flowfield-finished", "children"),
    Input("compute-flowfield", "n_clicks"),
    Input('airfoil-data', 'data'),
    Input('toggle-points','value'),
    prevent_initial_call=True)
def make_flowfield(n_clicks,airfoil_data,show_points):
    # Parse data
    design_vec = airfoil_data['design-vec']
    airfoil_x  = airfoil_data['airfoil-x']
    airfoil_y  = airfoil_data['airfoil-y']
    
    # Setup fig
    layout={'clickmode':'event+select','margin':dict(t=10),'showlegend':False,"xaxis": {"title": 'x'}, "yaxis": {"title": 'y'},
            'paper_bgcolor':'white','plot_bgcolor':'white'}

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=airfoil_x,y=airfoil_y,mode='lines',name='Deformed',line_width=8,line_color='blue',fill='tozeroy',fillcolor='rgba(0, 0, 255, 1.0)'))

    #fig.update_xaxes(range=[-1.12844, 1.830583])
    fig.update_xaxes(range=[-0.8, 1.6], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-0.5822106,0.5001755],scaleanchor = "x", scaleratio = 1, showgrid=False, zeroline=False, visible=False)

    # Contour plot (if button has just been pressed)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'compute-flowfield' in changed_id:
        ypred = compute_flowfield(design_vec)
        fig.add_trace(go.Contour(x=x,y=y,z=ypred.reshape(len(x),len(y)),transpose=True,colorbar=dict(title=var_name[var], titleside='right'), contours=dict(
            start=np.nanmin(ypred),
            end=np.nanmax(ypred),
            size=(np.nanmax(ypred)-np.nanmin(ypred))/20,
            )
        ))

    if show_points: 
        xx,yy = np.meshgrid(x,y,indexing='ij')
        fig.add_trace(go.Scatter(x=xx.flatten(),y=yy.flatten(),mode='markers',marker_color='black',opacity=0.4,marker_symbol='circle-open',marker_size=6,marker_line_width=2))
       
    return fig, None

###################################################################
# Sufficient summary plot 
###################################################################
@app.callback(
    Output('summary-plot', 'figure'),
    Input('flowfield-plot', 'clickData'),
    Input('airfoil-data','data'),
    prevent_initial_call=True)
def display_click_data(clickData,airfoil_data):
    layout={"xaxis": {"title": 'W^Tx'}, "yaxis": {"title": var_name[var]}}
    fig = go.Figure(layout=layout)
    if clickData is not None:
        pointdata = clickData['points'][0]
        if "pointIndex" in pointdata: #check click event corresponds to the point cloud
            n = pointdata['pointIndex']
            xn = pointdata['x']
            yn = pointdata['y']

            # Plot training design
            Yn = Y[n,:,var]
            u = (X @ W[pts][n,var,:,:]).flatten()
            fig.add_trace(go.Scatter(x=u,y=Yn,mode='markers',name='Training designs',
                marker=dict(color='LightSkyBlue',size=15,opacity=0.5,line=dict(color='black',width=1))
            ))

            design_vec = airfoil_data['design-vec']

            # Set poly
            mybasis = eq.Basis("total-order")
            param = eq.Parameter(distribution='uniform', lower=lowers[pts][n,var],upper=uppers[pts][n,var],order=2)
            newpoly = eq.Poly(param, mybasis, method='least-squares')
            newpoly._set_coefficients(user_defined_coefficients=coeffs[pts][n,var,:])
 
            # Plot poly
            u_poly = np.linspace(np.min(u)-0.25,np.max(u)+0.25,50)
            Y_poly = newpoly.get_polyfit(u_poly.reshape(-1,1))
            fig.add_trace(go.Scatter(x=u_poly,y=Y_poly.squeeze(),mode='lines',name='Ridge approximation',line_width=4,line_color='black' ))

            # Plot deformed design
            u_design = design_vec @ W[pts][n,var,:,:]
            Y_design = newpoly.get_polyfit(u_design)
            fig.add_trace(go.Scatter(x=u_design.squeeze(),y=Y_design.squeeze(),mode='markers',name='Deformed design',
                marker=dict(symbol='circle-open',color='firebrick',size=25,line=dict(width=5))
            ))
            print(Y_design)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

# TODO:
# 1) dropdown to select var
# 3) Have figure below summary plot with W projected over aerofoil to tell user how to deform the aerofoil
# 4) Tabs to switch between summary plot and inactive subspace
