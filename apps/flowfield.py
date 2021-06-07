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
# Load data
###################################################################
dataloc = 'SUBSAMPLED_DATA'

# Load baseline aerofoil
base_airfoil = pv.read(os.path.join(dataloc,'surface_base.vtk')).points

# Load baseline data
x = np.load(os.path.join(dataloc,'xpts.npy'))
y = np.load(os.path.join(dataloc,'ypts.npy'))
nx = len(x)
ny = len(y)
npts = int(nx*ny)
pts = np.arange(npts)
ypred = np.empty(len(pts))

# Load poly coeffs etc
coeffs = np.load(os.path.join(dataloc,'coeffs.npy'))
lowers = np.load(os.path.join(dataloc,'lowers.npy'))
uppers = np.load(os.path.join(dataloc,'uppers.npy'))
W      = np.load(os.path.join(dataloc,'W.npy'))
var_name = [r'$C_p$',r'$\nu_t/\nu$',r'$u/U_{\infty}$',r'$v/U_{\infty}$']

# Load training data to plot on summary plots
X = np.load(os.path.join(dataloc,'X.npy'))
Y = np.load(os.path.join(dataloc,'Y.npy'))
X = standardise(X)

###################################################################
# Global definitions
###################################################################
cache = Cache(app.server, config={"CACHE_TYPE": "SimpleCache"})

# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Introduction", href="/home")),
        dbc.NavItem(dbc.NavLink("Airfoil", href="/page-1")),
        dbc.NavItem(dbc.NavLink("Flowfield", href="/page-2")),
        dbc.NavItem(dbc.NavLink("Performance", href="/page-3"))
    ],
    brand="NavbarSimple",
    brand_href="#",
    color="primary",
    dark=True,
)

###################################################################
# Homepage
###################################################################


###################################################################
# Flowfield estimation
###################################################################
# Card containing the Interface to define deformed airfoil via Hicks-Henne bumps 
airfoil_bumps_def = dbc.Container(
    [
        dbc.Row(dbc.Col(dbc.Button("Add Bump", id="add-bump", color="primary", n_clicks=0),width=3),align='center',justify='start'
        ),
        dbc.Row(dbc.Col(html.Div(id='slider-container', children=[]))) #The slider-container callback adds an extra row here each time "Add Bump" is pressed
    ],fluid=True 
)

# The airfoil plot
#fig = create_airfoil_plot()
fig_layout = {"xaxis": {"title": r'$x/C$'}, "yaxis": {"title": r'$y/C$'},'margin':{'t':10,'r':0,'l':0,'b':0,'pad':0},'autosize':True,
        'paper_bgcolor':'white','plot_bgcolor':'white'}
data = go.Scatter(x=base_airfoil[:,0],y=base_airfoil[:,1],mode='lines',name='NACA0012',line_width=4,line_color='black')
fig = go.Figure(data=data, layout=fig_layout)
fig.update_xaxes(range=[-0.01,1.01],color='black',linecolor='black',showline=True,tickcolor='black',ticks='outside')
fig.update_yaxes(range=[-0.102,0.102],scaleanchor='x',scaleratio=1, #scaleratio and scaleanchor overridden by this annoyingly (have to hardcode width and height for now)
    color='black',linecolor='black',showline=True,tickcolor='black',ticks='outside')

airfoil_plot = dbc.Container(
        dcc.Graph(figure=fig,id="airfoil-plot",style={'width': 'inherit','height':'inherit'},config={'responsive': True}),
        fluid=True,style={'height':'25vh'})

# Airfoil definitions card
airfoil_definitions_card = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown("**Airfoil definition**")),
        dbc.CardBody(airfoil_bumps_def),
        ],style={'height':'38vh'}
)


###################################################################
# Flowfield estimation page
###################################################################
# Active subspace (sufficient summary) plot
summary_plot = dbc.Container(
    [
        dcc.Graph(id="summary-plot", style={}),#'width':'100%','height':'100%'}),
        dcc.Graph(id="Wproject-plot",style={})#'width':'100%','height':'100%'})
    ]
)

## Inactive subspace plot (sampling inactive subspace with hit and run)
#inactive_plot = dbc.Container(
#    [
#        dbc.Row(
#            [
#                dbc.Col(dbc.Label("Number of samples"),width=3),
#                dbc.Col(dbc.Input(id='nsamples',type="number", min=10, max=1000, step=10,value=10),width=3),
#                dbc.Col(dbc.Spinner(html.Div(id='samples-finished'),color="primary"),width=2)
#            ],align='center',justify='start'
#        ),
#        dbc.Row(
#            [
#                dbc.Col(daq.ToggleSwitch(id='toggle-samples', value=False,label={'label':'Show samples','style':{'font-weight':'bold','font-size':16}}),width='auto'),
#                dbc.Col(daq.ToggleSwitch(id='toggle-stats', value=True,label={'label':'Show two sigma bounds','style':{'font-weight':'bold','font-size':16}}),width='auto'),
#
#            ]
#        ),
#        dbc.Row(dbc.Col(dcc.Graph(id="inactive-plot")))
#    ]
#)

# point information card
point_info_card = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown('**Local ridge information** \(select an approximation point first!)')),
        dbc.CardBody(summary_plot)
    ], 
)

# Flowfield plot
flowfield_plot = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id="var-select",
                    options=[
                        {"label": "Pressure coefficient", "value":0},
                        {"label": "Turbulent viscosity", "value":1},
                        {"label": "u velocity", "value":2},
                        {"label": "v velocity", "value":3},
                    ],value=0,placeholder="Pressure coefficient",clearable=False,searchable=False
                    ),width=4
                ),
                dbc.Col(dbc.Button("Compute Flowfield", id="compute-flowfield", color="primary"),width='auto'),
                dbc.Col(daq.ToggleSwitch(id='toggle-points', value=False,label={'label':'Show approximation points','style':{'font-weight':'bold','font-size':16}}),width='auto'),
                dbc.Col(dbc.Spinner(html.Div(id='flowfield-finished'),color="primary"),width=1)
            ],justify='start',align='center'
        ),
#        dbc.Row(dbc.Col(dcc.Graph(id="flowfield-plot",style={'height': 'inherit'}),width=12))#,style={'width':'100vw'})#style={'height': '60vh'})))
        dcc.Graph(id="flowfield-plot",style={'width': 'inherit'})#,style={'width':'100vw'})#style={'height': '60vh'})))
        ], fluid=True
)

# Visualisation tabs
visual_tabs = dbc.Card(
    dbc.Tabs(
        [
            dbc.Tab(airfoil_plot,label='Airfoil Definition'),
            dbc.Tab(flowfield_plot,label='Flowfield Estimation')
        ],    
    ), style={'height':'70vh'}
)

tooltips = html.Div([
        dbc.Tooltip("Add Hicks-Henne bump functions to deform the baseline NACA0012 airfoil", target="add-bump"),
    ])

# The overall app layout
layout = dbc.Container(
    [
    dbc.Row(
        [
            dbc.Col(
                [
                airfoil_definitions_card,
                point_info_card       
            ],width=5),
            dbc.Col(visual_tabs,width=7)
        ]
    ),
    dcc.Store(id='airfoil-data'),
    tooltips
    ],
    fluid = True
)

###################################################################
# Function to compute flowfield
###################################################################
# This function is moderately time consuming so memoize it
@cache.memoize(timeout=600)
def compute_flowfield(design_vec,var):
    ypred = np.array(Parallel(n_jobs=ncores,verbose=1,)(delayed(eval_poly)(design_vec,lowers[pt,var],uppers[pt,var],coeffs[pt,var,:],W[pt,var,:,:]) for pt in pts))
    return ypred

###################################################################
# Utilities to specify bumps and plot airfoils
###################################################################
# Simple callback to disable limit number of bumps added
@app.callback(
    Output('add-bump','disabled'),
    Input('add-bump','n_clicks'))
def limit_bumps(n_clicks):
    max_bumps=3
    if n_clicks < max_bumps-1:
        return False
    else:
        return True

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
#                    dbc.Label('Surface',html_for='select-surface'),
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
                ,width=3
            ),

            dbc.Col(
                dbc.Form([
                    dbc.Label('Bump location (x/C)',html_for='slider-x'),
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
                    dbc.Label('Bump amplitude',html_for='slider-amp',id="bump-amp-label"),
                    dbc.Tooltip(f"+ve to deform outwards, -ve to deform inwards", target=f"slider-amp-wrapper-{n_clicks}"),
                    html.Div(
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
                    ),id=f"slider-amp-wrapper-{n_clicks}")
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
    layout = {"xaxis": {"title": r'$x/C$'}, "yaxis": {"title": r'$y/C$'},'margin':{'t':0,'r':0,'l':0,'b':0},'autosize':True}
    data = go.Scatter(x=base_airfoil[:,0],y=base_airfoil[:,1],mode='lines',name='NACA0012',line_width=4,line_color='black')
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(range=[-0.01,1.01])
    fig.update_yaxes(range=[-0.102,0.102],scaleanchor='x',scaleratio=1) #scaleratio and scaleanchor overridden by this annoyingly (have to hardcode width and height for now)
#    fig.update_layout(height=300,width=300*(1/0.2))
    return fig

# callback to create aerofoil plots
@app.callback(
    Output("airfoil-plot", "figure"),
    Output('airfoil-data', 'data'),
    Input({'type': 'slider-x'  , 'index': ALL}, 'value'),
    Input({'type': 'slider-amp', 'index': ALL}, 'value'),
    Input({'type': 'select-surface', 'index': ALL}, 'value'),
    )
def make_graph(xs,amps,surfs):
    deformed_airfoil, design_vec = deform_airfoil(base_airfoil,xs,amps,surfs)
    #fig.add_trace(go.Scatter(x=deformed_airfoil[:,0],y=deformed_airfoil[:,1],mode='lines',name='Deformed',line_width=4,line_color='blue'))
    ntraces = len(fig.data)
    if ntraces == 1: # If only baseline plotted, plot Deformed trace
        fig.add_trace(go.Scatter(x=deformed_airfoil[:,0],y=deformed_airfoil[:,1],mode='lines',name='Deformed',line_width=4,line_color='blue'))
    else: # otherwise, update existing trace
        fig.data[1].y = deformed_airfoil[:,1]
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
    Input('var-select', 'value'),
    Input('toggle-points','value'),
    prevent_initial_call=True)
def make_flowfield(n_clicks,airfoil_data,var,show_points):
    # Parse data
    design_vec = airfoil_data['design-vec']
    airfoil_x  = airfoil_data['airfoil-x']
    airfoil_y  = airfoil_data['airfoil-y']
    airfoil_x  = np.hstack([airfoil_x[-1],airfoil_x])
    airfoil_y  = np.hstack([airfoil_y[-1],airfoil_y])

    # Setup fig
    layout={'clickmode':'event+select','margin':dict(t=0,r=0,l=0,b=0,pad=0),'showlegend':False,"xaxis": {"title": 'x/C'}, "yaxis": {"title": 'y/C'},
            'paper_bgcolor':'white','plot_bgcolor':'white','autosize':False}
    fig = go.Figure(layout=layout)

    # Plot airfoil
    fig.add_trace(go.Scatter(x=airfoil_x,y=airfoil_y,mode='lines',name='Deformed',line_width=8,line_color='blue',fill='tozeroy',fillcolor='rgba(0, 0, 255, 1.0)'))

    # Contour plot (if button has just been pressed)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'compute-flowfield' in changed_id:
        ypred = compute_flowfield(design_vec,var)
        fig.add_trace(go.Contour(x=x,y=y,z=ypred.reshape(len(x),len(y)),transpose=True, colorbar=dict(len=0.7),#title=dict(text=var_name[var],side='right')), colorbar title removed for now as doesn't work with latex. (Plotly issue #2231)
            contours=dict(
            start=np.nanmin(ypred),
            end=np.nanmax(ypred),
            size=(np.nanmax(ypred)-np.nanmin(ypred))/20,
            )

        ))

    if show_points: 
        xx,yy = np.meshgrid(x,y,indexing='ij')
        xx,yy = airfoil_mask(xx,yy,airfoil_x,airfoil_y)
        fig.add_trace(go.Scatter(x=xx.flatten(),y=yy.flatten(),mode='markers',marker_color='black',opacity=0.4,marker_symbol='circle-open',marker_size=6,marker_line_width=2))

    #fig.update_xaxes(range=[-1.12844, 1.830583])
    fig.update_xaxes(range=[-0.8, 1.6], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-0.5822106,0.5001755],scaleanchor = "x", scaleratio = 1, showgrid=False, zeroline=False, visible=False)

    return fig, None

###################################################################
# Sufficient summary plot (and W project over airfoil plot)
###################################################################
@app.callback(
    Output('summary-plot', 'figure'),
    Output('Wproject-plot', 'figure'),
    Input('flowfield-plot', 'clickData'),
    Input('airfoil-data','data'),
    Input('var-select', 'value'),
    prevent_initial_call=True)
def display_active_plot(clickData,airfoil_data,var):
    # Sufficient summary plot
    layout1={"xaxis": {"title": r'$\mathbf{W}^T\mathbf{x}$'}, "yaxis": {"title": var_name[var]},'margin':{'t':0,'r':0,'l':0,'b':60}}

    fig1 = go.Figure(layout=layout1)
    # W projection plot
    layout2={'margin':dict(t=0,r=0,l=0,b=0,pad=0),'showlegend':False,"xaxis": {"title": r'$x/C$'}, "yaxis": {"title": r'$y/C$'},
            'paper_bgcolor':'white','plot_bgcolor':'white'}#,'height':350}
    fig2 = go.Figure(layout=layout2)
    fig2.update_xaxes(title=r'$x/C$',range=[-0.02,1.02],showgrid=True, zeroline=False, visible=True,gridcolor='rgba(0,0,0,0.2)',showline=False,linecolor='black')
    fig2.update_yaxes(scaleanchor = "x", scaleratio = 1, showgrid=False, showticklabels=False, zeroline=False, visible=False)
    fig2.add_trace(go.Scatter(x=base_airfoil[:,0],y=base_airfoil[:,1],mode='lines',line_width=4,line_color='black'))

    if clickData is not None:
        pointdata = clickData['points'][0]
        if "pointIndex" in pointdata: #check click event corresponds to the point cloud
            # Get point info
            n = pointdata['pointIndex']
            xn = pointdata['x']
            yn = pointdata['y']
            w = W[pts][n,var,:,:]

            # Summary plot
            ##############
            # Plot training design
            Yn = Y[n,:,var]
            u = (X @ w).flatten()
            fig1.add_trace(go.Scatter(x=u,y=Yn,mode='markers',name='Training designs',
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
            fig1.add_trace(go.Scatter(x=u_poly,y=Y_poly.squeeze(),mode='lines',name='Ridge approximation',line_width=4,line_color='black' ))

            # Plot deformed design
            u_design = design_vec @ w
            Y_design = newpoly.get_polyfit(u_design)
            fig1.add_trace(go.Scatter(x=u_design.squeeze(),y=Y_design.squeeze(),mode='markers',name='Deformed design',
                marker=dict(symbol='circle-open',color='firebrick',size=25,line=dict(width=5))
            ))
            
            # W projection plot
            ###################
            # Split into suction and pressure
            scale = 0.2
            wp = -w[0::2,0]*scale
            ws = w[1::2,0]*scale
            x_bumps = np.linspace(0.05,0.9,25)
            airfoilp = base_airfoil[:128,:]
            airfoils = base_airfoil[128:,:]
            yp = np.interp(x_bumps, airfoilp[::-1,0], airfoilp[::-1,1])
            ys = np.interp(x_bumps, airfoils[:,0], airfoils[:,1])
            # Pressure
            fig2.add_trace(go.Scatter(x=x_bumps,y=yp,mode='lines',line_width=1,line_color='black'))  # plot hidden line to use with fill=tonexty below
            fig2.add_trace(go.Scatter(x=x_bumps,y=yp+wp,mode='lines',line_width=4,line_color='LightSkyBlue',fill='tonexty',fillcolor='rgba(135,206,250, 0.3)'))
            # Suction
            fig2.add_trace(go.Scatter(x=x_bumps,y=ys,mode='lines',line_width=1,line_color='black'))  # plot hidden line to use with fill=tonexty below
            fig2.add_trace(go.Scatter(x=x_bumps,y=ys+ws,mode='lines',line_width=4,line_color='LightSkyBlue',fill='tonexty',fillcolor='rgba(135,206,250, 0.3)'))
            # Annotation
            fig2.add_annotation(x=x_bumps[3],y=ys[3]+ws[3],text='Deformations leading to increased W^Tx',xanchor='left',ax=50,ay=-120,font={'size':14,'color':'LightSkyBlue'},valign='top',showarrow=True,arrowhead=3,arrowsize=2,arrowcolor='LightSkyBlue')

    return fig1, fig2

####################################################################
## Inactive plot
####################################################################
#@app.callback(
#    Output('inactive-plot', 'figure'),
#    Output("samples-finished", "children"),
#    Input('flowfield-plot', 'clickData'),
#    Input('airfoil-data','data'),
#    Input('var-select', 'value'),
#    Input('nsamples', 'value'),
#    Input('toggle-samples','value'),
#    Input('toggle-stats','value'),
#    prevent_initial_call=True)
#def display_inactive_plot(clickData,airfoil_data,var,nsamples,plot_samples,plot_stats):
#    layout={'margin':dict(t=0,r=0,l=0,b=0,pad=0),'showlegend':False,"xaxis": {"title": 'x/C'}, "yaxis": {"title": 'y/C'},
#            'paper_bgcolor':'white','plot_bgcolor':'white'}
#    fig = go.Figure(layout=layout)
#    fig.update_xaxes(title=r'$x/C$',range=[-0.02,1.02],showgrid=True, zeroline=False, visible=True,gridcolor='rgba(0,0,0,0.2)',showline=False,linecolor='black')
#    fig.update_yaxes(scaleanchor = "x", scaleratio = 1, showgrid=False, showticklabels=False, zeroline=False, visible=False)
#    fig.add_trace(go.Scatter(x=base_airfoil[:,0],y=base_airfoil[:,1],mode='lines',line_width=4,line_color='black'))
#
#    if clickData is not None:
#        pointdata = clickData['points'][0]
#        if "pointIndex" in pointdata: #check click event corresponds to the point cloud
#            # Get point info
#            n = pointdata['pointIndex']
#
#            # Sample the inactive subspace and return nsamples number of design vectors
#            w = W[pts][n,var,:,:]
#            Xsamples = get_samples_constraining_active_coordinates(w, nsamples, np.array([0]))
#            
#            # Get the airfoil coordinates from the design vectors
#            y_p, y_s = get_airfoils(base_airfoil,Xsamples) 
#        
#        # Plot the samples
#        if plot_samples:
#            opacity = min(0.2,float(10/nsamples))
#            for s in range(nsamples):
#                fig.add_trace(go.Scatter(x=base_airfoil[:,0],y=y_p[:,s],mode='lines',line_width=1,line_color='grey',opacity=opacity))
#                fig.add_trace(go.Scatter(x=base_airfoil[:,0],y=y_s[:,s],mode='lines',line_width=1,line_color='grey',opacity=opacity))
#        if plot_stats: 
#            s_mean = np.mean(y_s,axis=1); p_mean = np.mean(y_p,axis=1)
#            s_std = 1.96*np.std(y_s,axis=1); p_std = 1.96*np.std(y_p,axis=1)
#            fig.add_trace(go.Scatter(x=base_airfoil[:,0],y=p_mean-p_std,mode='lines',line_width=2,line_color='LightSkyBlue',opacity=0.1))
#            fig.add_trace(go.Scatter(x=base_airfoil[:,0],y=p_mean+p_std,mode='lines',line_width=2,line_color='LightSkyBlue',opacity=0.1,
#                fill='tonexty',fillcolor='rgba(135,206,250,0.6)'))
#            fig.add_trace(go.Scatter(x=base_airfoil[:,0],y=s_mean-s_std,mode='lines',line_width=2,line_color='LightSkyBlue',opacity=0.1))
#            fig.add_trace(go.Scatter(x=base_airfoil[:,0],y=s_mean+s_std,mode='lines',line_width=2,line_color='LightSkyBlue',opacity=0.1,
#                fill='tonexty',fillcolor='rgba(135,206,250,0.6)'))
#
#    return fig, None


# TODO:
#2) pretyify!
#3) instructions etc
#4) go live!
