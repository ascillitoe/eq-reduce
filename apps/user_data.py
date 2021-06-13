import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_table
from dash_table.Format import Format, Scheme, Trim
from dash.dependencies import Input, Output, State, ALL
from flask_caching import Cache
import plotly.graph_objs as go

import os
import base64
import io
import pickle
import jsonpickle
import math
import numpy as np
import pandas as pd
import equadratures as eq
from utils import convert_latex

from app import app

###################################################################
# Setup cache (simple cache if locally run, otherwise configured
# to use memcachier on heroku)
###################################################################
cache_servers = os.environ.get('MEMCACHIER_SERVERS')
if cache_servers == None:
    # Fall back to simple in memory cache (development)
    cache = Cache(app.server,config={'CACHE_TYPE': 'SimpleCache'})
else:
    cache_user = os.environ.get('MEMCACHIER_USERNAME') or ''
    cache_pass = os.environ.get('MEMCACHIER_PASSWORD') or ''
    cache = Cache(app.server,
        config={'CACHE_TYPE': 'SASLMemcachedCache',
                'CACHE_MEMCACHED_SERVERS': cache_servers.split(','),
                'CACHE_MEMCACHED_USERNAME': cache_user,
                'CACHE_MEMCACHED_PASSWORD': cache_pass,
                'CACHE_OPTIONS': { 'behaviors': {
                    # Faster IO
                    'tcp_nodelay': True,
                    # Keep connection alive
                    'tcp_keepalive': True,
                    # Timeout for set/get requests
                    'connect_timeout': 2000, # ms
                    'send_timeout': 750 * 1000, # us
                    'receive_timeout': 750 * 1000, # us
                    '_poll_timeout': 2000, # ms
                    # Better failover
                    'ketama': True,
                    'remove_failed': 1,
                    'retry_timeout': 2,
                    'dead_timeout': 30}}})

###################################################################
# Collapsable more info card
###################################################################
info_text = r'''
Upload your data in *.csv* format using the **Load Data** card. Take note of the following:
- The data must in standard *wide-format* i.e. with each row representing an observation/sample. 
- There must be no NaN's or empty cells.
- For computational cost purposes datasets are currently capped at $N=600$ rows and $d=30$ input dimensions. For guidance on handling larger datasets checkout the [docs](https://equadratures.org/) or [get in touch](https://discourse.equadratures.org/).
- Particularly when higher polynomial orders are selected, monitor the test $R^2$ score to check for *over-fitting*.
- If the $R^2$ scores are low, try to vary the polynomial order and number of subspace dimensions.

'''

info = html.Div(
    [
    dbc.Button("More Information",color="primary",id="data-info-open",className="py-0"),
    dbc.Modal(
        [
            dbc.ModalHeader(dcc.Markdown('**More Information**')),
            dbc.ModalBody(dcc.Markdown(convert_latex(info_text),dangerously_allow_html=True)),
            dbc.ModalFooter(dbc.Button("Close", id="data-info-close", className="py-0", color='primary')),
        ],
        id="data-info",
        scrollable=True,size='lg'
    ),
    ]
)


###################################################################
# Load data card
###################################################################
data_select = dbc.Form(
    [
    dbc.Label('Select dataset',html_for='data-select'),
    dcc.Dropdown(id="data-select",
    options=
        [
        {"label": "Upload my own", "value":"upload"},
        {"label": "Temperature probes", "value":"probes"},
        {"label": "Fan blade A", "value":"fan_blades_a"},
        {"label": "Fan blade C", "value":"fan_blades_c"},
        ],
        value="upload",placeholder="Upload my own",clearable=False,searchable=False)
    ]
)

upload = dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select',style={'font-weight':'bold','color':'var(--primary)'}),
            ' CSV/Excel file'
        ]),
        # Don't allow multiple files to be uploaded
        multiple=False, disabled=False,
        style = {
                'width': '100%',
                'height': '50px',
                'lineHeight': '50px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '5px',
                }

    )

qoi_input = dbc.Row(
    [
    dbc.Col(
        dbc.FormGroup(
            [
                dbc.Label('Select output variable', html_for="qoi-select"),
                dcc.Dropdown(id="qoi-select",searchable=False)
            ],
        ),width=4
    ),
    dbc.Col(
        dbc.Form(
            [
            dbc.FormGroup(
                [
                dbc.Label('Number of input dimensions:', html_for="report-Xd",width=9),
                dbc.Col(html.Div(id='report-Xd'),width=3)
                ], row=True
            ),
            dbc.FormGroup(
                [
                dbc.Label('Number of output dimensions:', html_for="report-yd",width=9),
                dbc.Col(html.Div(id='report-yd'),width=3)
                ], row=True
            ),
            ]
        ), width=4
    ),
    dbc.Col(
        dbc.Form(
            [
            dbc.FormGroup(
                [
                dbc.Label('Number of training samples:', html_for="report-train",width=9),
                dbc.Col(html.Div(id='report-train'),width=3)
                ], row=True
            ),
            dbc.FormGroup(
                [
                dbc.Label('Number of test samples:', html_for="report-test",width=9),
                dbc.Col(html.Div(id='report-test'),width=3)
                ], row=True
            ),
            ]
        ), width=4
    ),

    ]
)

data_card = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown('**Load Data**')),
        dbc.CardBody(
            [
            dbc.Row(
                [
                dbc.Col(data_select,width=4),
                dbc.Col(upload,width=5),
                dbc.Col(
                    dbc.Alert('Error loading file!',id='upload-error',color='danger',is_open=False,style={'height':'40px','margin':'10px'}),
                width=3),
                ],align='start'
            ),
            dbc.Row(dbc.Col(
                dbc.Alert('Click bin icons to delete columns as necessary',id='upload-help',color='info',
                    is_open=False,dismissable=True,style={'margin-top':'0.4rem'}),
                width=5)
            ),
            dbc.Row(dbc.Col(
                dash_table.DataTable(data=[],columns=[],id='upload-data-table',
                style_table={'overflowX': 'auto','overflowY':'auto','height':'35vh'}, #change height to maxHeight to get table to only take up space when populated.
                editable=True,fill_width=False,page_size=20)
                ,width=12),style={'margin-top':'10px'}),
            dbc.Row(
                [
                dbc.Col(qoi_input,width=12),
                ]
            )
                
            # TODO - either summary of example dataset, or table of data metrics appear here
            ]
        )
    ]
)

###################################################################
# Settings card
###################################################################
method_dropdown = dbc.FormGroup(
    [
        dbc.Label('Method', html_for="method-select"),
        dcc.Dropdown(id="method-select",options=[
            {'label': 'Active Subspace', 'value': 'active-subspace'},
            {'label': 'Variable Projection', 'value': 'variable-projection'},
            ],
        value='active-subspace',clearable=False,searchable=False
        )
    ],
    id='method-select'
)

order_slider = dbc.FormGroup(
    [
        dbc.Label('Polynomial order', html_for="order-slider"),
        dcc.Slider(id='order-slider',min=1, max=3,value=1,
            tooltip = { 'always_visible': True, 'placement': 'bottom' }
        )
    ]
)

subdim_slider = dbc.FormGroup(
    [
        dbc.Label('Subspace dimensions', html_for="subdim-slider"),
        dcc.Slider(id='subdim-slider',min=1, max=3,value=1,
            tooltip = { 'always_visible': True, 'placement': 'bottom' }
        )
    ]
)

traintest_slider = dbc.FormGroup(
    [
        dbc.Label('Test split (%)', html_for="traintest-slider"),
        dcc.Slider(id='traintest-slider',min=0, max=50,value=0,
            tooltip = { 'always_visible': True, 'placement': 'bottom' }
        )
    ]
)

settings_card = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown('**Compute Subspace**')),
        dbc.CardBody(
            [
            dbc.Row(dbc.Col(method_dropdown)),
            dbc.Row(dbc.Col(order_slider)),
            dbc.Row(dbc.Col(subdim_slider)),
            dbc.Row(dbc.Col(traintest_slider)),
            dbc.Row(
                [
                dbc.Col(dbc.Button("Compute", id="compute-subspace", color="primary"),width='auto'),
                dbc.Col(dbc.Spinner(html.Div(id='compute-finished'),color="primary"),width=2),
                dbc.Col(dbc.Alert(id='compute-warning',color='danger',is_open=False),width='auto'),
                ], justify='start',align='center'
            ),
            dbc.Row(dbc.Col(dcc.Markdown(id='r2-train',dangerously_allow_html=True)),style={'margin-top':'10px'}),
            dbc.Row(dbc.Col(dcc.Markdown(id='r2-test',dangerously_allow_html=True))),
            ]
        )
    ]
)

###################################################################
# Results card
###################################################################
subspace_matrix_plot = dbc.Container(
    [
        dbc.Row(dbc.Col(
            dcc.Markdown(convert_latex(
                r'Weights in chosen column of $\mathbf{W}$ - These weights identify how the original variables must be altered in order to increase the corresponding *active variable* e.g. a negative weight implies the original variable must be decreased in order to increase the chosen *active variable*.'),
                dangerously_allow_html=True,style={'text-align':'justify','margin-top':'10px'}),
        )),
        dbc.Row(dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label(dcc.Markdown(convert_latex(r'Select $\mathbf{W}$ column to plot'),dangerously_allow_html=True),
                        html_for="qoi-select",width='auto'),
                    dbc.Col(dcc.Dropdown(id="W-select",value=0,placeholder='1',clearable=False,searchable=False),width='auto')
                ],row=True
            )
        )),
        dbc.Row(dbc.Col(
            dcc.Graph(figure={},id="data-W-plot",style={'height':'50vh','width':'inherit'})
        )),
    ],fluid=True
)

eigenvalue_plot = dbc.Container(
    [
        dbc.Row(dbc.Col(
            dcc.Markdown(convert_latex(
                r'Ideally, the number of *subspace dimensions* should be chosen such that is a large gap between the last eigenvalue of $\mathbf{\Lambda}_1$ (those corresponding to the *active dimensions*) and the first eigenvalue of $\mathbf{\Lambda}_2$ (those corresponding to the *inactive dimensions*).'),
                dangerously_allow_html=True,style={'text-align':'justify','margin-top':'10px'}),
        )),
        dbc.Row(dbc.Col(
            dcc.Graph(figure={},id="eigen-plot",style={'height':'50vh','width':'inherit'})
        )),

    ],fluid=True
)

subspace_msg = r'''
*.pickle* files containing `eq.Subspaces` and `eq.scalers` objects. The subspace can be used as described in the [docs](https://equadratures.org/_documentation/subspaces.html). 
Input data must first be transformed with the scaler object (see example).
'''

eg_msg = r'''
**Example**

```python
# Load eq objects
with open("subspace.pickle","rb") as f:
    subspace = pickle.load(f)
with open("scaler.pickle","rb") as f:
    scaler = pickle.load(f)

# Use subspace polynomial to predict for new data
subdim = subspace.subspace_dimension
W = subspace.get_subspace()[:,:subdim]
subpoly = subspace.get_subspace_polynomial()
Xnorm = scaler.transform(Xorig)
ypred = subpoly.get_polyfit(Xnorm @ W)
```
'''

downloads = dbc.Container(
    [
        dbc.Row(dbc.Col([dbc.Button("Download reduced data", 
            id="download-csv-button",color='primary',className='py-0',disabled=True),
            dcc.Download(id="download-csv")],width='auto'),style={'margin-top':'10px'}),
        dbc.Row(dbc.Col(dcc.Markdown('*.csv* file containing the training data in its reduced dimensional form.',style={'text-align':'justify'}),width='auto')),

        dbc.Row(
            [
                dbc.Col([dbc.Button("Download subspace", 
                    id="download-subspace-button",color='primary',className='py-0',disabled=True),
                    dcc.Download(id="download-subspace")],width='auto'),
                dbc.Col([dbc.Button("Download scaler", 
                    id="download-scaler-button",color='primary',className='py-0',disabled=True),
                    dcc.Download(id="download-scaler")],width='auto')
            ], style={'margin-top':'5px'}),
        dbc.Row(dbc.Col(dcc.Markdown(convert_latex(subspace_msg), dangerously_allow_html=True,style={'text-align':'justify'}),width='auto')),

        dbc.Row(dbc.Col(dcc.Markdown(convert_latex(eg_msg),dangerously_allow_html=True),width=12),className='my-3'),
    ],fluid=True
)

other_results_card = dbc.Card(dbc.Tabs(
    [
    dbc.Tab(subspace_matrix_plot,label='Subspace Matrix',    tabClassName='custom-tab',activeTabClassName='custom-tab--selected',
        labelClassName='custom-tablabel',activeLabelClassName='custom-tablabel--selected'),
    dbc.Tab(eigenvalue_plot,label='Eigenvalues', disabled=True, tabClassName='custom-tab',activeTabClassName='custom-tab--selected',
        labelClassName='custom-tablabel',activeLabelClassName='custom-tablabel--selected',id='eigen-tab'),
    dbc.Tab([downloads],label='Download', tabClassName='custom-tab',activeTabClassName='custom-tab--selected',
        labelClassName='custom-tablabel',activeLabelClassName='custom-tablabel--selected'),

    ],
    className='custom-tabs'
))


summary_card = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown('**Sufficient Summary Plot**')),
        dbc.CardBody(
            [
            dbc.Row(dbc.Col(
                dcc.Markdown('**Sufficient Summary Plot**',style={'text-align':'center'}),
            )),
            dbc.Row(dbc.Col(
                dcc.Markdown(convert_latex(
                    r'Visualises variation of the chosen output over its dimension reducing subspace $\mathbf{W}^T\mathbf{x}$.'),
                    dangerously_allow_html=True, style={'text-align':'left'}),
            )),
            dbc.Row(
                [
                dbc.Col('1D',width='auto'),
                dbc.Col(daq.ToggleSwitch(id='toggle-summary', value=False),width='auto'),
                dbc.Col('2D',width='auto'),
                ],justify='center'
            ),
            dbc.Row(dbc.Col(dcc.Markdown(id='toggle-text'),width='auto'),justify='center'),
            dbc.Row(dbc.Col(
                dcc.Graph(figure={},id="data-summary-plot", config={'responsive':True}, style={'height':'50vh','width':'40vw'}) 
            ,width='auto'),justify='center')
            ]
        )
    ],
)

###################################################################
# The overall app layout
###################################################################
layout = dbc.Container(
    [
    html.H2("Reduce Me!"),
    dbc.Row(
        [
            dbc.Col(dcc.Markdown('This app computes dimension reducing subspaces for your data! Upload your data, or choose an example dataset.'),width='auto'),
            dbc.Col(info,width='auto')
            ], align='center', style={'margin-bottom':'10px'}
    ),
    dbc.Row(
        [
        dbc.Col(data_card,width=9),
        dbc.Col(settings_card,width=3),
        ]
    ),
    dbc.Row(
        [
        dbc.Col(summary_card,width=7),
        dbc.Col(other_results_card,width=5)
        ],style={'margin-top':'10px'},
    ),
    dcc.Store(id='subspace-data'),
#    tooltips
    ],
    fluid = True
)


###################################################################
# Callbacks
###################################################################
# More info collapsable
@app.callback(
    Output("data-info", "is_open"),
    [Input("data-info-open", "n_clicks"), Input("data-info-close", "n_clicks")],
    [State("data-info", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Show upload interface callback
@app.callback(
    Output("upload-data", "disabled"),
    [Input("data-select", "value")],
)
def toggle_upload_box(option):
    if option == 'upload':
        return False
    else:
        return True

# Load csv file callback
@app.callback(Output('upload-data-table', 'data'),
        Output('upload-data-table', 'columns'),
        Output('upload-error', 'is_open'),
        Output('upload-help', 'is_open'),
        Input('upload-data', 'contents'),
        Input("data-select", "value"),
        State('upload-data', 'filename'),
        prevent_initial_call=True)
def load_csv(content, data_option, filename):
    df = None
    if data_option == 'upload':
        # Only load csv once button pressed
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'upload-data' in changed_id:
            # Parse csv
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename:
                    # Assume that the user uploaded a CSV file
                    df = pd.read_csv(
                        io.StringIO(decoded.decode('utf-8')))
                elif 'xls' in filename:
                    # Assume that the user uploaded an excel file
                    df = pd.read_excel(io.BytesIO(decoded))
            except Exception as e:
                print(e)
                return [],[],True,False
            bin_msg = True
    # temperature probe dataset
    elif data_option == 'probes':
        data = eq.datasets.load_eq_dataset('probes')
        data = np.hstack([data['X'],data['y2']])
        cols = ['Hole ellipse','Hole fwd/back','Hole angle','Kiel lip','Kiel outer','Kiel inner','Hole diam.','Recovery ratio objective']
        df = pd.DataFrame(data=data, columns=cols)
        bin_msg = False
    # Fan blade A
    elif data_option == 'fan_blades_a':
        data = eq.datasets.load_eq_dataset('3Dfan_blades')
        data = np.hstack([data['X_a'],data['y2_a'].reshape(-1,1)])
        cols = ['X_%d' %j for j in range(25)]
        cols.append('Pressure ratio')
        df = pd.DataFrame(data=data, columns=cols)
        bin_msg = False
    # Fan blade B
    elif data_option == 'fan_blades_c':
        data = eq.datasets.load_eq_dataset('3Dfan_blades')
        data = np.hstack([data['X_c'],data['y1_c'].reshape(-1,1)])
        cols = ['X_%d' %j for j in range(25)]
        cols.append('Efficiency')
        df = pd.DataFrame(data=data, columns=cols)
        bin_msg = False


    # Create a datatable
    if df is not None:
        data=df.to_dict('records')
        columns=[{'name': i, 'id': i, 'deletable': True,'type':'numeric','format':Format(precision=4,trim=Trim.yes)} for i in df.columns]
        return data,columns,False,bin_msg
    else:
        return [], [], False, False

# Populate qoi options
@app.callback(Output('qoi-select','options'),
        Output('qoi-select','value'),
        Input('upload-data-table', 'columns'),
        State("data-select", "value"),
        prevent_initial_call=True)
def populate_qoi(columns,data_option):
    if data_option == 'upload':
        options = [{'label': i['name'], 'value': i['name']} for i in columns]
        value = None
    else:
        output = columns[-1]['name']
        options = [{'label': output, 'value': output}]
        value = output
    return options, value

##################################################################
# Function to compute subspace
###################################################################
# This function is moderately time consuming so memoize it
@cache.memoize(timeout=600)
def compute_subspace_memoize(X_train, y_train, method, order, subdim):
    subspace = eq.Subspaces(method=method,sample_points=X_train,sample_outputs=y_train,
            polynomial_degree=order, subspace_dimension=subdim)
    return jsonpickle.encode(subspace)

# callback to compute subspace
@app.callback(Output('compute-finished','children'),
        Output('compute-warning','is_open'),
        Output('compute-warning','children'),
        Output('subspace-data','data'),
        Output('r2-train','children'),
        Output('r2-test','children'),
        Input('compute-subspace', 'n_clicks'),
        Input('upload-data-table', 'data'),
        Input('upload-data-table', 'columns'),
        Input('qoi-select','value'),
        Input('method-select','value'),
        Input('order-slider','value'),
        Input('subdim-slider','value'),
        Input('traintest-slider','value'),
        prevent_initial_call=True)
def compute_subspace(n_clicks,data,cols,qoi,method,order,subdim,test_split):
    # Compute subspace (if button has just been pressed)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'compute-subspace' in changed_id:
        # Check qoi selected
        if qoi is None:
            return None, True, 'Output variable not selected!', None, None, None
        else:
            # Parse data to dataframe
            df = pd.DataFrame.from_records(data)
            # Check for missing values and NaN
            problem = df.isnull().values.any() 
            if problem:
                return None, True, 'Missing/NaN values in data', None, None, None

            # Get X and y
            y = df.pop(qoi).to_numpy()
            X = df.to_numpy()
            
            # Check X and y dimensions
            N,d = X.shape
            if d <= subdim:
                return None, True, 'The subspace dimension must be less than the original dimensions', None, None, None

            # TODO - Scale data (TODO - and unscale somewhere)
            scaler = eq.scalers.scaler_minmax()
            X = scaler.transform(X)

            # Train/test split
            test_split /= 100
            X_train, X_test, y_train, y_test = eq.datasets.train_test_split(X, y,
                                   train=float(1-test_split),random_seed=42)
 
            # Compute subspace
            pickled = compute_subspace_memoize(X_train, y_train, method, order, subdim)
            subspace = jsonpickle.decode(pickled)

            # compute scores
            subpoly = subspace.get_subspace_polynomial()
            W = subspace.get_subspace()[:,:subdim]
            r2_train = eq.datasets.score(y_train, subpoly.get_polyfit(X_train @ W), metric='adjusted_r2', X=X_train)
            if y_test.size > 0:
                r2_test  = eq.datasets.score(y_test , subpoly.get_polyfit(X_test  @ W), metric='adjusted_r2', X=X_test )
                r2_train = convert_latex(r'Train $R^2$ score = %.3f' %r2_train)
                r2_test  = convert_latex(r'Test $R^2$ score = %.3f' %r2_test)
            else:
                r2_train = convert_latex(r'Train $R^2$ score = %.3f' %r2_train)
                r2_test = None

            # Package up result into dict fo return
            subspace.test_points = X_test
            subspace.test_outputs = y_test
            results = {'scaler':scaler,'subspace':subspace}
            results = jsonpickle.encode(results)

            # Return data
            return None, False, None, results, r2_train, r2_test

    return None, False, None, None, None, None

# callback to disable eigenvalue tab
@app.callback(Output('eigen-tab','disabled'),
        Input('method-select','value'))
def disable_eigentab(method):
    if method=='active-subspace':
        return False
    else:
        return True

# callback to disable summary toggle
@app.callback(Output('toggle-summary','disabled'),
        Output('toggle-summary','value'),
        Input('subdim-slider','value'))
def disable_summarytoggle(subdim):
    if subdim==1:
        return True, False
    else:
        return False, True

# callback to set 1d/2d toggle text
@app.callback(Output('toggle-text','children'),
        Input('toggle-summary','value'))
def disable_summarytoggle(twod):
    if twod:
        return ('Leading two *active variables* plotted')
    else:
        return ('Leading *active variables* plotted')

###################################################################
# Sufficient summary plot
###################################################################
@app.callback(Output('data-summary-plot', 'figure'),
    Input('subspace-data','data'),
    Input('qoi-select', 'value'),
    Input('toggle-summary', 'value'),
    )
def display_summary_plot(subspace_data,qoi,twod):
    if qoi is None: qoi = ''

    # Layout
    ########
    if twod: #2D
        layout=dict(margin={'t':0,'r':0,'l':0,'b':0,'pad':10},autosize=True,
            scene = dict(
                aspectmode='cube',
                xaxis = dict(
                    title='1st active variable',
                    gridcolor="white",
                    showbackground=False,
                    linecolor='black',
                    tickcolor='black',
                    ticks='outside',
                    zerolinecolor="white",),
                yaxis = dict(
                    title='2nd active variable',
                    gridcolor="white",
                    showbackground=False,
                    linecolor='black',
                    tickcolor='black',
                    ticks='outside',
                    zerolinecolor="white"),
                zaxis = dict(
                    title=qoi,
                    backgroundcolor="rgb(230, 230,200)",
                    gridcolor="white",
                    showbackground=False,
                    linecolor='black',
                    tickcolor='black',
                    ticks='outside',
                    zerolinecolor="white",),
            ),
        )
        fig = go.Figure(data=go.Scatter3d(x=[],y=[],z=[],mode='markers'),layout=layout)

    else: # 1D
        layout={"xaxis": {"title": r'$\mathbf{W}^T\mathbf{x}$'}, "yaxis": {"title": qoi},'margin':{'t':0,'r':0,'l':0,'b':60},
                'paper_bgcolor':'white','plot_bgcolor':'white','autosize':True}
        fig = go.Figure(layout=layout)
        fig.update_xaxes(color='black',linecolor='black',showline=True,tickcolor='black',ticks='outside')
        fig.update_yaxes(color='black',linecolor='black',showline=True,tickcolor='black',ticks='outside')

    if subspace_data is not None:
        # Parse data
        results = jsonpickle.decode(subspace_data)
        scaler = results['scaler']
        subspace = results['subspace']
        X_train = subspace.sample_points
        y_train = subspace.sample_outputs
        X_test = subspace.test_points
        y_test = subspace.test_outputs
        subdim = subspace.subspace_dimension
        W = subspace.get_subspace()[:,:subdim]
        subpoly = subspace.get_subspace_polynomial()

        if twod: # 2D summary plot
            # Training
            u_train = (X_train @ W)
            fig.add_trace(go.Scatter3d(x=u_train[:,0],y=u_train[:,1],z=y_train,mode='markers',name='Training samples',
                marker=dict(size=10,color="rgb(135, 206, 250)",opacity=0.6,line=dict(color='rgb(0,0,0)',width=1))
            ))
            # Test
            if y_test.size > 0:
                u_test = X_test @ W
                fig.add_trace(go.Scatter3d(x=u_test[:,0],y=u_test[:,1],z=y_test,mode='markers',name='Test samples',
                    marker=dict(size=10,color="rgb(144, 238, 144)",opacity=0.6,line=dict(color='rgb(0,0,0)',width=1))
                ))
            # Poly
            if subdim == 2:
                N = 20
                u1_poly = np.linspace(np.min(u_train[:,0]), np.max(u_train[:,0]), N)
                u2_poly = np.linspace(np.min(u_train[:,1]), np.max(u_train[:,1]), N)
                [u11, u22] = np.meshgrid(u1_poly, u2_poly)
                u1_vec = np.reshape(u11, (N*N, 1))
                u2_vec = np.reshape(u22, (N*N, 1))
                y_poly = subpoly.get_polyfit(np.hstack([u1_vec, u2_vec]))
                y_poly = np.reshape(y_poly, (N, N))
                color = np.zeros([N,N]) # horrible hack to set fixed surface color
                colorscale = [[0,'rgb(178,34,34)'],[1,'rgb(0,0,0)']]
                fig.add_trace(go.Surface(x=u11,y=u22,z=y_poly,opacity=0.5,name='Ridge profile',
                    surfacecolor=color,colorscale=colorscale,cmin=0,cmax=1,showscale=False))
#                fig.add_trace(go.Surface(x=u11,y=u22,z=y_poly,opacity=0.5,name='Ridge profile',
#                    surfacecolor=y_poly,showscale=True,
#                    contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True,
#                                  start=y_poly.min(),end=y_poly.max(),size=(y_poly.max()-y_poly.min())/(20))))
#
        else: # 1D summary plot
            # Plot training samples
            u_train = X_train @ W
            if subdim > 1: u_train = u_train[:,0]
            fig.add_trace(go.Scatter(x=u_train.flatten(),y=y_train,mode='markers',name='Training samples',
                marker=dict(color='LightSkyBlue',size=15,opacity=0.5,line=dict(color='black',width=1))
            ))
            # Plot test samples
            if y_test.size > 0:
                u_test = X_test @ W
                if subdim > 1: u_test = u_test[:,0]
                fig.add_trace(go.Scatter(x=u_test.flatten(),y=y_test,mode='markers',name='Test samples',
                    marker=dict(color='lightgreen',size=15,opacity=0.5,line=dict(color='black',width=1))
                ))
            # Plot poly
            if subdim == 1:
                u_poly = np.linspace(np.min(u_train)-0.25,np.max(u_train)+0.25,50)
                y_poly = subpoly.get_polyfit(u_poly.reshape(-1,1))
                fig.add_trace(go.Scatter(x=u_poly,y=y_poly.flatten(),mode='lines',name='Ridge profile',line_width=4,line_color='firebrick' ))

        # TODO - rescale X, and add this info to scatter points for user to hover over (if possible)
    return fig

###################################################################
# W plot
###################################################################
# callback to populate dropdown
@app.callback(Output('W-select','options'),
        Input('subdim-slider', 'value'),
        )
def populate_W_dropdown(subdim):
    options = []
    for j in range(subdim):
        options.append({'label': j+1, 'value': j})
    return options

# Plot callback
@app.callback(Output('data-W-plot', 'figure'),
        Input('subspace-data','data'),
        Input('W-select','value'),
        Input('upload-data-table', 'columns'),
        )
def display_W_plot(subspace_data,to_plot,cols):
        # layout
        layout={"xaxis": {"title": 'Input variables'}, "yaxis": {"title": r'$w_{%dj}$' %(to_plot+1)},'margin':{'t':0,'r':0,'l':0,'b':60},
                'paper_bgcolor':'white','plot_bgcolor':'white','autosize':True}
        fig = go.Figure(layout=layout)
        fig.update_xaxes(color='black',linecolor='black',showline=True,tickcolor='black',ticks='outside')
        fig.update_yaxes(color='black',linecolor='black',showline=True,tickcolor='black',ticks='outside',zerolinecolor='lightgrey')
    
        # Parse results
        if subspace_data is not None:
            results = jsonpickle.decode(subspace_data)
            subspace = results['subspace']
            w = subspace.get_subspace()[:,to_plot]
            names = [col['name'] for col in cols]
    
            # Plot
            fig.add_trace(go.Bar(x=names, y=w,marker_color='LightSkyBlue',
                    marker_line_width=2,marker_line_color='black'))
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',xaxis_tickangle=-30)

        return fig

###################################################################
# eigenvalue plot
###################################################################
# Plot callback
@app.callback(Output('eigen-plot', 'figure'),
        Input('subspace-data','data'),
        Input('method-select','value'),
        )
def display_eigen_plot(subspace_data,method):
        if method=='active-subspace':
            # layout
            layout={"xaxis": {"title": r'$i$'}, "yaxis": {"title": r'$\lambda_i$'},'margin':{'t':0,'r':0,'l':0,'b':60},
                    'paper_bgcolor':'white','plot_bgcolor':'white','autosize':True}
            fig = go.Figure(layout=layout)
            fig.update_xaxes(color='black',linecolor='black',showline=True,tickcolor='black',ticks='outside')
            fig.update_yaxes(color='black',linecolor='black',showline=True,tickcolor='black',ticks='outside',zerolinecolor='lightgrey')
    
            # Parse results
            if subspace_data is not None:
                results = jsonpickle.decode(subspace_data)
                subspace = results['subspace']
                lambdas = subspace.get_eigenvalues()
                i = np.arange(len(lambdas)) + 1
    
                # Plot
                fig.add_trace(go.Scatter(x=i,y=lambdas,mode='markers',name='Test samples',
                    marker=dict(color='LightSkyBlue',size=15,opacity=0.5,line=dict(color='black',width=1))
                ))
            
            return fig
        else:
            return {}

###################################################################
# Download callbacks
###################################################################
# disable buttons until subspace computed
@app.callback(
    Output("download-csv-button", "disabled"),
    Output("download-subspace-button", "disabled"),
    Output("download-scaler-button", "disabled"),
    Input('subspace-data', 'data'), #TODO - is there a better way to do this which avoids passing entire dataset? Could just base on compute-subspace click, but then download buttons become active even if compute failed
    prevent_initial_call=True,
)
def disable_downloads(subspace_data):
    if subspace_data is None:
        return True, True, True
    else:
        return False,False,False

# CSV data file
@app.callback(
    Output("download-csv", "data"),
    Input("download-csv-button", "n_clicks"),
    State("subspace-data",'data'),
    State("qoi-select",'value'),
    prevent_initial_call=True,
)
def download_csv(n_clicks,subspace_data,qoi):
    # Parse data
    results = jsonpickle.decode(subspace_data)
    subspace = results['subspace']
    X_train = subspace.sample_points
    y_train = subspace.sample_outputs
    subdim = subspace.subspace_dimension
    W = subspace.get_subspace()[:,:subdim]
    u_train = X_train@W

    # Build dataframe
    data = np.hstack([u_train,y_train.reshape(-1,1)])
    cols = ['active dim. %d' %j for j in range(subdim)]
    cols.append(qoi)
    df = pd.DataFrame(data=data, columns=cols)

    return dcc.send_data_frame(df.to_csv, "reduced_results.csv")

# subspace pickle file
@app.callback(
    Output("download-subspace", "data"),
    Input("download-subspace-button", "n_clicks"),
    State("subspace-data",'data'),
    prevent_initial_call=True,
)
def download_subspace(n_clicks,subspace_data):
    # Parse data
    results = jsonpickle.decode(subspace_data)
    subspace = results['subspace']
    # Pickle
    obj = pickle.dumps(subspace)
    return dcc.send_bytes(obj,'subspace.pickle')

# subspace scaler file
@app.callback(
    Output("download-scaler", "data"),
    Input("download-scaler-button", "n_clicks"),
    State("subspace-data",'data'),
    prevent_initial_call=True,
)
def download_scaler(n_clicks,subspace_data):
    # Parse data
    results = jsonpickle.decode(subspace_data)
    scaler = results['scaler']
    # Pickle
    obj = pickle.dumps(scaler)
    return dcc.send_bytes(obj,'scaler.pickle')

###################################################################
# Limit data size
###################################################################
# Callback to limit rows
@app.callback(Output('compute-subspace','disabled'),  
    Output('report-train','children'),
    Output('report-test','children'),
    Output('report-train','style'),
    Output('report-test','style'),
    Output('report-Xd','children'),
    Output('report-yd','children'),
    Output('report-Xd','style'),
    Output('report-yd','style'),
    Input('traintest-slider','value'),
    Input('upload-data-table', 'data'),
    Input('upload-data-table', 'columns'),
    Input('qoi-select','value'),
    prevent_initial_call=True)
def check_size(test_split,data,cols,qoi):
    MAX_ROWS = 600
    MAX_D = 30

    # Number of rows
    N = len(data)
    test_split = float(test_split/100)
    Ntest = math.ceil(N*test_split)
    Ntrain = N-Ntest

    # Number of dims
    d = len(cols)
    if qoi is None:
        Xd = d
        yd = 0
    else:
        Xd = d-1
        yd = 1

    # check dataset size
    toobig = False
    Ncolor = 'black'
    Dcolor = 'black'
    if Ntrain > MAX_ROWS: 
        toobig = True
        Ncolor = 'red'
    if d > MAX_D: 
        toobig = True
        Dcolor = 'red'

    return toobig, str(Ntrain), str(Ntest), {'color':Ncolor}, {'color':'black'}, str(Xd), str(yd), {'color':Dcolor}, {'color':'black'}
