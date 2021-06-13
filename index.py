import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app, server
from apps import flowfield, user_data
from navbar import navbar
from utils import convert_latex

###################################################################
# Homepage text
###################################################################
# Below is a pretty horrendous hack (see apps/flowfield.py for a much cleaner approach). The markdown sections are split here to get the equations nicely centered (can't use divs etc due to convert_latex function). Hopefully one day dash will sort out their latex support.

home_text = r'''
## Data-Driven Dimension Reduction

The apps contained in these pages utilise the [*equadratures*](https://equadratures.org/) data-driven dimension reduction capability for a number of tasks. In **Flowfield Estimation**, dimension reducing ridges are embedded within an airfoil flow for rapid flowfield estimation and design exploration, whilst in **My Data**, you can apply the techniques to your own datasets. But first, the underlying ideas are briefly introduced. 

Many physical systems are high dimensional, which can make it challenging to obtain approximations of them, and it often even more challenging to visualise these approximations. However, all is not lost! Many seemingly high-dimensional systems have intrinsically low-dimensional structure. Although the quantity of interest $f(\mathbf{x})$ might be defined as a function of a large set of parameters $\mathbf{x} \in \mathbb{R}^d$, its variation can often be approximately captured with a small number of linear projections of the original parameters $\mathbf{W}^T \mathbf{x} \in \mathbb{R}^n$. Here, $n \ll d$ and $\mathbf{W} \in \mathbb{R}^{d\times n}$ is a tall matrix whose column span is called the *dimension reducing subspace*, or *active subspace*. 

In other words, we assume that our quantities of interest are well--approximated by *ridge functions*,

$$f(\mathbf{x}) \approx g(\mathbf{W}^T \mathbf{x}),$$

where $g:\mathbb{R}^n \rightarrow \mathbb{R}$ is a low-dimensional non-linear function called the *ridge profile*. The intuitive computational advantage behind *ridge approximations* is that instead of estimating a function in $\mathbb{R}^d$, we approximate it in $\mathbb{R}^n$, which also facilitates easy visualisation. The *equadratures* code uses orthogonal polynomials to represent $g$, and so identifying ridge functions consists of computing coefficients for $g$ as well as identifying a suitable subspace matrix $\mathbf{W}$. Two techniques are avaiable for this in the code, *active subspaces* and *variable projection*.

#### Active Subspaces
The active subspaces approach, introduced in \[1], involves estimating a covariance matrix using the gradient of a polynomial approximation

$$\mathbf{C} = \int_{\mathcal{X}} \nabla_{\mathbf{x}} g(\mathbf{x}) \nabla_{\mathbf{x}} g(\mathbf{x})^T\,\tau~d\mathbf{x},$$

where $\tau = 2^{-d}$ defines a uniform distribution over the $d$-dimensional hypercube $\mathcal{X}$. As $\mathbf{C}$ is a symmetrix matrix, one can write its eigendecomposition as

$$\mathbf{C} = [\mathbf{W} \; \mathbf{V}] \begin{bmatrix} \mathbf{\Lambda}_1 & \mathbf{0}\\ \mathbf{0} &\mathbf{\Lambda}_2 \end{bmatrix} \begin{bmatrix} \mathbf{W}^T\\ \mathbf{V}^T \end{bmatrix},$$
    
where $\mathbf{\Lambda}_1$ is a diagonal matrix containing the largest eigenvalues, and $\mathbf{\Lambda}_2$ the smallest eigenvalues, both sorted in descending order. This partition should be chosen such that there is a large gap between the last eigenvalue of $\mathbf{\Lambda}_1$ and the first eigenvalue of $\mathbf{\Lambda}_2$ \[1]. Thus, this partitioning of $\mathbf{Q}$ yields the active subspace matrix $\mathbf{W}$ and the inactive subspace matrix $\mathbf{V}$.

An example of this method in action is given below, where a $n=1$ dimensional approximation is obtained for the $d=7$ temperature probe dataset available from the [equadratures dataset repository](https://github.com/Effective-Quadratures/data-sets). 

```python
import equadratures as eq
# Load the probe data, standardise to -1/1, and split into train/test
data = eq.datasets.load_eq_dataset('probes')
X = data['X']; y = data['y1']
X = eq.scaler_minmax().transform(X)
X_train, X_test, y_train, y_test = eq.datasets.train_test_split(X, y,train=0.8, random_seed = 42)
N,d = X_train.shape

# Apply active subspace method
subdim = 1
subspace = eq.Subspaces(method='variable-projection',sample_points=X_train,
                            sample_outputs=y_train,polynomial_degree=2, subspace_dimension=subdim)
W = subspace.get_subspace()[:,0:subdim]
u_test = (X_test@W).reshape(-1,1)
subpoly = subspace.get_subspace_polynomial()
print('Ridge Poly. R2 score = %.3f' %eq.datasets.score(y_test,subpoly.get_polyfit(u_test),metric='r2'))
```

#### Variable Projection

In order to construct the $\mathbf{C}$ matrix, we must first obtain $g(\mathbf{x})$, a polynomial fitted in the full $d$-dimensional space. This becomes problematic as the number of dimensions is increased. For such circumstances, *equadratures* offers *variable projection* \[2]. The non-linear least squares problem


$$\underset{\mathbf{W}, \boldsymbol{\alpha}}{\text{minimize}} \; \; \left\Vert f\left(\mathbf{x}\right)-g_{\boldsymbol{\alpha}}\left(\mathbf{W}^{T} \mathbf{x}\right)\right\Vert _{2}^{2}$$
    
is solved by recasting it as a separable non-linear least squares problem. Here, Gauss-Newton optimization used to solve for the polynomial coefficients $\alpha$ and subspace matrix $\mathbf{W}$ together. 

#### References
\[1]: P. Constantine. "Active subspaces : emerging ideas for dimension reduction in parameter studies". *SIAM Spotlights* (2015). [Book](https://doi.org/10.1137/1.9781611973860). 

\[2]: J. Hokanson and P. Constantine. "Data-Driven Polynomial Ridge Approximation Using Variable Projection". *SIAM Journal of Scientific Computing* (2017). [Paper](https://doi.org/10.1137/17M1117690).
'''

home_text = dcc.Markdown(convert_latex(home_text), dangerously_allow_html=True, style={'text-align':'justify'})

# disclaimer message
final_details = r'''
This app is currently hosted *on the cloud* via a free [Heroku dyno](https://www.heroku.com/dynos). It can take a while to boot up, and can be slow when there are multiple users. If it is too slow please do come back later! 

Please report any bugs to [ascillitoe@effective-quadratures.org](mailto:ascillitoe@effective-quadratures.org).
'''
final_details = dbc.Alert(dcc.Markdown(final_details),
        dismissable=True,is_open=True,color='info',style={'padding-top':'0.4rem','padding-bottom':'0.0rem'})


###################################################################
# Footer
###################################################################
footer = html.Div(
        [
            html.P('App built by Ashley Scillitoe'),
#            html.A(html.P('ascillitoe.com'),href='https://ascillitoe.com'),
            html.P(html.A('ascillitoe.com',href='https://ascillitoe.com')),
            html.P('Copyright © 2021')
        ]
    ,className='footer'
)

###################################################################
# App layout (adopted for all sub-apps/pages)
###################################################################
homepage = dbc.Container([home_text,final_details])

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        navbar,
        html.Div(homepage,id="page-content"),
        footer,
    ],
    style={'padding-top': '70px'}
)

###################################################################
# Callback to return page requested in navbar
###################################################################
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

###################################################################
# Run server
###################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
