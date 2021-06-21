# eq-flow

This is a python [Dash](https://plotly.com/dash/) app, accesible at [reduce.ascillitoe.com](https://reduce.ascillitoe.com/). It demonstrates a number of use cases for the data-driven dimension reduction capability in [equadratures](https://equadratures.org/). Two sub-apps are currently available, one demonstrates the use of dimension reducing ridge functions for rapid flowfield estimation and design exploration of an airfoil, whilst the other allows you to upload your own datasets and perform data-driven dimension reduction.

The app is hosted on the cloud via a [Heroku](https://www.heroku.com/about) dyno. 

### Installation
The easiest way to run the app is to simply go to [reduce.ascillitoe.com](https://reduce.ascillitoe.com/)! 

Alternatively, if you want to run locally, you must first install the full dash stack, and a number of other packages. The full list of requirements for local running can be installed with:

```console
python -m pip install dash dash_core_components dash_html_components dash_daq dash-extensions==0.0.57
python -m pip install plotly numpy pandas equadratures>=9.1.0 func-timeout requests>=2.11.1
```

I recommend performing the above in a virtual enviroment such as virtualenv. After installing requirements, the app can be launched with:

```console
python index.py
```
