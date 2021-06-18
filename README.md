# eq-flow

This is a python [Dash](https://plotly.com/dash/) app, accesible at [reduce.ascillitoe.com](http://reduce.ascillitoe.com/). It demonstrates a number of use cases for the data-driven dimension reduction capability in [equadratures](https://equadratures.org/). Two sub-apps are currently available, one demonstrates the use of dimension reducing ridge functions for rapid flowfield estimation and design exploration of an airfoil, whilst the other allows you to upload your own datasets and perform data-driven dimension reduction.

The app is hosted on the cloud via a [Heroku](https://www.heroku.com/about) dyno. Memory-caching is performed with [Flask-Caching](https://flask-caching.readthedocs.io/en/latest/), [pylibmc](https://pypi.org/project/pylibmc/), and [MemCachier](https://www.memcachier.com/).
