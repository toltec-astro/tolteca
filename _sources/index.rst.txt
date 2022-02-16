*********************
TolTECA Documentation
*********************

``tolteca`` is a Python package developed to handle data taken by the TolTEC
Camera on the Large Millimeter Telescope.

While TolTEC is its main target instrument, the infrastructure and generic workflow
are implemented in a instrument agnostic fashion, which allows the functionalities
to be extendable to other instruments.


Getting Started
===============

.. todo::
    Add details.


Using ``tolteca``
=================

Overview of Workflow
--------------------

* RuntimeContext and RuntimeBase

* Commandline Interface

* Programming Interface

Data File and Data Product Management
-------------------------------------

Data Reduction Pipeline
-----------------------

Observation Simulator
---------------------

Web-based Tools
---------------

.. toctree::
   :maxdepth: 2

   tolteca/web

Recipes
-------

.. toctree::
   :maxdepth: 2

   tolteca/recipes

Scripts
-------

Reference/API
=============

.. automodapi:: tolteca
.. automodapi:: tolteca.cli
.. automodapi:: tolteca.common.toltec
.. automodapi:: tolteca.common.lmt
.. automodapi:: tolteca.datamodels.toltec
.. automodapi:: tolteca.datamodels.fs
.. automodapi:: tolteca.datamodels.db
.. automodapi:: tolteca.datamodels.io
.. automodapi:: tolteca.reduce
.. automodapi:: tolteca.reduce.engines
.. automodapi:: tolteca.reduce.toltec
.. automodapi:: tolteca.simu
.. automodapi:: tolteca.simu.base
.. automodapi:: tolteca.simu.mapping
.. automodapi:: tolteca.simu.sources
.. automodapi:: tolteca.simu.exports
.. automodapi:: tolteca.simu.plots
.. automodapi:: tolteca.simu.toltec
.. automodapi:: tolteca.utils
.. automodapi:: tolteca.web
.. automodapi:: tolteca.web.apps
