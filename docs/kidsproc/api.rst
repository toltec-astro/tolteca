KIDs processing API
===================

Construction and views
----------------------

``KidsDataset`` and ``KidsDataTree`` are static typing aids. At runtime the
constructors return ordinary xarray containers with the ``kids`` accessor.

.. autofunction:: tolteca_kidsproc.accessors.dataset.make_kids_dataset

.. autofunction:: tolteca_kidsproc.accessors.dataset.open_datatree

.. automodapi:: tolteca_kidsproc.accessors.kids

.. automodapi:: tolteca_kidsproc.accessors.views

.. _kidsproc-analysis-models:

Analysis models
---------------

.. autopydantic_model:: tolteca_kidsproc.analysis.d21.D21Analysis
   :class-doc-from: class
   :special-members: __call__

.. autopydantic_model:: tolteca_kidsproc.analysis.psd.PSDAnalysis
   :class-doc-from: class
   :special-members: __call__

Result mappings and views
-------------------------

.. automodapi:: tolteca_kidsproc.analysis.d21
   :skip: D21Analysis

.. automodapi:: tolteca_kidsproc.analysis.psd
   :skip: PSDAnalysis

.. autodata:: tolteca_kidsproc.analysis.d21.D21SmoothMethod
