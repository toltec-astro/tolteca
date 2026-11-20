TolTECA documentation
=====================

TolTECA supports observation inspection, input selection, detector processing,
and reusable scientific products. Read the :doc:`README <readme>` for installation
and a quick start, then choose the domain for your task.

Choose a domain
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Domain
     - What you can do
     - Start here
   * - Observation data
     - Read acquisitions, compare metadata, search data trees, and discover inputs.
     - :doc:`datamodels`
   * - Data products
     - Interpret products, items and indexes.
     - :doc:`data-products-api`
   * - Detector processing
     - Select KIDs inputs, reduce data, and generate quicklook feedback.
     - :doc:`kids-inputs`, :doc:`kids-reduction`
   * - Signal processing
     - Work directly with KIDs arrays, derivatives, and spectra.
     - :doc:`kidsproc/index`

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   readme

.. toctree::
   :maxdepth: 1
   :caption: Observation data

   datamodels
   read-acquisition
   file-tables
   file-stores
   input-discovery
   obs_spec
   datamodels-api

.. toctree::
   :maxdepth: 1
   :caption: Framework architecture (for developers)

   datamodels-architecture

.. toctree::
   :maxdepth: 1
   :caption: Data products

   data-products

.. toctree::
   :maxdepth: 1
   :caption: Detector processing

   kids-inputs
   kids-reduction
   kids-model-reports

.. toctree::
   :maxdepth: 1
   :caption: KIDs signal processing

   kidsproc/index
   kidsproc/d21
   kidsproc/psd
   kidsproc/api

.. toctree::
   :maxdepth: 1
   :caption: Shared reference

   terminology

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
