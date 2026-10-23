KIDs reducer pipeline API
=========================

See :doc:`kids-reduction` for Python and command-line workflows, run-name selection,
tag history and promotion of saved results. Steps update a
working DataTree and publish live result views through ``context.data`` and
presentation objects through ``context.artifacts``.
Calling, aliases and live context lookup follow :class:`tollan.pipeline.Step`.
The inherited ``enabled`` and ``alias`` settings are described by
:class:`tollan.pipeline.StepConfig`; ``config``, ``completed`` and ``failures``
are described by :class:`tollan.pipeline.StepContext`.

Configured block workflows
--------------------------

.. autoclass:: tolteca_kids.core.Kids
   :members: sweep_pipeline, timestream_pipeline, pipeline_for

Acquisition execution
---------------------

.. automodule:: tolteca_kids.runner
   :members: create_kids, run_data, run_input, run_inputs, run_source, run_sources, publish_quicklook

Reduction steps
---------------

.. autoclass:: tolteca_kids.block_identify.BlockIdentify
   :members: run

.. autopydantic_model:: tolteca_kids.block_identify.BlockIdentifyContext
   :class-doc-from: class
   :model-show-json: False

.. autoclass:: tolteca_kids.block_create.BlockCreate
   :members: run

.. autopydantic_model:: tolteca_kids.block_create.BlockCreateContext
   :class-doc-from: class
   :model-show-json: False

.. autoclass:: tolteca_kids.sweep_reduce.SweepReduce
   :members: run, view, result_path

.. autopydantic_model:: tolteca_kids.config.SweepReduceConfig
   :class-doc-from: class

.. autopydantic_model:: tolteca_kids.sweep_reduce.SweepReduceContext
   :class-doc-from: class
   :model-show-json: False

.. autoclass:: tolteca_kids.timestream_reduce.TimestreamReduce
   :members: run, view

.. autopydantic_model:: tolteca_kids.config.TimestreamReduceConfig
   :class-doc-from: class

.. autopydantic_model:: tolteca_kids.timestream_reduce.TimestreamReduceContext
   :class-doc-from: class
   :model-show-json: False

.. Live contexts contain Python views and a producing class, which have no JSON
   validation schema. The product functions document their metadata exports.

Sweep diagnostics
-----------------

.. autoclass:: tolteca_kids.sweep_check.SweepCheck
   :members: run, view, result_path, find_spike, make_chunks, make_data_mask_from_unified

.. autopydantic_model:: tolteca_kids.sweep_check.SweepCheckConfig
   :class-doc-from: class

.. autopydantic_model:: tolteca_kids.sweep_check.SweepCheckContext
   :class-doc-from: class
   :model-show-json: False

.. autoclass:: tolteca_kids.sweep_check.SweepBitMask
   :members:

.. autoclass:: tolteca_kids.sweep_check_plot.SweepCheckPlot
   :members: run

.. autopydantic_model:: tolteca_kids.sweep_check_plot.SweepCheckPlotConfig
   :class-doc-from: class

.. autopydantic_model:: tolteca_kids.sweep_check_plot.SweepCheckPlotContext
   :class-doc-from: class
   :model-show-json: False

Sweep visualizations and artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tolteca_kids.sweep_check_viz

.. autopydantic_model:: tolteca_kids.sweep_check_viz.ChannelSummaryViz
   :class-doc-from: class
   :special-members: __call__

.. autopydantic_model:: tolteca_kids.sweep_check_viz.S21FrequencyViz
   :class-doc-from: class
   :special-members: __call__

.. autopydantic_model:: tolteca_kids.sweep_check_viz.S21FrequencyGridViz
   :class-doc-from: class
   :special-members: __call__

.. autopydantic_model:: tolteca_kids.sweep_check_viz.IQGridViz
   :class-doc-from: class
   :special-members: __call__

.. autopydantic_model:: tolteca_kids.sweep_check_viz.SweepCheckArtifacts
   :class-doc-from: class
   :special-members: __call__
   :members: write, show

Tone preparation and Tlaloc files
----------------------------------

.. autopydantic_model:: tolteca_kids.kids_find.KidsFindConfig
   :class-doc-from: class

.. autoclass:: tolteca_kids.kids_find_plot.KidsFindPlot
   :members: run

.. autopydantic_model:: tolteca_kids.kids_find_plot.KidsFindPlotConfig
   :class-doc-from: class

.. autopydantic_model:: tolteca_kids.kids_find_plot.KidsFindPlotContext
   :class-doc-from: class
   :model-show-json: False

.. autopydantic_model:: tolteca_kids.kids_find_plot.KidsFindPlotArtifacts
   :class-doc-from: class
   :members: write

.. autopydantic_model:: tolteca_kids.kids_find_artifacts.KidsFindArtifacts
   :class-doc-from: class
   :members: write

.. autopydantic_model:: tolteca_kids.kids_find_artifacts.KidsFindTableArtifact
   :class-doc-from: class
   :members: write

.. autopydantic_model:: tolteca_kids.peaks1d.Peaks1D
   :class-doc-from: class

.. autoclass:: tolteca_kids.tone_prepare.TonePrepare
   :members: run

.. autopydantic_model:: tolteca_kids.tone_prepare.TonePrepareConfig
   :class-doc-from: class

.. autopydantic_model:: tolteca_kids.tone_prepare.TonePrepareContext
   :class-doc-from: class
   :model-show-json: False

.. autopydantic_model:: tolteca_kids.tone_prepare.TonePhasesRandom
   :class-doc-from: class
   :special-members: __call__

.. autopydantic_model:: tolteca_kids.tone_prepare.TonePhasesFromFile
   :class-doc-from: class
   :special-members: __call__

.. autopydantic_model:: tolteca_kids.tone_prepare.PlaceHolderTones
   :class-doc-from: class
   :special-members: __call__

.. autopydantic_model:: tolteca_kids.tone_prepare.TonePrepareArtifacts
   :class-doc-from: class
   :members: write

.. autofunction:: tolteca_kids.tone_prepare.prepare_tone_table

.. autoclass:: tolteca_kids.tlaloc_output.TlalocTones

.. autoclass:: tolteca_kids.tlaloc_output.TlalocToneArtifact
   :members: write

The shared :class:`tolteca_kids.tlaloc_output.TlalocOutput` step and its
configuration are documented in :doc:`drive-fit-api`.

Acquisition feedback
--------------------

.. automodule:: tolteca_kids.quicklook
   :members:

Scientific views
----------------

.. autoclass:: tolteca_kids.blocks.ToltecBlocksView
   :members:

.. autoclass:: tolteca_kids.blocks.ToltecBlockView
   :members:

.. autoclass:: tolteca_kids.blocks.ToltecBlocksMapper

.. autoclass:: tolteca_kids.sweep.ReducedSweepView
   :members:

.. autoclass:: tolteca_kids.timestream.TimestreamView
   :members:

.. autoclass:: tolteca_kids.sweep_check_view.SweepCheckView
   :members:

.. autoclass:: tolteca_kids.sweep_check_view.SweepCheckMapper

.. autoclass:: tolteca_kids.sweep_check_view.SweepCheckSchema

Publication and saved metadata
------------------------------

.. autoclass:: tolteca_kids.dataprod_output.DataProdOutput
   :members: run

.. autopydantic_model:: tolteca_kids.config.DataProdOutputConfig
   :class-doc-from: class

.. autopydantic_model:: tolteca_kids.dataprod_output.DataProdOutputContext
   :class-doc-from: class
   :model-show-json: False

.. automodule:: tolteca_datamodels.data_prod_store
   :members:

.. automodule:: tolteca_datamodels.data_prod_namespace
   :members:

.. automodule:: tolteca_datamodels.data_prod_run
   :members:

.. automodule:: tolteca_datamodels.data_prod
   :members:

.. automodule:: tolteca_kids.product
   :members:

Supporting configuration and metadata
-------------------------------------

.. autopydantic_model:: tolteca_kids.config.KidsConfig
   :class-doc-from: class

.. autopydantic_model:: tolteca_kids.config.InputConfig
   :class-doc-from: class

.. currentmodule:: None

.. py:class:: tolteca_datamodels.toltec.metadata.ToltecRawObsBlocksMetadata

   Acquisition metadata, discovered block count, ``block_index``, ``block_index_table``,
   ``chan_axis_table`` and ``probe_axis_table``. Available through
   ``ds.toltec_kids.block_index.meta`` before block creation, and through
   the prepared block collection's ``meta`` afterwards.

.. py:class:: tolteca_datamodels.toltec.metadata.ToltecRawObsBlockMetadata

   Acquisition fields and scalar geometry for one recorded block.

.. autoclass:: tolteca_datamodels.toltec.kids.blocks.ToltecBlockIndex

.. autofunction:: tolteca_kidsproc.reduction.reduce_sweep_samples

.. py:class:: tolteca_datamodels.toltec.metadata.ToltecRawObsMetadata

   Acquisition metadata available through ``ds.toltec_kids.meta``. Product
   publication reads this metadata from the supplied working tree. See
   :doc:`kids-reduction` for native Dataset and block access.

.. py:class:: tolteca_datamodels.toltec.metadata.ToltecSweepMetadata

   Sweep-block metadata available through ``view.meta``, including the inherited
   acquisition fields. See :doc:`kids-reduction` for block selection and axis
   tables.

.. py:class:: tolteca_datamodels.toltec.metadata.ToltecTimestreamMetadata

   Timestream-block metadata, including the inherited acquisition and block fields.

.. autoclass:: tolteca_datamodels.toltec.types.ToltecDataKind
