Drive-fit API
=============

See :doc:`drive-fit` for the fit, commit, publication and Tlaloc workflows.

Fit workflow
------------

.. autoclass:: tolteca_kids.drive_fit.DriveFit
   :special-members: __call__

.. autoclass:: tolteca_kids.drive_fit.DriveFitStep

.. autoclass:: tolteca_kids.drive_fit.DriveFitPlotStep

.. autoclass:: tolteca_kids.drive_fit.DriveFitOutputStep

.. autoclass:: tolteca_kids.drive_fit.DriveFitResult
   :members: table, n_success

.. autoclass:: tolteca_kids.drive_fit.DriveFitViz
   :special-members: __call__

.. autoclass:: tolteca_kids.drive_fit.DriveFitGridViz
   :special-members: __call__

.. autopydantic_model:: tolteca_kids.drive_fit.DriveFitSweepArtifacts
   :class-doc-from: class
   :special-members: __call__
   :members: write

.. autoclass:: tolteca_kids.drive_fit.DriveFitArtifacts

.. autoclass:: tolteca_kids.drive_fit.DriveFitDataArtifact

.. autoclass:: tolteca_kids.drive_fit.DriveFitTableArtifact

.. autopydantic_model:: tolteca_kids.drive_fit.DriveFitConfig
   :class-doc-from: class
   :model-show-json: False

.. autopydantic_model:: tolteca_kids.drive_fit.DriveFitStepConfig
   :class-doc-from: class
   :model-show-json: False

.. autopydantic_model:: tolteca_kids.drive_fit.DriveFitPlotStepConfig
   :class-doc-from: class
   :model-show-json: False

.. autofunction:: tolteca_kids.drive_fit.prepare_drive_fit_inputs

.. autofunction:: tolteca_kids.drive_fit.make_drive_fit_product_index

.. autofunction:: tolteca_kids.drive_fit.write_drive_fit_product

.. autofunction:: tolteca_kids.drive_fit.open_drive_fit_product

.. autofunction:: tolteca_kids.drive_fit.read_drive_fit_context

.. autofunction:: tolteca_kids.drive_fit.run

.. autofunction:: tolteca_kids.drive_fit.find_best_drive

Commit workflow
---------------

.. autoclass:: tolteca_kids.drive_fit_commit.DriveFitCommit
   :special-members: __call__

.. autoclass:: tolteca_kids.drive_fit_commit.DriveFitCommitStep

.. autoclass:: tolteca_kids.drive_fit_commit.DriveFitCommitPlotStep

.. autoclass:: tolteca_kids.drive_fit_commit.DriveFitCommitResult
   :members: table, n_success

.. autoclass:: tolteca_kids.drive_fit_commit.DriveFitCommitViz
   :special-members: __call__

.. autopydantic_model:: tolteca_kids.drive_fit_commit.DriveFitCommitConfig
   :class-doc-from: class
   :model-show-json: False

.. autopydantic_model:: tolteca_kids.drive_fit_commit.DriveFitCommitStepConfig
   :class-doc-from: class
   :model-show-json: False

.. autopydantic_model:: tolteca_kids.drive_fit_commit.DriveFitCommitPlotStepConfig
   :class-doc-from: class
   :model-show-json: False

.. autofunction:: tolteca_kids.drive_fit_commit.prepare_drive_fit_commit_inputs

.. autofunction:: tolteca_kids.drive_fit_commit.make_drive_fit_commit_product_index

.. autofunction:: tolteca_kids.drive_fit_commit.open_drive_fit_commit_product

.. autofunction:: tolteca_kids.drive_fit_commit.compute_ampcor

.. autofunction:: tolteca_kids.drive_fit_commit.run

Tlaloc output
-------------

.. autoclass:: tolteca_kids.tlaloc_output.TlalocOutput

.. autoclass:: tolteca_kids.tlaloc_output.TlalocSettings

.. autoclass:: tolteca_kids.tlaloc_output.TlalocSettingsArtifact

.. autopydantic_model:: tolteca_kids.tlaloc_output.TlalocOutputConfig
   :class-doc-from: class
   :model-show-json: False


Artifact and execution types
----------------------------

.. autoclass:: tolteca_kids.drive_fit.plot.DriveFitPlotArtifacts

.. autoclass:: tolteca_kids.drive_fit.step.DriveFitStepContext

.. autoclass:: tolteca_kids.drive_fit_commit.artifacts.DriveFitCommitArtifacts

.. autoclass:: tolteca_kids.drive_fit_commit.artifacts.DriveFitCommitTableArtifact

.. autoclass:: tolteca_kids.drive_fit_commit.plot.DriveFitCommitPlotArtifacts

.. autoclass:: tolteca_kids.match1d.Match1D
