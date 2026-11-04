Observation field types
=======================

These aliases describe values accepted by observation models and their public
APIs. Model fields show whether a value is required or nullable.

.. currentmodule:: None

Shared fields
-------------

.. py:type:: tolteca_datamodels.core.types.DataModelsT

   Immutable set of applicable data-model names. Collection inputs are normalized
   to a ``frozenset`` of strings; duplicates and input order have no meaning.

.. py:type:: tolteca_datamodels.lmt_base.types.LabelT

   String label.

.. py:type:: tolteca_datamodels.lmt_base.types.SeqNumT

   Nonnegative observation sequence number.

.. py:type:: tolteca_datamodels.lmt_base.types.DatetimeT

   Timezone-aware datetime.

File-table options
------------------

.. py:type:: tolteca_datamodels.core.file.trace.SourceErrorPolicyT

   One of ``"record"`` or ``"raise"``. See
   :meth:`~tolteca_datamodels.core.file.FileAccessorBase.from_sources` for failures
   governed by this choice.

.. py:type:: tolteca_datamodels.core.file.workflow.FileTableModeT

   One of ``"guess"``, ``"guess_or_read"`` or ``"read"``. See
   :meth:`~tolteca_datamodels.file.FileAccessor.from_sources` for the
   population performed by each mode.

Observation selectors
---------------------

.. py:type:: tolteca_datamodels.core.obs_spec.selectors.FieldValue

   A nonnegative integer or selector label. Labels begin with an ASCII letter
   and continue with letters, digits or underscores.

.. py:type:: tolteca_datamodels.core.obs_spec.selectors.FieldValues

   One :py:type:`~tolteca_datamodels.core.obs_spec.selectors.FieldValue`, a
   nonempty tuple of those values, :class:`~tolteca_datamodels.core.obs_spec.ValueBounds`,
   or None for unconstrained values.

.. py:type:: tolteca_datamodels.core.obs_spec.selectors.Selector

   :py:type:`~tolteca_datamodels.core.obs_spec.selectors.FieldValues`,
   :class:`~tolteca_datamodels.core.obs_spec.FieldSelection` or
   :class:`~tolteca_datamodels.core.obs_spec.FieldPattern`. See :doc:`obs_spec`
   for the corresponding text syntax.

Telescope and readout labels
----------------------------

:class:`~tolteca_datamodels.lmt.types.LmtInfo` and
:class:`~tolteca_datamodels.toltec.types.ToltecInfo` define their instrument
and acquisition-master types. ``ToltecInfo`` also defines array
indices, array names, ROACH indices and observation-type labels and codes.

.. py:type:: tolteca_datamodels.lmt.metadata.LmtTelInterfaceT

   The telescope interface ``"tel"``.

.. py:type:: tolteca_datamodels.lmt.metadata.LmtTelInterfaceKindT

   The interface kind ``"tel"``.

.. py:type:: tolteca_datamodels.toltec.hwpr.ToltecHwprInterfaceT

   The readout interface ``"hwpr"``.

.. py:type:: tolteca_datamodels.toltec.hwpr.ToltecHwprInterfaceKindT

   The interface kind ``"hwpr"``.
