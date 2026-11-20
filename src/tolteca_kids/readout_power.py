"""Saved acquired-comb readout-power diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper

from .roach_tone import RoachReadoutPowerConfig


@dataclass
class RoachReadoutPowerSchema(Schema):
    """Power-chain arrays, channel frequencies and acquisition settings."""

    chan: NameMapping = NameMapping.dim("chan")
    element: NameMapping = NameMapping.var("element")
    tone_amps_db: NameMapping = NameMapping.var("tone_amps_db")
    gain: NameMapping = NameMapping.var("gain")
    frequency: NameMapping = NameMapping.var("frequency")
    predicted_per_tone_input: NameMapping = NameMapping.var("predicted_per_tone_input")
    predicted_per_tone_output: NameMapping = NameMapping.var(
        "predicted_per_tone_output"
    )
    inferred_per_tone_input: NameMapping = NameMapping.var("inferred_per_tone_input")
    inferred_per_tone_output: NameMapping = NameMapping.var("inferred_per_tone_output")
    predicted_total_input: NameMapping = NameMapping.var("predicted_total_input")
    predicted_total_output: NameMapping = NameMapping.var("predicted_total_output")
    inferred_total_input: NameMapping = NameMapping.var("inferred_total_input")
    inferred_total_output: NameMapping = NameMapping.var("inferred_total_output")
    adc_snap_frac: NameMapping = NameMapping.var("adc_snap_frac")
    roach: NameMapping = NameMapping.attr("roach")
    atten_drive: NameMapping = NameMapping.attr("atten_drive")
    atten_sense: NameMapping = NameMapping.attr("atten_sense")
    config: NameMapping = NameMapping.attr("config")


class RoachReadoutPowerMapper(XarrayMapper[RoachReadoutPowerSchema]):
    """Access to saved acquired-comb power diagnostics."""


class RoachReadoutPowerView(XarrayAccessorBase[RoachReadoutPowerMapper]):
    """Predicted and inferred powers through the acquired readout chain.

    Parameters
    ----------
    data_source : xarray.Dataset or xarray.DataTree
        Saved readout-power result. Keep its source open while accessing arrays.
    mapper : RoachReadoutPowerMapper, optional
        Existing field mapping; otherwise resolved from the supplied source.

    Notes
    -----
    Per-tone arrays have dimensions (element, chan); total arrays have dimension
    (element). Power arrays carry dB(mW), gains carry dB, and frequencies carry Hz.
    Accessing these properties reads saved arrays without repeating the analysis.
    """

    @property
    def config(self):
        """Calibration and limits used for the saved result."""
        return RoachReadoutPowerConfig.model_validate_json(
            self.mapper.get_value(self.data_source, self.mapper.schema.config)
        )

    @property
    def roach(self):
        """ROACH network number."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.roach)

    @property
    def atten_drive(self):
        """Acquired drive attenuation in dB."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.atten_drive)

    @property
    def atten_sense(self):
        """Acquired sense attenuation in dB."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.atten_sense)

    @property
    def element(self):
        """Readout stages in signal-path order."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.element)

    @property
    def tone_amps_db(self):
        """Relative voltage attenuation per channel, in dB."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.tone_amps_db)

    @property
    def gain(self):
        """Signed gain at each stage, in dB."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.gain)

    @property
    def frequency(self):
        """Recorded baseband tone frequencies, in Hz."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.frequency)

    @property
    def predicted_per_tone_input(self):
        """Predicted per tone input, in dBm."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.predicted_per_tone_input
        )

    @property
    def predicted_per_tone_output(self):
        """Predicted per tone output, in dBm."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.predicted_per_tone_output
        )

    @property
    def inferred_per_tone_input(self):
        """Inferred per tone input, in dBm."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.inferred_per_tone_input
        )

    @property
    def inferred_per_tone_output(self):
        """Inferred per tone output, in dBm."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.inferred_per_tone_output
        )

    @property
    def predicted_total_input(self):
        """Predicted total input, in dBm."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.predicted_total_input
        )

    @property
    def predicted_total_output(self):
        """Predicted total output, in dBm."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.predicted_total_output
        )

    @property
    def inferred_total_input(self):
        """Inferred total input, in dBm."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.inferred_total_input
        )

    @property
    def inferred_total_output(self):
        """Inferred total output, in dBm."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.inferred_total_output
        )

    @property
    def adc_snap_frac(self):
        """Mean 12-bit ADC usage fraction, or None without a recorded snapshot."""
        if self.mapper.schema.adc_snap_frac not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.adc_snap_frac)
