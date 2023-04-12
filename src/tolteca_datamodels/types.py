from enum import Flag, auto


class DataProdKind(Flag):
    """Type for data product."""

    Unspecified = auto()

    # Shape
    TimeOrderedData = auto()
    Image = auto()
    Cube = auto()
    Catalog = auto()

    # Calibration state
    Raw = auto()
    Calibrated = auto()

    # Some common sum types.
    RawTimeOrderedData = Raw | TimeOrderedData
    CalibratedTimeOrderedData = Calibrated | TimeOrderedData
    CalibratedImage = Calibrated | Image

    # Measurement types
    Signal = auto()
    NoiseRealization = auto()
    Measurement = Signal | NoiseRealization
    Weight = auto()
    Variance = auto()
    RMS = auto()
    Uncertainty = Weight | Variance | RMS

    # ancillary measurement types
    Coverage = auto()
    Flag = auto()
    Kernel = auto()
