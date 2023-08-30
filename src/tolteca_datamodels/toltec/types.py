from enum import Flag, IntEnum, auto

__all__ = ["ToltecDataKind", "DB_RawObsMaster", "DB_RawObsType"]


class ToltecDataKind(Flag):
    """TolTEC data kinds."""

    # Raw KIDs data
    VnaSweep = auto()
    """A full range sweep on a regular grid."""

    TargetSweep = auto()
    """A targed sweep on a list of frequencies."""

    Tune = auto()
    """A pack of two back-to-back target sweeps to improve the probe tones."""

    RawSweep = VnaSweep | TargetSweep | Tune
    """The sum of all sweep kinds."""

    RawTimeStream = auto()
    """A continuous capturing of the data at given probe tones."""

    RawKidsData = RawSweep | RawTimeStream
    """The sum of all raw kids data kinds."""

    # Reduced KIDs data kinds
    D21 = auto()
    ReducedVnaSweep = auto()
    ReducedTargetSweep = auto()
    ReducedSweep = ReducedVnaSweep | ReducedTargetSweep
    SolvedTimeStream = auto()
    ReducedKidsData = D21 | ReducedSweep | SolvedTimeStream

    # Raw and reduced sum types
    Sweep = RawSweep | ReducedSweep
    TimeStream = RawTimeStream | SolvedTimeStream
    KidsData = RawKidsData | ReducedKidsData

    # Observing-time reduction data types.

    TargFreqsDat = auto()
    """The legacy targ_freqs.dat consumed by tlaloc."""

    TargAmpsDat = auto()
    """The legacy default_amps.dat consumed by tlaloc."""

    KidsModelParamsTable = auto()
    """The legacy kids model fitting result from kidscpp."""

    KidsPropTable = auto()
    """The KIDs finding/fitting results."""

    TonePropTable = auto()
    """The table listing tone properties, derived from KPT."""

    ChanPropTable = auto()
    """The table listing channel properties, derived from KPT."""

    ArrayPropTable = auto()
    """The array property table derived from beammapping."""

    PointingTable = auto()
    """The pointing property table derived from pointing observation."""

    KidsTableData = KidsModelParamsTable | KidsPropTable | TonePropTable | ChanPropTable
    TableData = KidsTableData | ArrayPropTable | PointingTable

    # Raw infrastructural kinds
    Hwpr = auto()
    """The half wave place rotation data."""

    Wyatt = auto()
    """The wyatt robot arm."""

    LmtTel = auto()
    """LMT telescope file"""

    LmtTel2 = auto()
    """Suplementary LMT telescope file with data at their original sample rate."""

    HouseKeeping = auto()
    """The house keeping data."""

    # settings and configurations.
    LmtOtScript = auto()
    """LMT OT script."""

    ToltecaConfig = auto()
    """TolTECA Yaml config."""

    Unknown = auto()
    """Unknown data."""


class DB_RawObsMaster(IntEnum):  # noqa: N801
    """The is in line with the ``toltec/master`` table in the toltec db."""

    TCS = 0
    """The telescope control system."""

    ICS = 1
    """The instrument control system."""

    CLIP = 2
    """The ROACH manager."""

    @classmethod
    def get_master_name(cls, master):
        """Return the name of the master."""
        return DB_RawObsMaster(master).name.lower()


class DB_RawObsType(IntEnum):  # noqa: N801
    """The is in line with the ``toltec/obstype`` table in the toltec db."""

    Nominal = 0
    """Nominal observation."""

    Timestream = 1
    """Time stream probed at a set of tones."""

    VNA = 2
    """VnaSweep."""

    TARG = 3
    """TargetSweep."""

    TUNE = 4
    """TUNE."""

    @classmethod
    def get_data_kind(cls, raw_obs_type):
        """Return the data kind for ``raw_obs_type``."""
        return {
            0: ToltecDataKind.RawTimeStream,
            1: ToltecDataKind.RawTimeStream,
            2: ToltecDataKind.VnaSweep,
            3: ToltecDataKind.TargetSweep,
            4: ToltecDataKind.Tune,
        }.get(raw_obs_type, ToltecDataKind.Unknown)
