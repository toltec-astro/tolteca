"""Replay production v2 finding and Tlaloc output in its own environment."""

import sys
from pathlib import Path


def main():
    """Save numerical tables and the four original Tlaloc files."""
    from tollan.utils.log import logger  # noqa: PLC0415 -- production-v2 interpreter
    from tolteca_datamodels.toltec.ncfile import (  # noqa: PLC0415 -- production-v2 interpreter
        NcFileIO,
    )
    from tolteca_kids.roach_tone import (  # noqa: PLC0415 -- production-v2 interpreter
        TlalocEtcDataStore,
    )

    from tolteca_kids.kids_find import (  # noqa: PLC0415 -- production-v2 interpreter
        KidsFind,
    )
    from tolteca_kids.sweep_check import (  # noqa: PLC0415 -- production-v2 interpreter
        SweepCheck,
    )
    from tolteca_kids.tlaloc_output import (  # noqa: PLC0415 -- production-v2 interpreter
        PlaceHolderTones,
        TlalocOutput,
        TonePhasesRandom,
    )

    logger.remove()
    source, destination, block = sys.argv[1:]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "toltec0").mkdir(exist_ok=True)
    sweep = NcFileIO(source).read(block=int(block))
    SweepCheck()(sweep)
    KidsFind()(sweep)
    found = KidsFind.get_context(sweep).data
    found.detected.write(destination / "detected.ecsv", format="ascii.ecsv")
    cols = [
        "idx_det",
        "bitmask_det",
        "f_det",
        "Qr",
        "dist",
        "d_phi",
        "idx_chan",
        "f_chan",
        "amp_tone",
        "snr_d21",
        "snr_s21",
        "height_d21",
        "height_db_s21",
    ]
    found.detected_matched[cols].write(
        destination / "kids_find.ecsv", format="ascii.ecsv"
    )
    found.chan_matched[cols].write(destination / "chan_prop.ecsv", format="ascii.ecsv")
    tones = TlalocOutput.make_roach_tone_props(
        swp=sweep,
        tone_phases=TonePhasesRandom(method="random", seed=42),
        placeholders=PlaceHolderTones(),
        fix_n_chans="_tune.nc" in source,
    )
    TlalocEtcDataStore(destination).write_tone_props(tones, sort_fft=True)


if __name__ == "__main__":
    main()
