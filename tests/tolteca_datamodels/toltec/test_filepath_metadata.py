"""KIDs acquisition paths are metadata, not current source locations."""

import pytest

from tolteca_datamodels.toltec import open_toltec_kids


def test_path_field_preserves_text_but_requires_the_header(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
    with open_toltec_kids(path) as data:
        # Unrecognized text is kept verbatim, not silently replaced by a guess.
        meta = type(data.toltec_kids.meta).model_validate(
            data.toltec_kids.meta.model_dump(round_trip=True)
            | {"filepath_orig": "unknown.nc"}
        )
        assert meta.filepath_orig == "unknown.nc"
        missing = data.drop_vars("Header.Toltec.Filename")
        with pytest.raises(KeyError, match="Header.Toltec.Filename"):
            _ = missing.toltec_kids.meta
