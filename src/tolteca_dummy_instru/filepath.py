"""Filename information for synthetic counter observations."""

import re
from datetime import UTC, datetime

from tolteca_datamodels.core.filepath import FilepathInfo, register_filepath_parser
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt.types import LmtDataModel
from tolteca_datamodels.lmt_base import FilepathInfoMixin

from .types import ChannelT, DummyInstruDataModel, DummyInstruInfo

__all__ = ["DummyInstruFilepathInfo", "DummyInstruLmtFilepathInfo"]


class DummyInstruLmtFilepathInfo(LmtFilepathInfo):
    """Telescope filepath information for dummy-instrument observations."""


class DummyInstruFilepathInfo(FilepathInfo, FilepathInfoMixin):
    """Information in ``chan00-tcs-1-0-2-20251031T120000.nc`` filenames.

    The parts are prefix, master, observation number, sub-observation number,
    scan number, UTC file-creation time and extension. The filename contains
    its master independently of the directory layout.
    """

    instru: DummyInstruInfo.InstruT
    channel: ChannelT

    @staticmethod
    def _parse_identity(data, path):
        interface = data["prefix"]
        return {
            "interface": interface,
            "instru": DummyInstruInfo.instru,
            "data_models": [DummyInstruDataModel.name],
            "interface_kind": "counter",
            "channel": DummyInstruInfo.interface_channel[interface],
        }

    @staticmethod
    def _parse_ut(data, path):
        return {
            "ut": datetime.strptime(data["ut"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        }

    @staticmethod
    def _parse_suffix(data, path):
        return {"suffix": None}


register_filepath_parser(
    pattern=re.compile(
        r"^(?P<prefix>chan0[0-3])-(?P<master>[a-z]+)-"
        r"(?P<obsnum>\d+)-(?P<subobsnum>\d+)-(?P<scannum>\d+)-"
        r"(?P<ut>\d{8}T\d{6})\.(?P<ext>nc)$"
    ),
    parsers=(
        DummyInstruFilepathInfo._parse_identity,
        DummyInstruFilepathInfo._parse_ut,
        DummyInstruFilepathInfo._parse_suffix,
    ),
    filepath_info_cls=DummyInstruFilepathInfo,
)


def _parse_tel_data_models(data, path):
    """Identify a telescope filename registered for the dummy instrument."""
    return {"data_models": [LmtDataModel.name, DummyInstruDataModel.name]}


register_filepath_parser(
    pattern=LmtFilepathInfo._make_filename_pattern(r"tel_dummy_instru"),
    parsers=(*LmtFilepathInfo._parsers, _parse_tel_data_models),
    filepath_info_cls=DummyInstruLmtFilepathInfo,
)
