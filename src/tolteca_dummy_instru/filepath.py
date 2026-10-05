"""Filename information for synthetic counter observations."""

import re
from datetime import UTC, datetime
from typing import ClassVar

from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.lmt_base import FilepathInfoMixin

from .types import ChannelT, DummyInstruInfo

__all__ = ["DummyInstruFilepathInfo"]


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
            "instru_component": "counter",
            "channel": DummyInstruInfo.interface_channel[interface],
        }

    @staticmethod
    def _parse_ut(data, path):
        return {
            "ut": datetime.strptime(data["ut"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        }

    _filename_patterns: ClassVar = {
        re.compile(
            r"^(?P<prefix>chan0[0-3])-(?P<master>[a-z]+)-"
            r"(?P<obsnum>\d+)-(?P<subobsnum>\d+)-(?P<scannum>\d+)-"
            r"(?P<ut>\d{8}T\d{6})\.(?P<ext>nc)$"
        ): (_parse_identity, _parse_ut),
    }
