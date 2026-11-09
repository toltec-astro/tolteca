"""YAML I/O for portable product records."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml
from tollan.utils.yaml import yaml_dump

# Both loaders use PyYAML's safe constructors. Minimal installations may lack
# LibYAML; large provenance snapshots benefit from its native parser.
_RECORD_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _read_yaml(path: Path) -> object:
    with path.open("rb") as stream:
        return yaml.load(stream, Loader=_RECORD_LOADER)  # noqa: S506 -- safe constructors only


def _write_yaml(record, path, *, overwrite=False):
    contents = yaml_dump(record.model_dump(mode="json", round_trip=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".yaml", delete=False
    ) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
            if overwrite:
                tmp.replace(path)
            else:
                # Exclusive publication without exposing an incomplete YAML.
                path.hardlink_to(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    _sync_directory(path.parent)
    return path


def _sync_directory(path: Path) -> None:
    """Synchronize directory entries on POSIX filesystems."""
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
