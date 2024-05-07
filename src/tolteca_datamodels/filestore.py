from pathlib import Path


class FileStoreBase:
    """A base class manages a directory of files."""

    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"file store path does not exist: {path}")
        self._path = path

    _path: Path

    @property
    def path(self):
        """The rootpath."""
        return self._path
