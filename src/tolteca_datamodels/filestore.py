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

    def get_path_info_table(self):
        """Return the managed subpaths."""
        raise NotImplementedError

    def glob(self, pattern) -> list[Path]:
        """Return the info."""
        paths = self.get_path_info_table()["path"]
        results = []
        for path in paths:
            results.extend(p for p in path.glob(pattern))
        return results
