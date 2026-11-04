# File stores

A **file store** knows the directory layout of a data tree. Given an
[observation selector](obs_spec.md), it returns the candidate paths that can hold the
selected files, using directory and filename patterns alone. It opens no file and reads
no header. Pass the paths to a [file accessor](file-tables.md) to build a table, or let
`query_file_store()` do both.

This is the third level of the [observation-data stack](datamodels.md#choose-a-level).
To search several trees, include renamed files or share the search with command-line
programs, describe it in a [recipe](input-discovery.md) instead.

The examples find observation 145609 in the reference data. `data_root` is the
directory that holds your raw-data `tel/` and `toltec/` trees.

```{testsetup} file_stores
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec.file_store import ToltecFileStore

data_root = workspace / "toltec_astro_dev/run/data_lmt"
store = LmtFileStore(data_root)
```

## Find candidate paths

`LmtFileStore` searches an LMT acquisition tree: the telescope directory `tel/` and the
TolTEC subtree `toltec/<master>/<interface>/`. `ToltecFileStore` searches a TolTEC
subtree alone. Construction accepts any directory, existing or not.

Ask a store for the paths that a selector can match, then build a table from them:

```{testcode} file_stores
from tolteca_datamodels.toltec.file import ToltecFileAccessor

spec = ObsSpec.parse("145609/{tel,0}")

paths = list(store.partial(LmtFileStore.tel, LmtFileStore.toltec).glob_obs_spec(spec))
file_accessor = ToltecFileAccessor.from_sources(paths, mode="read")
selected = file_accessor.query_obs_spec(spec)

assert sorted(selected.uid_raw_obs_file) == [
    "tcs-145609-0-1-tel",
    "tcs-145609-0-1-toltec0",
    "tcs-145609-0-2-tel",
    "tcs-145609-0-2-toltec0",
]
```

`glob_obs_spec()` returns the unique existing paths in location and glob order; missing
directories and unmatched patterns contribute nothing, and the paths are not sorted.
The candidates come from directory and filename patterns, so they assume organized
filenames. The final `query_obs_spec()` applies the complete selector to the populated
table, including constraints that the patterns cannot express.

`query_file_store()` performs the same steps. Its `DataModelSet` supplies the store
branches and the file accessor types:

```{testcode} file_stores
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.input_discovery import query_file_store

models = DataModelSet.from_select(["toltec"])
selected = query_file_store(store, spec, mode="read", data_models=models)

assert sorted(selected.interface) == ["tel", "tel", "toltec0", "toltec0"]
```

## Choose which directories are searched

The interface field of the selector decides which directories a search visits. An
omitted or wildcard interface visits both the telescope directory and the instrument
trees; a `tel` interface visits the telescope directory only; an interface that excludes
`tel` skips it.

| Selector | Directories searched |
| --- | --- |
| `145609`, `145609/*` | `tel/` and every `toltec/<master>/<interface>/` |
| `145609/tel` | `tel/` only |
| `145609/0` | `toltec/<master>/toltec0/` only |
| `145609/{tel,0}` | `tel/` and `toltec/<master>/toltec0/` |

Telescope paths are identified with master `tcs`, so `tcs-145609/tel` also matches
them. Telescope streams of every instrument share the interface `tel`; select their
literal filename prefix afterward with `file_prefix == 'tel_toltec'`. A selector
constrains only the fields it supplies, and a file whose constrained metadata is
unknown does not match.

`partial()` chooses branches of the store explicitly and returns an immutable view that
supports the same search methods. Name branches by position or by the store's class
descriptors, or exclude some and keep the rest:

```{testcode} file_stores
tel_only = store.partial("tel")
toltec_only = store.partial(exclude=("tel",))
by_descriptor = store.partial(LmtFileStore.tel, LmtFileStore.toltec)

assert list(tel_only.iter_glob_patterns("145609")) == ["tel/tel_*_145609_*.nc"]
assert not any(p.startswith("tel/") for p in toltec_only.iter_glob_patterns("145609"))
assert list(by_descriptor.iter_glob_patterns("145609/tel")) == [
    "tel/tel_*_145609_*.nc"
]
```

Further `partial()` calls can only narrow a view. Positions and `exclude=` cannot be
combined, and omitting both raises `ValueError`; an empty `exclude` removes nothing.

## Inspect the patterns

`iter_glob_patterns(spec)` yields the patterns a search would use, without touching the
filesystem. It is the quickest way to see what a selector asks of a layout:

```{testcode} file_stores
assert list(store.iter_glob_patterns("145609/{tel,0}")) == [
    "tel/tel_*_145609_*.nc",
    "toltec/*/toltec0/toltec0_145609_*.nc",
]
assert list(store.iter_glob_patterns("145609/0")) == [
    "toltec/*/toltec0/toltec0_145609_*.nc"
]
```

Patterns on a store are relative to `store.path`; patterns on a view are relative to
`view.root_path`, even when the view selects a more specific `path`. The TolTEC tree
uses fixed-depth master and interface patterns, with a wildcard directory when the
master is unconstrained, and never searches recursively.

## Interfaces and positions

A store resolves interface constraints against its own interfaces before it globs. For
TolTEC, a bare `1/(1,5)` and the prefixed `1/toltec(1,5)` search the same four ROACH
directories:

```{testcode} file_stores
toltec_store = ToltecFileStore(data_root / "toltec")

bare = list(toltec_store.iter_glob_patterns("1/(1,5)"))
assert bare == list(toltec_store.iter_glob_patterns("1/toltec(1,5)"))
assert bare == [
    "*/toltec1/toltec1_000001_*.nc",
    "*/toltec2/toltec2_000001_*.nc",
    "*/toltec3/toltec3_000001_*.nc",
    "*/toltec4/toltec4_000001_*.nc",
]
```

Positional selection waits for the populated table: `1/(1,5)[-1]` still searches all
four directories, and the file accessor's `query_obs_spec()` then picks from the
candidates that exist. If only interface 1 has files for observation 1, those files are
selected.

## Substores

A store contains the stores of the instruments registered beneath it. Use
`store.substores[ToltecFileStore]` to get a child with its concrete type, or the named
selection `store.toltec`. The mapping is read-only, repeated access returns the same
instance, and looking up an unregistered class raises `KeyError`. Constructing a store
creates its children without searching directories or opening files.

```{testcode} file_stores
child = store.substores[ToltecFileStore]

assert child is store.substores[ToltecFileStore]
assert child.path == store.path / "toltec"
```

A search visits local locations first, then substores recursively, and keeps the first
occurrence of each candidate.

A selector can be passed as text or as a parsed `ObsSpec`. Text is parsed with the type
that the store you call supports. A substore that does not support the selector's type
skips its own files but still visits its children, and invalid text raises.

## Other layouts

A store assumes the organized acquisition layout. For renamed or reorganized files,
list them as exact `sources` or `{glob: pattern}` resolvers in an
[input discovery recipe](input-discovery.md#include-renamed-files). `mode:
guess_or_read` identifies renamed files from their headers, and `mode: read` also
guards against recognizable filenames with wrong identifiers.

## Next

- Select among the candidates, open files and read metadata:
  [file tables](file-tables.md).
- Describe the search in a recipe shared with command-line programs:
  [input discovery](input-discovery.md).
- Selector syntax: [observation selectors](obs_spec.md).
- Add a store for a new instrument: [framework architecture](datamodels-architecture.rst).
