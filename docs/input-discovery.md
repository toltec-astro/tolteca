# Discover inputs

Discovery finds the files for a task when you do not have a fixed list: it searches
data trees or configured locations, populates a [file table](file-tables.md), and
applies an [observation selector](obs_spec.md). This page covers the top level of the
[observation-data stack](datamodels.md#choose-a-level): a YAML recipe that Python code
and command-line programs share. A recipe uses the [file stores](file-stores.md) to
search acquisition trees, and you can call the stores directly in Python when you need
only candidate paths.

The examples find observation 145609 in the reference data. `/data/data_lmt` in the
recipes stands for your raw-data directory, which holds the `tel/` and `toltec/`
trees. Examples read inputs in place.

```{testsetup} input_discovery
import os
import tempfile
from pathlib import Path

import yaml

data_root = workspace / "toltec_astro_dev/run/data_lmt"
recipe_dir = workspace / "tolteca/docs/_static/input-discovery"
downloads = tempfile.TemporaryDirectory()
os.symlink(
    data_root / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
    Path(downloads.name) / "downloaded.nc",
)


def load_recipe(name):
    text = (recipe_dir / f"{name}.yaml").read_text()
    text = text.replace("/data/data_lmt", str(data_root))
    text = text.replace("/data/downloads", downloads.name)
    return yaml.safe_load(text)
```

```{testcleanup} input_discovery
downloads.cleanup()
```

## Choose how to find files

| Entry | Searches | Default `mode` | Use when |
| --- | --- | --- | --- |
| `data_lmt_path` | An LMT acquisition tree: `tel/` and the instrument trees such as `toltec/<master>/<interface>/` | `guess` | The files keep the acquisition layout and names |
| `data_toltec_path` | A TolTEC subtree, never the sibling `tel/` | `guess` | You have only the TolTEC subtree |
| `sources` | Exact locations and `{glob: pattern}` resolvers | `guess_or_read` | Files are listed explicitly, or were renamed, copied or reorganized |

Every entry also accepts `obs_spec` to narrow the files by observation and interface
(trees default to `*`; for `sources`, omit it to keep every file), `mode` to choose
how much to read ([modes](file-tables.md#choose-how-much-to-read)), and
`data_model_select` to choose the data models that handle the files. An entry has exactly
one of `data_lmt_path`, `data_toltec_path` and `sources`.

## Describe discovery in a recipe

`InputDiscovery` takes a Tollan runtime configuration with an `input_discovery:`
section. Each entry discovers candidates, populates its own table, and applies its own
`obs_spec`; the results are then combined. The recipes below are complete files; load
one with `yaml.safe_load` (these examples substitute the reference data directory for
`/data/data_lmt`) and discover:

```{literalinclude} _static/input-discovery/lmt-tree.yaml
:language: yaml
```

```{testcode} input_discovery
import yaml
from tollan.config import RuntimeContext
from tolteca_datamodels.input_discovery import InputDiscovery

config = load_recipe("lmt-tree")
table = InputDiscovery(RuntimeContext(config)).discover()

assert sorted(table.uid_raw_obs_file) == [
    "tcs-145609-0-1-tel",
    "tcs-145609-0-1-toltec0",
    "tcs-145609-0-2-tel",
    "tcs-145609-0-2-toltec0",
]
```

The table is the same kind of model table as in [file tables](file-tables.md):
query it with pandas, call `read_metadata()`, or open its files. `obs_spec` here selects
the telescope and TolTEC readout 0 of observation 145609; which directories each
selector searches is described in [file stores](file-stores.md#choose-which-directories-are-searched).

### Search a TolTEC subtree

`data_toltec_path` points directly at a TolTEC subtree, with master and interface
directories below it. It uses the same `obs_spec` and `mode` options and never searches
the sibling telescope directory.

```{literalinclude} _static/input-discovery/toltec-tree.yaml
:language: yaml
```

```{testcode} input_discovery
table = InputDiscovery(RuntimeContext(load_recipe("toltec-tree"))).discover()

assert table.interface.tolist() == ["toltec0", "toltec0"]
assert sorted(table.scannum) == [1, 2]
```

### List exact files and patterns

A `sources` entry lists locations. A string or a `{path: ...}` dictionary is one
exact location, even if it contains `*`, `?` or `[]`. Only `{glob: pattern}` expands a
pattern.

```{literalinclude} _static/input-discovery/sources-exact.yaml
:language: yaml
```

```{testcode} input_discovery
table = InputDiscovery(RuntimeContext(load_recipe("sources-exact"))).discover()

assert table.interface.tolist() == ["tel", "toltec0"]
```

In a local glob, `*` matches within one directory and `**` searches recursively. A
glob matches files only and returns them in the filesystem's enumeration order, not
sorted. A glob without a match contributes no rows. Remote globs are unsupported.

```{literalinclude} _static/input-discovery/sources-glob.yaml
:language: yaml
```

```{testcode} input_discovery
table = InputDiscovery(RuntimeContext(load_recipe("sources-glob"))).discover()

assert sorted(table.interface) == ["tel", "tel", "toltec0", "toltec0"]
```

### Include renamed files

A renamed or downloaded file has no filename that identifies it. `mode: guess_or_read`
(the default for `sources`) opens the files whose names identify no data model and
reads the observation from their headers, so `obs_spec` can select them:

```{literalinclude} _static/input-discovery/renamed.yaml
:language: yaml
```

```{testcode} input_discovery
table = InputDiscovery(RuntimeContext(load_recipe("renamed"))).discover()

assert table.uid_raw_obs_file.tolist() == ["tcs-145609-0-2-tel"]
```

### Read headers of recognized files

`mode: read` also reads the headers of files whose names the selected data models
recognize, so the table carries recorded metadata such as `obs_goal` and a wrong
filename cannot select the wrong observation:

```{literalinclude} _static/input-discovery/read-mode.yaml
:language: yaml
```

```{testcode} input_discovery
table = InputDiscovery(RuntimeContext(load_recipe("read-mode"))).discover()

assert set(table.obs_goal) == {"pointing"}
```

Recognized filenames that no selected data model handles keep their filename
information and are not opened in either read mode.

### Choose data models

`data_model_select` chooses the data models whose stores and readers discovery uses.
Set it at the top of `input_discovery:` for every entry, or on an entry to narrow the
global choice further. A list names models exactly, `{excludes: [name]}` removes
names, and a string or `{pattern: regex}` matches complete names. The default `.*`
selects every model available to the task. It does not change filename parsing.

```{literalinclude} _static/input-discovery/select-global.yaml
:language: yaml
```

```{testcode} input_discovery
table = InputDiscovery(RuntimeContext(load_recipe("select-global"))).discover()

assert table.interface.tolist() == ["tel", "tel"]
```

The TolTEC model covers its telescope companions, so `toltec` also finds `tel` files.
Each entry can use its own selection:

```{literalinclude} _static/input-discovery/select-entry.yaml
:language: yaml
```

```{testcode} input_discovery
table = InputDiscovery(RuntimeContext(load_recipe("select-entry"))).discover()

assert sorted(table.interface) == ["tel", "tel", "toltec0", "toltec0"]
```

In Python, `discover(data_models=DataModelSet.from_select(["toltec"]))` supplies the
scope that the recipe settings narrow. An empty selection searches no stores.

### Combine entries

Rows follow the entry order, and each entry applies its own `obs_spec`. Duplicate
locations, including symlinks to the same file, keep only their first occurrence. In
this recipe the second entry matches files the first already selected, and the third
is a renamed copy of one of them, so together they add no rows:

```{literalinclude} _static/input-discovery/combined.yaml
:language: yaml
```

```{testcode} input_discovery
table = InputDiscovery(RuntimeContext(load_recipe("combined"))).discover()

assert len(table) == 4
assert table.uid_raw_obs_file.is_unique
```

Rows whose source failed validation have no location to compare, so they stay
separate with their `source_trace` entries. A pipeline's own input selection is a
later step; discovery applies only the entries' `obs_spec`.

### Resolve relative paths

Relative paths in a recipe use the runtime's `root_path`, or the working directory when
none is supplied; they do not use the directory of the YAML file. An explicit FileLoc
`local_parent_path` takes precedence. Supply `root_path` through the validation
context:

```{literalinclude} _static/input-discovery/relative.yaml
:language: yaml
```

```{testcode} input_discovery
runtime = RuntimeContext(yaml.safe_load((recipe_dir / "relative.yaml").read_text()))
runtime = runtime.with_config(
    validation_context={"root_path": workspace / "toltec_astro_dev/run"}
)
table = InputDiscovery(runtime).discover()

assert sorted(table.interface) == ["tel", "tel", "toltec0", "toltec0"]
```

Set an absolute `root_path` when a saved recipe is reused elsewhere.

### Handle failures

By default discovery records a file that cannot be read in the table and keeps the row.
Use `on_error="raise"` to stop at the first failure. Invalid configuration always
raises. An unmatched glob, a missing tree or a selector without matches yields no
rows, and discovery may return an empty table.

```{testcode} input_discovery
missing = {"input_discovery": {"inputs": [{"sources": ["/data/downloads/missing.nc"]}]}}

table = InputDiscovery(RuntimeContext(missing)).discover()
for row in table.itertuples(index=False):
    for issue in row.source_trace.iter_error_entries():
        print(issue.kind.value)

assert table.source_trace.iloc[0].has_error

try:
    InputDiscovery(RuntimeContext(missing)).discover(on_error="raise")
except FileNotFoundError:
    pass
else:
    raise AssertionError("expected FileNotFoundError")
```

```{testoutput} input_discovery
source_open_failed
```

## Run discovery from the command line

Programs that process observations read the same `input_discovery:` section.
`tolteca_kids` and `tolteca_drive_fit` also accept `--input_discovery.*` overrides.
Save a recipe as `check.yaml` and let `tolteca_kids` show what it finds before
processing:

```console
tolteca_kids --config_paths check.yaml --show_inputs
```

Positional locations or an observation selector replace the configured entries; an
observation selector needs `--data_lmt_path`:

```console
tolteca_kids --data_lmt_path /data/data_lmt --show_inputs -- 'tcs-145609/0'
tolteca_kids --show_inputs -- sweep.nc
```

Under a deployment, `tolteca_deploy run -- tolteca_kids --config_paths check.yaml
--dry_run` forwards the same arguments. Output, input selection and the other
options belong to each program; [KIDs inputs](kids-inputs.md) describes them for
`tolteca_kids`.

## Next

- Select among the discovered files, check a KIDs run, and the command-line options:
  [KIDs inputs](kids-inputs.md).
- Open the files, read metadata and inspect recorded issues:
  [file tables](file-tables.md).
- Directory layouts and the candidate paths a selector asks for:
  [file stores](file-stores.md).
- Selector syntax: [observation selectors](obs_spec.md).
- Configuration fields and defaults: {ref}`the discovery API <observation-input-discovery>`.
