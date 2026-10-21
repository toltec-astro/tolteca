# Choose inputs and check a KIDs run

Use `tolteca_kids --dry_run` to validate input metadata and per-input configuration,
then execute a no-op pipeline. It does not load detector signals or write products,
even when output is enabled.

## Configure discovery and selection

Save a recipe such as `check.yaml`:

```yaml
input_discovery:
  inputs:
    - data_lmt_path: /data_lmt
      obs_spec: tcs-145609/0
      mode: guess
    - sources:
        - ./downloaded.nc
        - {glob: "./extra/**/*.nc"}
      obs_spec: tcs-145609/0
      mode: guess_or_read
kids:
  input:
    select: bits_any_set(data_kind_toltec, raw_sweep) & atten_drive == 7.25
    raise_on_errors: false
  output:
    path: /data/products
```

```console
tolteca_kids --config_paths check.yaml --show_inputs
tolteca_kids --config_paths check.yaml --dry_run
```

Each entry in `input_discovery.inputs` chooses one discovery method:

- `data_lmt_path` searches an acquisition tree. ObsSpec narrows the search globs,
  then filters the resulting table. The default mode is `guess`. An omitted or
  wildcard interface searches both the telescope directory and registered
  instrument stores (currently TolTEC). `/tel` searches only the telescope
  directory; `/0` or `/toltec0` skips it. A mixed selector such as `145609/{tel,1}`
  searches both trees and selects telescope and numbered-interface 1 rows
  (`toltec1` in the TolTEC store). `/toltec(1,5)` restricts numeric bounds to
  that prefix; `/(1,5)` uses the interface-to-number mapping each store provides.
- `data_toltec_path` searches a TolTEC subtree directly, with master/interface
  directories below it. It uses the same ObsSpec and mode options, defaulting to
  `guess`. It never searches the sibling telescope directory: `{tel,1}` can only
  return ROACH 1 rows from this subtree, and `/tel` normally returns no rows.
- `sources` lists exact locations and explicit `{glob: pattern}` resolvers.
  Locations accept FileLoc values, including dictionaries such as
  `{path: data.nc}`, and are not interpreted as patterns. Local globs
  return files only: `*` matches within one directory and `**` searches
  recursively. Matches retain the filesystem's glob enumeration order; they are
  not sorted. Remote globs are unsupported. The default mode is `guess_or_read`.

For a standalone TolTEC subtree, use:

```yaml
input_discovery:
  inputs:
    - data_toltec_path: /data/data_lmt/toltec
      obs_spec: tcs-145609/(0,3)
      mode: guess
```

Each entry chooses exactly one of `data_lmt_path`, `data_toltec_path`, or
`sources`. Both tree entries currently discover netCDF files. Use explicit
sources to describe other file formats.

Use `data_model_select` to choose discovery and reading support:

```yaml
input_discovery:
  inputs:
    - data_lmt_path: /data/data_lmt
      data_model_select: [toltec]
      obs_spec: tcs-145609/{tel,0}
```

TolTEC support searches both telescope and TolTEC stores. The value uses
{py:class}`tollan.config.types.StringSelector`: lists select exact data-model
names, `{excludes: [dummy_instru]}` excludes names, and a string or
`{pattern: 'toltec'}` matches complete names. The default `'.*'` selects all
support available to the task. The KIDs CLI limits that support to TolTEC;
configuration can narrow it but cannot enable other data models.

Set `input_discovery.data_model_select` to choose support for all entries;
each entry can narrow it with its own `data_model_select`. Discovery searches
the stores required by the selected data models and uses their handlers to
populate the table. An empty selection searches no stores. Explicit `sources`
still resolve and retain all registered filename information, but with no
selected handlers they are not opened or enriched. With TolTEC support enabled,
unsupported files are classified as TolTEC `unknown`.
ObsSpec can select exact interface labels; bare numeric interface shortcuts
require the corresponding handler.
These settings control discovery only. KIDs opens the resulting inputs and
fills missing TolTEC metadata before pipeline selection. Rows already populated
by discovery retain their metadata without another read.
If different entries enable different classifiers, the combined table has null
values where a classifier was not run; a classifier's `unknown` flag means it
ran but did not recognize that row.

An entry's `mode` controls information available to its `obs_spec`:

| Mode | Table population before ObsSpec |
| --- | --- |
| `guess` | Filename information only; no opens. |
| `guess_or_read` | Also read supported, openable files with unrecognized filenames. |
| `read` | Read openable candidates with unrecognized filenames or names supported by the selected handlers. |

Filename recognition uses all registered parsers, independently of the selected
handlers. Recognized filenames outside the selected handlers' scope retain their
guesses without being opened in either read mode.

Use explicit `sources` for renamed or reorganized files. An unrecognizable name
such as `downloaded.nc` keeps null `file_*` columns; header reads populate the
observation columns. Use `read` to refresh metadata for recognized files in scope.
A bare string containing `*`, `?` or `[]` remains an exact location, including
inside URL paths and query strings. Dictionaries matching neither FileLoc nor a
glob resolver fail validation.

Each entry applies its ObsSpec independently, including positional selectors,
before tables are combined. Rows follow entry and glob enumeration order.
An LMT-tree entry collects telescope candidates first, followed by instrument
candidates. TolTEC visits matching interfaces in their declared order and
preserves each glob's enumeration order. File stores do not sort paths.
Duplicate locations and local symlink targets keep the first selected row.
Unmatched globs, missing acquisition trees, and selectors with no matches
contribute no rows; debug logs report the match counts. Discovery may return an
empty table. Running a pipeline still requires selected inputs. Invalid
configuration raises. Discovery records read failures in the table by default;
use `discover(on_error="raise")` for fail-fast reads.

`[-1]` selects the largest available sequence value, not the newest file by time.
Use an explicit master such as `tcs-[-1]` when comparing observation numbers;
masters have independent sequences. See [ObsSpec](obs_spec.md).

Each pipeline owns its input selection. The KIDs input step's
`kids.input.select` applies to the **combined table after opening candidates and
reading authoritative metadata**, regardless of discovery mode. The CLI records
source-information failures during preparation so unselected bad rows do not
stop the run. After the query, unusable selected rows are skipped with debug
logs by default. Set `kids.input.raise_on_errors: true` to raise instead.
Invalid queries always raise. A filepath diagnostic does not reject a row whose
header was successfully read; missing or invalid acquisition metadata does.
Guesses are not a substitute for a successful metadata read.

Remote locations can be described during discovery but must be staged locally
before execution. Selected files stay open during dispatch
and close on success or failure. Detector signals remain lazy during selection.
Discovery does not apply any pipeline's query; different pipeline configurations
can select different subsets of the same opened table.

For the same scope and error handling in Python, pass
`data_models=DataModelSet.from_select(["toltec"])`
and `on_error="record"` to `discover()`, then use `table.toltec_file.open()`.
Call `opened.toltec_file.update_from_source_metadata(inplace=True)` before
`Kids.input.select()`. See the [worked example](guide.md#discover-files-in-a-data-tree).
Discovery and opening default to recording source-information failures.
Use `on_error="raise"` on both calls for fail-fast opening and metadata reads.
Alternatively, `open(update_from_metadata=True)` performs both steps. Pass
`force=True` to that call or the explicit update to reread populated rows.

Relative paths use the runtime validation context's `root_path`, or the working
directory when no root is supplied, not the directory of each YAML file.
Relative glob patterns without a root_path use the working directory at discovery
time. Supply an absolute root_path when saving configuration for reuse elsewhere.
An explicit FileLoc `local_parent_path` takes precedence over `root_path`;
already constructed FileLoc objects retain their parent settings.

## Use CLI shorthand

Positional locations and observation selectors replace configured discovery
entries. Put them after `--`:

```console
tolteca_kids --dry_run -- sweep.nc
tolteca_kids --data_lmt_path /data_lmt --dry_run -- 'tcs-145609/0'
tolteca_kids --select 'bits_any_set(data_kind_toltec, raw_sweep)' --show_inputs -- sweep.nc
```

Positional ObsSpec requires `--data_lmt_path`. With no positional inputs, that
option supplies a default `*` tree entry only if discovery is not configured.
It does not replace an explicit recipe. Positional locations are literal; use `{glob: ...}` in configuration
for patterns, or let the shell expand an unquoted local glob into exact paths.

`--select` overrides `kids.input.select`. Dotted options are restricted to `--kids.*`
and `--input_discovery.*`. For example:

```console
tolteca_kids --config_paths check.yaml --show_config
tolteca_kids --config_paths check.yaml --dry_run --kids.sweep_reduce.block 0
tolteca_kids --config_paths check.yaml --show_inputs '--input_discovery.inputs.[0].mode' read
tolteca_kids --config_paths check.yaml --show_inputs --input_discovery.inputs.0.data_model_select 'toltec'
```

Quote options containing brackets if your shell treats them as patterns.
`--config_paths` accepts multiple files or directories; later paths override
earlier ones. Explicit CLI shorthand overrides YAML, and dotted overrides apply
after shorthand.

`--show_inputs` prints the selected table, including an empty selection, without
running reduction or output. `--show_config` prints discovery, selection, output
defaults and effective KIDs configuration for each selected input. A run with no
selected inputs fails. Both inspection flags exit after input selection, before
per-file steps, even when combined with `--dry_run`.

Conditional configuration (`enable_if`) uses each selected input's header
information. It configures processing steps; it does not replace `kids.input.select`.
Keep discovery and input selection unconditional. Scalar header values such as
`atten_drive`, `obs_type_code` and `n_chans` are available to selection and
conditions. Tune block counts are not computed during input preparation.

## Outputs and deployment

Each pipeline owns its output settings. Configure KIDs products with
`kids.output.path` and enable writing with `kids.output.enabled: true`.
The launcher convenience option `--dataprod_toltec_path` supplies a default for
`kids.output.path`; explicit recipe settings and `--kids.output.path` take
precedence. Setting a path alone does not enable output or create directories.

Kids does not discover a deployment root or derive paths from environment
variables. Deploy can supply explicit paths and forward the same CLI arguments:

```console
tolteca_deploy run -- tolteca_kids --config_paths check.yaml --dry_run
```

Options before Deploy's `--` configure the launcher; options after it belong to
Kids. A config passed only to Deploy is not automatically passed to Kids. See
the [observation-file guide](guide.md) for Python workflows.

Add `--log_level DEBUG` to trace discovery, header population, selection,
per-input configuration and step execution. Logs include paths and configuration
values, zero-based discovery entry indices, candidate/selection counts, and
duplicate-removal counts; handle saved logs accordingly. A successful dry run
prints `<source path>: no-op completed` for each selected input in table order,
including inputs sharing an acquisition. Failures return a nonzero exit status.
