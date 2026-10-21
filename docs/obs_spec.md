# Observation selectors

`ObsSpec` selects observation and interface metadata. It does not identify a
database row or open data. Import it from
`tolteca_datamodels.lmt_base.obs_spec`.

Parsing records the constraints without choosing an instrument. File stores
and file-table accessors supply interface label interpretations when applying a
selector. The numbered-interface examples below use the TolTEC context.

The full form is `[master-]obsnum-subobsnum-scannum-interface`. Trailing fields
may be omitted. A dash assigns fields from the left; slash-separated fields are
aligned to the right. Master labels include `tcs`, `ics`, and `clip`.

`obsnum` is the primary field: the first numeric or unconstrained field selects
it. A preceding label or label set selects master and must be separated by a
dash. Omit an unconstrained master. Thus `1` selects observation
1 from any master, while `*-1` selects subobservation 1 from any observation.
`tcs-1` and the full quartet `tcs-1-1-1` are both accepted. A wildcard can follow
an explicit master: `tcs-*-1` selects TCS subobservation 1 from any observation.
`{tcs,ics}-(100,200)[-1]` selects the largest observation number in those bounds
from either master.

| Selector | Selection |
| --- | --- |
| `1000-2` | Observation 1000, subobservation 2 |
| `1000/7` | Observation 1000, numbered interface 7 |
| `1000/2/7` | Observation 1000, scan 2, numbered interface 7 |
| `1000-2/7` | Observation 1000, subobservation 2, numbered interface 7 |
| `tcs-1000/toltec7` | TCS observation 1000, ROACH 7 |
| `1000/hwpr` | Observation 1000, interface hwpr |
| `1000/toltec_hk` | Observation 1000, interface toltec_hk |
| `1000/{tel,1,3}` | Observation 1000, telescope or numbered interface 1 or 3 |
| `{tcs,ics}-1000` | Observation 1000 from either master |
| `145609/tel` | Observation 145609, telescope interface |
| `1000-{0,2}/toltec(0,7)` | Subobservations 0 or 2, ROACHes 0 through 6 |
| `(100,200)[-1]/(1,5)` | Largest obsnum in the bounds with a matching numbered interface |
| `(100,200)/(1,5)[-1]` | Largest interface number in the bounds with a matching obsnum |
| `[-1]/7` | Largest available obsnum for numbered interface 7 |

## Values and positions

Fields have three kinds: `label` for master, `integer` for obsnum, subobsnum and
scannum, and `mixed` for interface. Integer selectors use nonnegative values;
mixed fields also accept numbered labels and numeric-slot patterns. A set lists
alternatives within a field; constraints on different fields all have to match.
Master accepts labels; obsnum, subobsnum and scannum accept sequence values;
interface accepts both. Quote selectors containing shell metacharacters.
Whitespace is not allowed, including before or after the expression.

Every field uses the same expression grammar, followed by validation against
that field's constraints. A numeric-only field rejects literal parts with a
validation error: `(100,200)[-1]` is valid for obsnum, but
`foo(100,200)[-1]bar` is not. Invalid expressions raise; valid expressions with
no available matches return an empty selection.

Sequence values support bounds and positional selection:

| Value constraint | Meaning |
| --- | --- |
| `100` | Exactly 100 |
| `{100,150,199}` | Any listed value |
| `(100,200)` | At least 100 and less than 200 |
| `(100,)` | At least 100 |
| `(,200)` | Less than 200 |
| `()`, `(,)` | Any numeric value |
| `{,}`, `{}`, `*`, or omitted | Any value, whether label or numeric |

`*` is the canonical unrestricted form: `{}` and `{,}` normalize to `*`.
Empty parentheses `()` normalize to `(,)`. Parentheses always require numeric
values, even without bounds; a label-only field rejects them. For example,
`1/*` includes telescope and auxiliary interfaces, while `1/()` includes only
interfaces with a numeric interpretation.

All value constraints, including master and interface labels, are applied first.
A positional selector then acts on the **sorted, distinct, non-null values** of
one field among the remaining records:

| Positional selector | Meaning |
| --- | --- |
| `[0]` | Smallest available value |
| `[-1]` | Largest available value |
| `[:10]` | First ten available values |
| `[-10:]` | Last ten available values |
| `[::2]` | Every other available value, starting at the smallest |
| `[-1,-3]` | Largest and third-largest available values |
| `[]`, `[:]`, or omitted | No positional selection |

Positions may follow a whole-field value, set, wildcard, or numeric bounds.
`{100,199}[-1]`, `100[-1]`, and `{tcs,ics}[-1]` are valid syntax. Applying the
positions requires a common numeric ordering for every selected value.
For labels, this is determined from the declared context, not from which files
happen to be present. If no common ordering is available, the positions are
ignored with a debug message, while the value constraint still applies. Labels
are not sorted lexicographically or discarded to make the remainder sortable.

An entire field may omit the wildcard: `[-1]` is shorthand for `*[-1]`.
In a numeric-only field such as obsnum, this selects the largest available
value. In the mixed interface field, use `()[-1]` to restrict the selection to
numeric interfaces; `*[-1]` retains all interfaces because the declared context
also contains unordered labels such as `tel` and `toltec_hk`.

Slice bounds clip to the available sequence. An out-of-range index selects
nothing; out-of-range members of a positional list are ignored. Negative slice
steps are allowed, but zero is invalid. Selection retains **all records** with
the selected values and preserves their original row order, even for negative
steps. It does not return just one row per distinct value.

At most **one effective positional selector** is allowed in an ObsSpec.
`(100,200)[-1]/(1,5)[-1]` is therefore invalid. Empty/full forward slices such as
`[]`, `[:]`, `[::1]`, and `[0:]` are omitted during normalization and do not count
toward that limit. A whole-field wildcard with no effective positional selector
does not filter the field, including records whose value is null.

For example, `(100,200)[-1]/(1,5)` first finds records with
`100 <= obsnum < 200` and interface numbers from 1 through 4, then selects the largest obsnum
represented by those records. `*[-1]` and `[-1]` are equivalent. Bare negative
numbers and negative members of literal sets are invalid: positions belong in
square brackets.

Labels are opaque: digits in `foo01` are not split off or interpreted as a
number. Bare numeric constraints use a label-to-number mapping supplied by the
resolution context. TolTEC provides its interface-to-ROACH mapping, so `1/1`
selects `toltec1`. Unmapped labels do not match bare numeric constraints.

Literal parts can surround an explicit numeric slot:

| Interface expression | Selection |
| --- | --- |
| `foo01` | The exact label `foo01` |
| `foo0(1,2)` | Numeric component 1 after literal `foo0`, including `foo01` |
| `foo()bar` | Any numeric component between `foo` and `bar` |
| `foo(100,)bar` | Numeric component at least 100 between those literals |
| `foo(,200)bar` | Numeric component less than 200 between those literals |
| `foo(100,200)bar` | Numeric component from 100 through 199 between `foo` and `bar` |
| `foo(100,200)[-1]bar` | Largest available component in those bounds, after all value filters |
| `foo()[-1]bar` | Largest available component between those literals, without bounds |
| `toltec()` | Any numeric component immediately after `toltec`, not `toltec_hk` |

An explicit slot identifies its numeric component without a label-to-number
mapping. Matching uses available labels, not generated filenames. Positions
immediately follow the parentheses, before the trailing literal part. One
contiguous numeric slot is supported per field expression.
Only parentheses may occur between literal parts: `foo[-1]bar`, `foo*bar`,
`foo{}bar`, and `toltec{1,3}` are invalid. `*` and sets are whole-field forms.
`foo(,)bar` and `foo(,)[-1]bar` are equivalent to `foo()bar` and
`foo()[-1]bar`. To select whole labels instead, use a whole-field set such as
`{toltec1,toltec3}`.

Parsing does not require an installed instrument or validate hardware IDs.
Stores resolve the selector against their declared interfaces; tables match
against the interface values they contain. Unavailable names select nothing.
Other interface names, including `tel`, `hwpr`, `toltec_hk`, and `wyatt`, are
matched literally. There are no interface aliases: `hk` does not mean
`toltec_hk`. Malformed expressions raise; they never fall back to an unrestricted
query. Arbitrary-text globs, negative identifiers, and product/association
selectors are not supported.
An empty string is not a wildcard.

`{tel,1,3}` matches any of the three alternatives. A positional selector on
another field still works:
`[-1]/{tel,1,3}` selects the largest obsnum represented
by any of those interfaces.
Interface positions order distinct resolved numeric values, not label strings:
`toltec10` follows `toltec2`. If multiple prefixes share the selected component,
all of them match unless the selector restricts the prefix.

For `{toltec1,toltec5}[-1]`, both labels have a numeric ordering, so the largest
component present after all value filters is selected. For
`{tel,toltec1,toltec5}[-1]`, the positions are ignored and all three alternatives
remain selected. This remains true even if no telescope file is present.
Likewise, `{tcs,ics}[-1]-1` selects both masters for obsnum 1.

Observation-number sequences are independent for each acquisition master.
`tcs-[-1]` selects the largest TCS observation number; bare `[-1]` compares numbers
across all available masters, not acquisition times. Neither selector consults
file creation times or the readout system's current-file symlinks. Resolving
those symlinks is a separate file-discovery operation.

```python
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

spec = ObsSpec.parse("tcs-1000-{0,2}/7")
selected = table.toltec_file.query_obs_spec(spec)
```

Python construction uses the same five fields as the text expression:

```python
spec = ObsSpec(master="tcs", obsnum=1000, interface=("tel", 1, 3))
```

Construction raises `ValueError` if a leading field would be mistaken for
obsnum in the normalized text. Leave master as `None` when it is unconstrained;
use a label or label set to constrain it. A wildcard positional selector cannot
be assigned to master.

The interface selector reads the `interface` column, including for numeric
constraints. It does not need a `roach` column.

Query a file table through its accessor; it supplies the label interpretations
for the selected data models. For TolTEC data, including telescope companions:

```python
from tolteca_datamodels.toltec.file import ToltecFileAccessor

selected = ToltecFileAccessor(table).query_obs_spec(spec)
```

For telescope-only interpretations:

```python
from tolteca_datamodels.lmt.file import LmtFileAccessor

selected = LmtFileAccessor(table).query_obs_spec(spec)
```

The generic `FileAccessor` combines the interpretations of its selected handlers.
There is no need to import or combine instrument contexts when querying file
tables.

## Defining label interpretations for an extension

Data-model extensions can create an `ObsSpecContext` to declare numeric
interpretations of labels and reuse it across their file accessor and store.
The same parsed selector can be applied with different contexts.

Without a context, literal labels, numeric columns and explicit numeric-slot
patterns work, but bare numeric constraints do not match string labels.
Contexts do not restrict literal matching to the declared labels.

```python
from tolteca_datamodels.core.obs_spec import ObsSpecContext

numbers = {"foo00": 0, "foo01": 1}
context = ObsSpecContext(field_label_numbers={"interface": numbers})
spec = ObsSpec.parse("1/(0,2)[-1]")
selected = spec.select(table, context=context)
candidates = spec.resolve_values("interface", list(numbers), context=context)
assert candidates == ["foo00", "foo01"]  # Positions are deferred during discovery.
```

Include unnumbered labels with None when declaring a context. For example,
`{"tel": None, "foo00": 0, "foo01": 1}` keeps wildcard interface positions
unordered even when the table happens to contain only numbered interfaces.
Contexts are immutable. Combining them with `|` accepts identical
declarations and raises for conflicting numeric values for the same field and
label. Creating or importing another context does not change existing selections.

## Configuration fields

Use `ObsSpecField` to accept selector text, an observation number, or an existing
`ObsSpec` in a Pydantic model. The validated field holds an `ObsSpec`; both Python
and JSON dumps contain its normalized string.

```python
from tollan.config import FrozenBaseModel
from tolteca_datamodels.lmt_base.obs_spec import ObsSpecField

class SelectionConfig(FrozenBaseModel):
    obs_spec: ObsSpecField

config = SelectionConfig.model_validate({"obs_spec": "tcs-1000/toltec7"})
selected = table.toltec_file.query_obs_spec(config.obs_spec)
assert config.model_dump() == {"obs_spec": "tcs-1000/toltec7"}
assert SelectionConfig.model_validate_json(config.model_dump_json()) == config
```

`str(spec)` produces the same normalized text. Normalization removes numeric
padding, sorts and deduplicates sets, and omits trailing unconstrained fields.
Literal text retains its spelling. Single positions become equivalent slices
and the wildcard is explicit (`[-1]` becomes `*[-1:]`). Parsing the normalized
string preserves the selector's meaning, not the original spelling.
Removing a no-op slice does not remove an explicit numeric slot:
`foo()[:]bar` and `foo(,)[:]bar` normalize to `foo(,)bar`, not the exact label
`foobar`. Unbounded numeric selectors retain their parentheses; they never
normalize to `*`.

When the final field is constrained, the rightmost middle run of unconstrained
fields is omitted and the remaining suffix uses `/`:

```text
1000-*-*-7  -> 1000/7
1000-*-2-7  -> 1000/2/7
1000-2-*-7  -> 1000-2/7
```

If the final field is unconstrained, dash notation is retained: `1000-*-2-*`
becomes `1000-*-2`, not `1000/2`. The former selects scan 2; the latter selects
numbered interface 2. An unconstrained primary field remains `*`, as in `*/tel`.

## File discovery

LMT file discovery searches the telescope directory `data_root/tel/` and
registered instrument trees, including `data_root/toltec/<master>/<interface>/`.
An unqualified `145609` or wildcard `145609/*` searches both. `145609/tel`
searches only the telescope directory; interface constraints excluding `tel`,
such as `145609/0`, skip that directory. Telescope streams share interface `tel`:
select their literal prefix afterward with `file_prefix == 'tel_toltec'`.
Telescope paths are identified with master `tcs`, so `tcs-145609/tel` also matches
them. Other layouts can be
supplied with an `input_discovery.inputs` entry containing exact `sources` or explicit
`{glob: pattern}` resolvers. Choose `mode: guess_or_read` for renamed files,
or `mode: read` when even recognizable filenames may contain incorrect identifiers.
Selectors constrain only supplied fields; an unknown constrained metadata value
does not match. Repeated file matches are deduplicated before execution.
An explicit interface set containing both `tel` and TolTEC alternatives searches
both directory trees before applying the complete selector.

For candidate paths alone, use `LmtFileStore` from
`tolteca_datamodels.lmt.file_store`. Its `glob_obs_spec()` method uses directory
and filename globs without reading headers. Use
`tolteca_datamodels.input_discovery.InputDiscovery` when you also need exact selector
matching and a combined file table; see [configured input discovery](kids-inputs.md).

Every file store can have substores registered by class. Use
`store.substores[ToltecFileStore]` to obtain a child with its concrete return type.
The mapping is read-only, and repeated access returns the same instance.
`store.toltec` provides a bound selection view of LMT's TolTEC subtree.
Register extensions before constructing stores. Construction creates the child
stores without searching directories or opening files. Later registrations
affect only new stores; existing instances and their views retain their children.
Declare a named child in the parent class with
`toltec = Substore(ToltecFileStore, subpath="toltec")`.
Class creation registers its path and makes the selection descriptor available.
For children without descriptor names, use a `_substore_paths` mapping such as
`{ToltecFileStore: PurePosixPath("toltec")}`. Subclasses inherit this mapping unless
they replace it; new descriptors add their children. Both declarations use
`register_substore()` for validation. Extensions register a child with
`ParentStore.register_substore(ChildStore, subpath, name="child")`; the optional
name makes `store.child` and `ParentStore.child` selection descriptors available.
The dummy instrument registers its store
when `tolteca_dummy_instru.register()` is called. Each child is rooted at
`parent.path / subpath`. Subpaths use Tollan's `SubPath` contract; for example,
`"./instruments//toltec/"` normalizes to `PurePosixPath("instruments/toltec")`.
Each class has one registered subpath per parent. Registering that class again
raises `ValueError`, even at the same subpath.
Looking up an unregistered class raises `KeyError`.
Discovery searches local files first,
then substores recursively, retaining the first occurrence of each candidate.
Use `store.partial("tel", "toltec").glob_obs_spec(spec)` to search telescope and
TolTEC locations. `store.partial(exclude=("tel",))` searches the other declared
locations. Positional selections and `exclude=` cannot be combined; omitting
both raises `ValueError`. An empty `exclude` removes nothing.
These calls return immutable views. Class descriptors are also accepted:
`store.partial(LmtFileStore.tel, LmtFileStore.toltec)` selects telescope files and
the TolTEC subtree. Further `partial()` calls can only narrow the selection.
Use ObsSpec for master/interface constraints, for example `"tcs-1/{tel,7}"`.

Call `iter_glob_patterns(spec)` on a store or bound view to inspect the candidate
patterns without searching. Store patterns are relative to `store.path`; view
patterns are relative to `view.root_path`, even when its selected `path` is more
specific. TolTEC manages its root with fixed-depth master/interface globs;
an unconstrained master uses a wildcard directory. Positions remain
deferred to the populated table, including when a view selects fewer directories.

Store subclasses implement `_iter_local_glob_patterns(spec, subpath)` for their
own managed locations. Each pattern is relative to that location; public
`iter_glob_patterns(spec)` includes selected substores and reports paths relative
to the entry root.

Substores may support different selector types. Pass selector text to parse it
using the entry store's type, or pass an already parsed selector. A store that
does not support that selector skips its local files but still visits its
substores. Invalid selector text raises. Concrete stores supply their selector
interpretations automatically. Configured input discovery also applies the full
selector after populating the candidate table.

Each instrument store resolves interface value constraints against its own
declared interfaces and numeric mapping before globbing. For TolTEC, `1/(1,5)` and `1/toltec(1,5)`
search the same four ROACH directories. Another store with interfaces `bar0`
through `bar3`, mapped to numbers 0 through 3, resolves the bare bounds to
`bar1` through `bar3`; it finds no
matches for the explicit `toltec` prefix. The instrument name need not equal
its interface prefix.

Positional selection is deferred until the discovered table has been populated
and all value constraints applied. The configured file accessor's
`query_obs_spec()` selects from the combined candidate table using that
accessor's label interpretations. For example, `1/(1,5)[-1]` still searches
every eligible directory. If only interface 1 has files for observation 1, those
files are selected. Candidate order follows declared interfaces and then glob
enumeration; paths are not sorted.
