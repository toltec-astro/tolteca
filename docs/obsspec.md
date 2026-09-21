# Observation selectors

`ObsSpec` selects observation and interface metadata. It does not identify a
database row or open data. Import it from
`tolteca_datamodels.toltec.obsspec`; `tolteca_db.obsspec` exports the same class.

The full form is `[master-]obsnum-subobsnum-scannum-interface`. Trailing fields
may be omitted. A dash assigns fields from the left; slash-separated fields are
aligned to the right. Masters are `tcs`, `ics`, and `clip`.

| Selector | Selection |
| --- | --- |
| `1000-2` | Observation 1000, subobservation 2 |
| `1000/7` | Observation 1000, ROACH 7 |
| `1000/2/7` | Observation 1000, scan 2, ROACH 7 |
| `1000-2/7` | Observation 1000, subobservation 2, ROACH 7 |
| `tcs-1000/toltec7` | TCS observation 1000, ROACH 7 |
| `1000/hwpr` | Observation 1000, interface hwpr |
| `1000-{0,2}/[0:7]` | Subobservations 0 or 2, ROACHes 0 through 6 |

Each numeric field accepts a nonnegative integer, a set `{0,2,5}`, or a half-open
range `[start:stop:step]`. The step defaults to one and must be positive. Bounds
may be omitted: `[5:]` has no upper bound, `[:5]` starts at zero. Ranges select
**identifier values**, not row positions. `*`, `{}`, `[]`, and `[:]` leave a field
unconstrained. Quote selectors containing shell metacharacters.

The last field accepts ROACH IDs 0 through 12, their `toltecN` names, or an exact
interface name. Lists and ranges there select ROACH IDs. Malformed expressions
raise; they never fall back to an unrestricted query. Prefix globs such as
`123*`, negative identifiers, and product/association selectors are not supported.
An empty string is not a wildcard. Latest-observation policy is separate: use
the database query's `get_raw_obs_latest()` method.

```python
from tolteca_datamodels.toltec.obsspec import ObsSpec

spec = ObsSpec.parse("tcs-1000-{0,2}/7")
selected = spec.select(catalog)
# The same parsed selector can be passed to a database ObsQuery:
selected_from_db = query.get_raw_obs_info_table(spec)
```

Local CLI discovery searches the NetCDF acquisition layout beneath
`data_root/toltec/<master>/<interface>/`. Other layouts can be supplied as
explicit paths or quoted filesystem globs. Input strings with a file extension,
or beginning with `/`, `./`, `../`, or `~/`, are filesystem inputs rather than
selectors. Python callers can pass a `Path` for an extensionless filename.
Selectors constrain only supplied fields; an unknown constrained metadata value
does not match. Repeated file matches are deduplicated before execution.
