# TolTEC Kids Submodule - Migration Guide

## Overview

The TolTEC data reduction workflow has been refactored to follow best practices from `kidsproc` and `D21Analysis`, separating raw I/O from data processing with proper namespacing.

## What Changed

### Old Approach (Deprecated)
```python
from tolteca_datamodels.toltec.io import open_toltec, reduce_raw_sweep

ds = open_toltec("sweep.nc")
ds_reduced = reduce_raw_sweep(ds)  # Creates variables: "I", "Q", etc.
```

### New Approach (Recommended)
```python
from tolteca_datamodels.toltec import open_toltec, SweepReducer

ds = open_toltec("sweep.nc")
reducer = SweepReducer(detect_blocks=True, compute_uncertainty=True)
ds_reduced = reducer(ds)  # Creates namespaced variables
```

## Key Changes

### 1. Namespace Simplification
- **Old**: `tolteca_datamodels.toltec.toltec_kids.sweep.I`
- **New**: `tolteca_datamodels.toltec.kids.sweep.I`

### 2. Module Structure
```
tolteca_datamodels/toltec/
├── __init__.py (public API + backward compatibility)
├── accessor.py (ToltecKidsAccessor)
└── kids/ (NEW - replaces io.py)
    ├── core.py (ToltecKidsIOSchema - raw data, no namespacing)
    ├── sweep.py (ToltecKidsSchema, SweepReducer, ReducedSweepView)
    └── timestream.py (placeholder)
```

### 3. Backward Compatibility
The old `reduce_raw_sweep()` function still works:
```python
from tolteca_datamodels.toltec import reduce_raw_sweep

ds_reduced = reduce_raw_sweep(ds)  # Uses SweepReducer internally
```

But new code should use `SweepReducer` directly for better control.

## SweepReducer Features

### Configuration Options
```python
reducer = SweepReducer(
    detect_blocks=True,         # Auto-detect multi-block tune files
    compute_uncertainty=True,   # Calculate std for each sweep
    sweep_axis=None,            # Auto-detect from LO frequency
    time_axis="time",           # Time dimension name
)
```

### Automatic Features
- **Sweep detection**: Finds sweep steps from LO frequency changes
- **Multi-block handling**: Detects breaks in frequency (tune files)
- **Uncertainty**: Computes std/√N for each sweep step
- **Caching**: Results cached in dataset attrs
- **Metadata preservation**: All original metadata retained

### Output Format
Reduced dataset has namespaced variables:
```python
ds_reduced.data_vars:
  - tolteca_datamodels.toltec.kids.sweep.I (channel, sweep)
  - tolteca_datamodels.toltec.kids.sweep.Q (channel, sweep)
  - tolteca_datamodels.toltec.kids.sweep.unc_I (channel, sweep)
  - tolteca_datamodels.toltec.kids.sweep.unc_Q (channel, sweep)
  - f_lo (sweep) - LO frequency for each sweep step
  - [all original metadata preserved]
```

## Working with Namespaced Variables

### Accessing Data
```python
# Get the namespace
NAMESPACE = "tolteca_datamodels.toltec.kids.sweep"

# Access variables
I = ds_reduced[f"{NAMESPACE}.I"]
Q = ds_reduced[f"{NAMESPACE}.Q"]
unc_I = ds_reduced[f"{NAMESPACE}.unc_I"]
unc_Q = ds_reduced[f"{NAMESPACE}.unc_Q"]
```

### Using ReducedSweepView
```python
from tolteca_datamodels.toltec.kids import ReducedSweepView

view = ReducedSweepView(ds_reduced)

# Convenient property access
I = view.I  # DataArray with proper dimensions
Q = view.Q
unc_I = view.unc_I
unc_Q = view.unc_Q

# Metadata access
f_lo = view.f_lo
sweep_offset = view.sweep  # Frequency offsets from center
n_blocks = view.n_blocks
is_multi_block = view.is_multi_block
```

## Multi-Block Data

For tune files with multiple frequency sweeps:
```python
reducer = SweepReducer(detect_blocks=True)
ds_reduced = reducer(ds_tune)

if ds_reduced.attrs["is_multi_block"]:
    # Data has shape (block, channel, sweep)
    n_blocks = ds_reduced.sizes["block"]
    I_block0 = ds_reduced[f"{NAMESPACE}.I"][0, :, :]
```

## Testing

### Test Organization
```
tests/tolteca_datamodels/
├── test_toltec_accessor.py (21 tests) - Accessor functionality
└── test_sweep_reducer_edge_cases.py (13 tests) - Edge cases
```

### Running Tests
```bash
# All TolTEC tests
pytest tests/tolteca_datamodels/test_toltec_accessor.py -v

# Edge case tests
pytest tests/tolteca_datamodels/test_sweep_reducer_edge_cases.py -v

# Both
pytest tests/tolteca_datamodels/ -k "toltec" -v
```

## Breaking Changes

### None for Normal Usage
- `open_toltec()` still works
- `reduce_raw_sweep()` wrapper provided
- Existing scripts should continue to work

### For Advanced Users
- `io.py` module removed (use `kids` submodule)
- `create_sweep_dataset()` removed (use `SweepReducer`)
- Variable names are now namespaced (use view or construct names)

## Benefits of New Approach

1. **Cleaner separation**: Raw I/O vs. processed data
2. **Better namespacing**: Avoids variable name collisions
3. **Following patterns**: Consistent with D21Analysis
4. **More flexible**: Configure reducer behavior
5. **Better tested**: 34 tests including edge cases
6. **Easier to extend**: Add timestream, other data types

## Examples

### Basic Reduction
```python
from tolteca_datamodels.toltec import open_toltec, SweepReducer

# Load and reduce
ds = open_toltec("vnasweep.nc")
reducer = SweepReducer()
ds_reduced = reducer(ds)

# Access data via view
from tolteca_datamodels.toltec.kids import ReducedSweepView
view = ReducedSweepView(ds_reduced)

print(f"Reduced to {view.I.shape[1]} sweeps, {view.I.shape[0]} channels")
```

### Multi-Block Tune File
```python
from tolteca_datamodels.toltec import open_toltec, SweepReducer

ds = open_toltec("tune.nc")
reducer = SweepReducer(detect_blocks=True)
ds_reduced = reducer(ds)

if ds_reduced.attrs["is_multi_block"]:
    n_blocks = ds_reduced.attrs["n_blocks"]
    print(f"Detected {n_blocks} blocks in tune file")
```

### Custom Configuration
```python
reducer = SweepReducer(
    detect_blocks=False,  # Force single block
    compute_uncertainty=False,  # Skip uncertainty calculation
)
ds_reduced = reducer(ds)
```

## Getting Help

- Check docstrings: `help(SweepReducer)`
- Check tests for examples
- See NAMESPACE_SIMPLIFICATION_SUMMARY.md for implementation details
