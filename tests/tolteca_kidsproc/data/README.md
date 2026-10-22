# Phase-fit regression extracts

These small extracts come from network 0 of observation 145484, acquired in
seven sweeps at drive attenuations 4–16 dB. Source: `tolteca_ref_data` revision
`9cdbc76c4327b11e9ec45ce7895ef13b537d03a5`,
`data_lmt/toltec/tcs/toltec0/*145484*targsweep.nc`.

- `channel596.npz`: the reduced frequency/I/Q arrays, power axis and tone
  frequency for recorded channel index 596. Reduction used TolTECA delivery
  revision `f8bd7c27674ba002d29bdd4ede4b496c5acf3e61`. Tiny I/Q changes made
  the legacy fitter alternate between an accepted crossing and a flagged
  endpoint. The test checks numerical repeatability through the full channel
  fitter and unchanged selection policy.
- `channel586-phase.npz`: the original nonlinear phase objective at 6 dB for
  channel 586: frequency `f`, phase `y`, standard-deviation weights `sigma`,
  initial parameters `p0`, circle radius `r`, and rotated complex samples `z`.
  It exercises two distinct minima and the optional, bounded alternate start.
  A smaller residual is not itself a claim of physical correctness.

The extracts are numerical arrays only and load with NumPy's default
`allow_pickle=False`. Full replay instructions and measured timing belong in
`delivery/drive-fit-normalization/phase-fit-stability.md`.
