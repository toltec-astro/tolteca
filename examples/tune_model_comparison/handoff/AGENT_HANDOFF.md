# Agent handoff: run the Tune-model comparison on a new observation

Read README.md and the focused PDF here before changing code. The comparison
baseline is Python ModelFit with full nonlinear M-channel/N-resonator groups.
Kidscpp/Citlali consume that same fitted model through the common ECSV/stub
contract. Do not replace it with a separate singlet-only fit or a private solver.

1. Record this Tolteca commit and fetch the exact revisions in revisions.json.
   Preserve unrelated user checkouts; bootstrap into a new directory. The
   pinned native branch is required for the stub and component APT interfaces.
2. Obtain the observation's raw KIDs/telescope files, its header-recorded
   calibration Tune/sweep and original observation-matched beammap APT. Ask
   for the missing files; do not manufacture detector identities or substitute
   a nearby Tune. The reference native YAML should already describe the
   desired science reduction and legacy calibration.
3. Run run_observation.py. Fit through the installed tolteca_kids workflow,
   retain both Tune blocks, and let the raw context select the recorded block
   for the target. Inspect channels.ecsv and components.ecsv before interpreting
   maps. Keep original APT bad flags and ambiguous/missing identities explicit.
4. Use the same scientific settings, APT optical calibration and weighting
   prescription for the legacy control. Keep sample policies distinct. Numerical
   acceptance still rejects hard failures and does not enlarge the joint search
   domain. Do not silently relax thresholds to obtain comparable populations.
5. Compare all three arrays: actual contributors, singlet/multiplet coverage,
   nonlinearity/fit flags, sample acceptance, map RMS on identical sky pixels
   and fitted source properties. Saved timestreams/PSDs are useful diagnostics,
   but one observation's annulus RMS is not a mapping-speed measurement.
6. Return the exact configs, source/binary hashes, indexes, ledgers, native logs,
   execution receipts and map figures. Record failures and unsupported groups.
   The next independent calibration task is a beammap-derived V3 APT using
   this same solving path, not a prerequisite to trying this comparison release.

Known unresolved issues are in the PDF appendix and README, including 145609
Python/native group parity, per-row rather than full joint-covariance map
weights, and V2-derived optical calibration. This package does not request
changes to Tlaloc, production services, shared DPDB tags or remote acquisition
machines. Its data access and output paths are supplied by the recipient.
