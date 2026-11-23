# Detector response and timestream quality

Status: scientific review draft. Evidence checked on 2026-11-21. Section 3.4 reports the group-aware comparison; sections 3.2–3.3
retain the earlier isolated-response and census results. Absolute optical calibration and accepted 145609 roster
associations remain unestablished.

## 1. Separate optical response from readout suitability

A noise comparison must first identify what a detector measures and whether its
readout is usable at the acquired operating point. Multiplicity and fitted
nonlinearity describe the model within those populations. They do not define
the populations or establish sensitivity.

**Table 1. Independent assessments for each detector observation.**

| Assessment | Evidence | What it does not establish |
| --- | --- | --- |
| Detector identity and observation association | Exact APT reference, UID, acquisition, block and original finding links; retain tentative and ambiguous associations | Optical activity or a good calibration |
| Response population | Measured trajectories under a specified loading or power experiment | A universal physical class |
| Response class | Optical response, dark-candidate evidence, or unknown; with experimental support | Suitable readout in another acquisition |
| Calibration and readout suitability | Sweep adequacy, applied power and probe placement, stability, model support, solving validity and signal recovery | Optical sensitivity from numerical convergence alone |
| Model structure | Singlet/multiplet, fitted nonlinearity, parameter bounds and joint conditioning | A detector-quality ranking |

The useful quality matrix is therefore **response class × readout suitability**,
subdivided by multiplicity and nonlinearity. Optically responsive detectors with
qualified readout enter astronomical sensitivity estimates. Responsive detectors
with inadequate readout remain in the failure accounting. Dark detectors have a
separate role as possible monitors of readout or non-optical disturbances.
Unclassified components remain unclassified; the complement of a dark list is
not automatically an active-detector list.

The KIDs/KIDsproc work distinguishes an established identity from a physical
response class. Its maintained roster initially records `response_class=unknown`.
Nearly stationary Skydip tracks support a dark-candidate interpretation, but bath
temperature, readout drift and optical loading need not be independently controlled.
The existing finder `dark` flag is a Qr/peak-quality screen, not an optical-response
measurement. A high fitted Qr is useful evidence, especially with the known TolTEC
population, but is not sufficient to assign a persistent UID or certify darkness.

## 2. Ideal signal and noise

At one fixed operating point, write the intrinsic state perturbation as

\[
\begin{pmatrix}\delta r\\\delta x_{\mathrm{opt}}\end{pmatrix}
=\begin{pmatrix}R_r\\R_x\end{pmatrix}\delta P_{\mathrm{opt}}
+\begin{pmatrix}B_r\\B_x\end{pmatrix}\delta T_{\mathrm{bath}}
+\begin{pmatrix}\eta_r\\\eta_x\end{pmatrix}.
\]

The last term includes remaining intrinsic fluctuations. Measurement also adds
readout noise and calibration error. These terms can be correlated. The response
slope \(\lambda=R_r/R_x=1/\beta\), with \(\beta=R_x/R_r\), describes the
optical perturbation when \(R_x\ne0\);
it is not the regression slope of every measured fluctuation.

**Table 2. Ideal expectations and the corresponding checks.**

| Case | Optical response | Expected noise in the stated ideal limit | Required comparison |
| --- | --- | --- | --- |
| Active detector, weak nonlinear correction, adequate readout | Nonzero \(R_x\), with measured \(R_r/R_x\) | Optical fluctuations move along the response direction; additive readout noise need not | Optical signal transfer and noise referred to that signal |
| Dark detector, weak nonlinear correction, readout-noise dominated | Both optical response components approximately zero; optical response ratio is undefined | Equal loss/effective-detuning PSDs for locally isotropic additive IQ noise | Test PSD ratio and cross-spectrum; check optical leakage and non-optical response |
| Nonlinear but adequately modeled singlet | Same physical response, transformed by the calibrated nonlinear map | Loss noise mixes into returned optical detuning; equality of the two output PSDs is no longer expected | Transform signal and covariance together; assess branch stability and calibration error |
| Detector with inadequate readout or calibration, active or dark | May exist physically, but is not reliably measured | No guaranteed PSD or response relation | Retain bounds, residuals, missing samples and branch/rank failures; do not infer sensitivity from a small ASD |

“Dark” does not require equal measured noise in every circumstance. Intrinsic
frequency noise, bath response and electronics can break the readout-only limit.
The distinction between reactive noise and optical response in nonlinear readout
is also discussed by [Swenson et al. (2013), section III](https://arxiv.org/html/1305.4281v1).
Their device result does not establish TolTEC's optimal drive setting.

### 2.1 Effective detuning and optical detuning

For a calibrated singlet, the complex response gives

\[
w=\frac{c}{1-z/H}=r+i x',\qquad
x_{\mathrm{opt}}=x'-\frac{b}{r^2+x'^2}.
\]

Here \(H\) and \(c\) are the fitted gain and coupling at the fixed probe;
\(b=2a_{\mathrm{ref}}r_{\mathrm{ref}}^3\) is held fixed during local solving. This is a
calibration hypothesis, not a new instantaneous measurement of microwave power.
Rewriting solved loss as \(b/(2r(t)^3)\) provides no independent test of it.
Independent sweep fits over changing loading are required to test that hypothesis.

For small perturbations on a stable branch, define

\[
D=1+\frac{2b x'}{(r^2+x'^2)^2},\qquad
\mu=\frac{2br}{(r^2+x'^2)^2},\qquad
J=\begin{pmatrix}1&0\\\mu&D\end{pmatrix}.
\]

Then \(\delta x_{\mathrm{opt}}=D\delta x'+\mu\delta r\). For a measured spectral
matrix \(C'(\nu)\) in coordinate order \((r,x')\),

\[
C_{\mathrm{opt}}(\nu)=J C'(\nu)J^T,
\quad
S_{x_{\mathrm{opt}}}=D^2 S_{x'}+\mu^2 S_r+2D\mu\operatorname{Re}S_{rx'}.
\]

Locally isotropic additive IQ noise passes through the complex singlet map into
equal, uncorrelated \(r,x'\) noise. In that limit,
\(S_r=S_{x'}\), \(S_{rx'}=0\), and
\(S_{x_{\mathrm{opt}}}/S_r=D^2+\mu^2\). Equal noise in \(r\) and \(x_{\mathrm{opt}}\)
is recovered when the nonlinear correction is negligible. A heavily overdriven
dark detector need not satisfy that equality even with ideal additive readout noise.

If loading changes \(b\), the additional term is
\(-\delta b/(r^2+x'^2)\). Fixed applied tone power alone does not guarantee that
coupling and the nonlinear energy scale remain fixed across a large loading range.
The local fixed-b calculation must not replace the independent Skydip response law.

### 2.2 Signal transfer and information

For an active detector with intrinsic optical response
\(s_{\mathrm{opt}}=(R_r,R_x)^T\), the response before the correction is

\[
s'=\begin{pmatrix}R_r\\(R_x-\mu R_r)/D\end{pmatrix},\qquad
\frac{\delta x'}{\delta x_{\mathrm{opt}}}=\frac{1-\mu\lambda}{D}.
\]

Thus comparing the two ASDs without their signal transfer is not an S/N comparison.
With a nonsingular local map, correct calibration and both coordinates retained,
\(s_{\mathrm{opt}}^{\dagger}C_{\mathrm{opt}}^{-1}s_{\mathrm{opt}}
=s'^{\dagger}C'^{-1}s'\). Changing coordinates does not create or destroy that
information. Discarding a coordinate, using an incorrect calibration, unresolved
branches, or rank loss changes the conclusion. Multiplets require the full joint
Jacobian and covariance; a singlet formula cannot qualify them.

The working-zone estimate for a fractional calibration error \(\epsilon\) in b
is a different calculation. Its optical signal-transfer error is
\(\epsilon[(D-1)+\mu\lambda]/D\). It is a conditional bias scenario, not a
prediction that the measured noise increases by the same percentage.

## 3. Evidence for 145609

The initial dark-candidate audit below establishes the exclusions. Section 3.2
adds the active-response association and acquired-readout comparison.

The analysis retains both Tune blocks and both gain models. Only the final block
calibrates the following measured scan. The initial and final blocks contain
628 and 625 model components respectively. Model-local IDs are not APT identities.

The previous complete-singlet bin with \(a>0.75\) has median Qr approximately
27,100. The known dark/high-Q population makes an optical interpretation of that
bin inappropriate. In the final affine fit, its 27 components comprise two with
tentative dark-roster associations, 17 with only the finder dark flag, and eight
without a response classification. Of the 15 complete native estimates, those
counts are one, seven and seven. This is a mixed population, not a controlled
increase of power on the same active detectors.

The exact APT study links UIDs 2 and 3 through original D21 peaks 186332 and 187702.
The acquired sweep arrays and finding measurements were checked against the
reference study before transferring the links. The associations remain tentative;
no frequency-nearest matching or reuse of model-local integers was performed.

**Table 3. Dark-candidate examples in the final affine fit.**

| APT UID | Current group/component | Fitted Qr | Fitted a | Fit at a parameter bound | Complete measured record | Frequency ASD, 5–20 Hz |
| --- | --- | ---: | ---: | --- | --- | ---: |
| 2, tentative | 282 / 316 | 100,000 | 5.49 | Yes | Yes | 5.90 Hz/√Hz |
| 3, tentative | 284 / 318 | 57,889 | 1.44 | Yes | No | Not estimated |

The first row demonstrates why numerical acceptance and low ASD cannot establish
a useful active detector. The second remains a failed measurement, not a quiet
detector. Neither is qualified here for optical sensitivity or dark-reference use.
The low-power study supports real resonances at these locations; its model quality
and power conditions must not be transferred to this overdriven acquisition.

Across all 625 final-block components, 29 have candidate evidence: the two
tentative roster links and 27 additional finder-flagged components. The other
596 lack a response classification in the evidence joined here. The affine
complete-record counts are 1, 14 and 468 respectively; delay counts are 1, 14 and
470. That initial audit supplied no active-detector qualification. The broader
Skydip evidence is attached separately in section 3.2.

### 3.1 Measured noise propagation

For the previous channel-10 example, \(D=1.314\) and \(\mu=1.436\). The
readout-only isotropic prediction for the x_opt/x′ ASD ratio is 1.947. Using the
measured covariance instead predicts 1.780, matching the measured 1.780 in
5–20 Hz. This explains the apparent noise increase without equating it with the
5% calibration-error scenario. Channel 10's optical class is not established by
that agreement.

For the linked dark candidate UID 2, channel 589, the measured loss/effective-
detuning ASD ratio is 0.711 and their band-averaged normalized cross-spectrum is
−0.448 in 5–20 Hz. Its measured x_opt/x′ ASD ratio is 3.939; propagation of the
measured covariance predicts 3.940. This candidate does not satisfy the ideal
isotropic readout-noise limit. Agreement with covariance propagation checks the
coordinate transformation; it does not validate the bounded sweep fit. The
separate two-case noise comparison uses no assumed optical response ratio.

The independent 112-trace Skydip sample has median optical-direction estimate
R_r/R_x=0.0290 and a 16–84% spread of 0.0246–0.0376. That initial illustration did not
attach response evidence to 145609 identities. Using it for channel 10 gives a
*hypothetical* x′ signal transfer of 0.729. It is useful for demonstrating the
calculation, not for assigning channel 10 an optical responsivity. It must not
be assigned to the dark candidates.

### 3.2 Skydip-supported active comparison

The maintained Skydip assessment supplies exact finding receipts and an APT
roster; its 112 isolated response contributors supply independent loss/detuning
laws. The new comparison uses full `Match1D` correspondences against Skydip
scans 2–8. A target requires at least five unique supporting acquisitions, no
competing UID inside the 200 kHz observation-APT matching window, no duplicate
target identity, and support in both Tune blocks. Every candidate edge remains
available. The source key includes product realization, namespace, acquisition,
interface, block and finding row. The target's stored fitting plan must reproduce
its finding ID, initial frequency/loss and finding group exactly.

These criteria yield 240 repeatable APT associations, of which 80 have the
isolated, coherent loss/frequency response used for the active comparison.
They remain provisional observation associations: repeated frequency matching
does not independently prove physical identity. The source roster is unchanged.
The Skydip experiment supports an active-response interpretation but does not
separately control bath motion or establish absolute optical responsivity.

Readout checks use existing convergence, bounds, rank, feature-support and
initialization warnings, contributing-channel quality, complete native solving,
and checked Python/native agreement. The declared sweep-residual limit is 5%
of measured complex sweep variation after centering each probe separately.
This tests calibration adequacy at a stated scale; it is not a noise-floor
acceptance test. Nonlinearity itself is not cut. Five candidates are excluded by
fit warnings (two) or sweep residuals (three), identically for affine and delay
fits. One additional candidate has no selected solver route. That missing
selection is not a demonstrated solving failure.

**Table 4. Final-block population accounting, each model family.**

| Population | Components | Use in the optical-direction comparison |
| --- | ---: | --- |
| Skydip-supported active; readout checks pass | 74 | Individual response law and measured noise |
| Individual response law; readout excluded or pending | 6 | Five quality exclusions and one unselected solver route |
| Loading response; individual loss law unqualified | 145 | Separate descriptive spectra |
| Dark-candidate evidence without loading-response conflict | 14 | Diagnostic spectra; no optical β |
| Dark indicator versus provisional loading-response match | 15 | Unresolved evidence conflict; all fail readout |
| Unresolved identity or response | 371 | Descriptive spectra only |

The qualified 74 are all singlets. Affine median Qr is 10,702 and fitted a spans
0.117–0.733. The quiet high-Qr, strongly nonlinear components that motivated the
earlier discussion do not supply evidence for better active-detector sensitivity.
No multiplet optical-sensitivity conclusion follows from this subset.

The Skydip coordinate is `f_ref/fr - 1`; the target uses `f_probe/fr - 1`.
Therefore the slope used at the target is
\(\lambda_{probe}=\lambda_{Skydip} f_{ref}/f_{probe}\), with the same scale applied
to its conditional error. This coordinate conversion does not establish that
the physical response transfers unchanged across the two acquisitions.

**Table 5. Measured and conditional signal/noise results for the 74 identities.**

| Quantity | Affine | Delay |
| --- | ---: | ---: |
| Median Skydip R_r/R_x, expressed at the probe | 0.03022 | 0.03022 |
| Median measured 145609 slope, 1 s averages | 0.01634 | 0.01790 |
| Median measured 145609 slope, 8 s averages (7 windows) | 0.02315 | 0.02483 |
| Median frequency ASD, 5–20 Hz [Hz/√Hz] | 19.52 | 19.55 |
| Median ASD(x_opt)/ASD(x′) | 1.330 | 1.332 |
| Median ratio after referring x′ noise to the same signal | 1.116 | 1.120 |
| Held-out two-coordinate ASD / x_opt ASD | 0.880 | 0.879 |

The paired delay/affine frequency-ASD ratio is 1.00048. Signal-referring x′ uses
\(T=(1-\mu\lambda)/D\), then divides its ASD by \(|T|\). The two-coordinate
estimate uses unit response to `(1, lambda)`; covariance weights come from the
first half and the reported noise from the second half. These are conditional
signal-transfer comparisons, not absolute NEP, measured source S/N or mapping
speed. The shorter measured trajectory has weaker median slope than the Skydip
law and substantial detector scatter. It contains all noise sources; the
difference does not independently demonstrate a response change.
The slope approaches the Skydip distribution with longer averaging; seven
8-second windows do not establish a new response law. Outliers remain visible.

Tightening the association window to 100 kHz retains 36 readout-qualified
components per family at the 5% residual tolerance. The report exposes the
100/150/200 kHz and 2/5/10% selection alternatives. Selection never optimizes
the measured timestream noise. Individual inspectors show the sweep, IQ cloud,
Skydip response, trajectory, time series and spectra for all 74 identities in
both model families. Fitted arrays and acquired data remain unchanged.

### 3.3 Full network coverage and exclusion accounting

The latest consolidated network-0 roster contains **556 established identities**:
548 reassessed trace baselines and eight retained reviewed identities. The
assessment finds loading motion in 545 traces, stationary behavior in three,
and fitted-width support for 544. These are broader evidence sets than the
112-detector individual loss/detuning study. Neither empirical loading motion
nor established identity alone assigns a physical optical-response class.

The 112-detector study deliberately excluded 338 traces in 133 connected groups
before estimating individual response laws. Another 98 isolated traces failed
that source study's quality/evidence conditions. Therefore its lack of multiplets
is a selection restriction, not a finding that multiplet fitting or solving
failed in 145609.

The current 145609 fits contain 625 final-block model components: 363 singlets
and 262 components in 111 multiplet groups. This is a different denominator
from distinct, established detector identities. Full-network accounting must
not subtract one from the other as a detector-loss estimate.

The conservative cross-acquisition correspondence rule admits 240 UIDs. Updating
its exact receipts to the consolidated roster leaves all 240 supported transfers
unchanged. UID 563 now supports final-block finding 321 but lacks the required
first-block support. The remaining 316 established UIDs retain their identity;
their correspondence to this target acquisition is not established by this rule.

**Table 6. All final-block components, independent of response evidence.**

| Model and multiplicity | Components | Readout screen passes | Complete solving, quality exclusion | Selected, incomplete solving | No selected route |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affine singlets | 363 | 270 | 41 | 4 | 48 |
| Affine multiplets | 262 | 64 | 108 | 54 | 36 |
| Delay singlets | 363 | 271 | 40 | 4 | 48 |
| Delay multiplets | 262 | 61 | 113 | 52 | 36 |

The 84 unselected components have unresolved multiple-group probe choices;
58 have only such choices, while 26 also share probes assigned elsewhere. They
must not be reported as attempted numerical solving failures. Of selected affine
components, 58 lack a complete valid record. Another 149 have complete records
but fail the declared quality screen. Only eight affine components have failed
optimizer convergence; bound warnings, residual exclusions and parity misses
are distinct and can overlap.

**Table 7. Affine response evidence versus readout, exhaustive partition.**

| Evidence | Readout passes | Complete, quality exclusion | Selected, incomplete | Unselected | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Individual source response law | 74 | 5 | 0 | 1 | 80 |
| Loading response; outside individual-law study | 121 | 10 | 0 | 14 | 145 |
| Dark indicator/loading-response conflict | 0 | 10 | 4 | 1 | 15 |
| Target-acquisition identity transfer unresolved | 139 | 124 | 54 | 68 | 385 |

Thus 260 of 334 components passing readout were excluded from the signal-transfer
comparison because of missing or incomplete evidence, not failed readout. Of
the 64 multiplet components passing readout, four have a transferred
loading-responsive identity and 60 lack a qualifying transfer. None has an
individual response law in the isolated source study. Their numerical coverage
must remain visible without assigning an unsupported optical response.

The group-aware analysis below uses established group/observation
evidence to resolve target-acquisition correspondences and retain supported joint
response information. A population slope can be an explicit hypothesis or prior; it is
not an independently measured response for every component. The report keeps
one ledger for all target model components and another for all established UIDs.


### 3.4 Group-aware reconciliation and measured multiplet comparison

The new replay uses the latest consolidated APT receipts and complete Skydip
and target fitting groups. It retains every admissible frequency-ordered
assignment covering the smaller feature list. Only edges forced in every such
assignment count as positive support; all alternatives and unknown identities
remain vetoes. Infeasible complete groups remain unresolved. The same 200 kHz
window and five independent acquisitions are required in both Tune blocks.
This is a conditional no-crossing correspondence test, not roster promotion.

**292** provisional identities now have both-block support. Of the 556 established
roster UIDs, 264 still lack this transfer; they have not lost their identity.
Of 625 target model components, 333 lack this correspondence. These are different
denominators. The numerical fits, stubs and native timestream arrays are unchanged.

Source regression no longer excludes connected groups. It tests every constituent
with seven predeclared acquisitions, exact accepted receipts, dual finding
support, usable native fits and positive labelled frequency/loss covariance.
There are **208** measured source response slopes. The other 340 source traces
have evidence gaps (83), quality exclusions (82), or both (175). These source
exclusions do not establish a failure of the corresponding 145609 readout.
Each available full covariance block remains referenced; each regression uses
its component marginal including frequency/loss covariance. Repeated fitted rows
on different probes are not independent measurements of that component.

**Table 8. New affine evidence/readout accounting for all target components.**

| Evidence | Readout passes | Complete, quality exclusion | Selected, incomplete | Unselected | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Component source response law | 103 | 15 | 2 | 13 | 133 |
| Loading response without qualified component law | 110 | 19 | 3 | 11 | 143 |
| Dark indicator/loading-response conflict | 0 | 10 | 4 | 2 | 16 |
| Target correspondence unresolved | 121 | 105 | 49 | 58 | 333 |

The comparison contains **99 singlets and four multiplet components** in each
family. The multiplet UIDs are 31 and 546 (group 229), 228 (group 319), and 475
(group 94). All use the full nonlinear joint native solution. Fourteen multiplet
components have transferred response laws: four pass readout, seven have complete
solving but quality exclusions, two have incomplete solving, and one has no
selected route in the affine model. Delay changes the latter quality/incomplete
counts to eight/one. The other 248 multiplet components have missing or conflicting
evidence, independently of their readout state. This is not a four-detector yield.

**Table 9. Measured comparison; medians over the qualified components.**

| Family / components | Number | Frequency ASD, Hz/√Hz | Skydip R_r/R_x at probe | 1-s trajectory slope | 8-s trajectory slope | Held-out projected/x ASD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Affine singlets | 99 | 19.54 | 0.02975 | 0.01645 | 0.02411 | 0.8760 |
| Delay singlets | 99 | 19.59 | 0.02975 | 0.01810 | 0.02591 | 0.8758 |
| Affine multiplets | 4 | 18.72 | 0.02404 | 0.03239 | 0.03792 | 0.7998 |
| Delay multiplets | 4 | 18.75 | 0.02404 | 0.03335 | 0.03789 | 0.7990 |

The PSD band is 5–20 Hz. The projection preserves a component's independently
assumed Skydip direction, trains on half the timestream, and measures the other
half. It does not mix other detector coordinates. The supplied response is a
hypothesis at the target loading, not a measurement made by this projection.
Leave-one-acquisition source slopes expose transfer sensitivity; source reduced
chi-square exposes departure from a straight response law under its conditional
covariance. Neither is hidden by the cohort label.

Four components cannot determine population-level multiplet performance. The
report retains cross-component coherence and makes no independent-detector
mapping-speed sum. Native branch continuity and full-model validity remain
conditions of this comparison. The component explorer shows the independent
Skydip points, solved joint trajectories, time series and PSD together. Statistical
errors conditional on a fitted model do not account for incorrect identities,
unmodeled coupled dynamics or a changed loading response.

## 4. Report science and diagnostic utility separately

For qualified active detectors, evaluate an optical source template and the
measured noise after the intended processing. Retain correlated noise and joint
component covariance. Report sensitivity within response/readout groups before
subdividing by multiplicity and nonlinearity. Mapping-speed contributions require
the same source, beam, exposure and processing; raw Hz/√Hz is not that measure.

A qualified dark detector can help estimate disturbances in active channels.
In a linear noise-prediction model, dark references D can reduce the residual
active-channel covariance to
\(C_{AA|D}=C_{AA}-C_{AD}C_{DD}^{-1}C_{DA}\). This is an indirect benefit, not an
optical responsivity for the dark detectors. Estimate and validate that prediction
on separate data; measure optical leakage and resulting source transfer. Badly
calibrated or unstable dark channels can inject error into the prediction, so the
existing overdriven candidates remain diagnostic-only until they pass those checks.

The next scientific extension is to verify the provisional observation
associations and response transfer, then include supported multiplets and an
optical source template. The present population comparison does not authorize
science-mask installation or an optimum drive-power setting.

## 5. Evidence and reproduction

- KIDs/KIDsproc chat: `01a0ea26-6070-7930-83ee-13bd169c0c5f`, response/identity
  vocabulary and implementation status reviewed on 2026-11-21.
- Main-workspace manuscript: `design/manuscripts/detector-identity.md`.
- Main-workspace studies: `run/kids-skydip-populations-20261109`,
  `run/kids-dark-lowpower-20261108`, `run/kids-dark-workflow-20261109`, and
  `run/apt-evolution-comparison-20261109`. The first three establish empirical
  candidate evidence; the fourth provides exact observation associations.
- Local report: `run/network-145609/response-qualification/`. Its CSV retains
  every component, separate assessment fields, original flags and exact index
  locations. Its evidence JSON records source hashes and equality checks.
- Reproduce the ledger with `examples/tune_model_comparison/network_populations.py`,
  then render `network_report.py`. These operations do not refit models, alter
  solver arrays, assign new APT identities or install a science mask.
- Active comparison: `examples/tune_model_comparison/network_active.py` and
  `run/network-145609/active-response/`. Source checksums include the current
  `nw0-network-response-20261121`, `nw0-baseline-replay-20261121` and
  `nw0-companion-promotion-20261120` evidence. The source APT retains its unknown
  physical response classes; this report creates no new roster observations.
- Full network census: `examples/tune_model_comparison/network_census.py`, with
  outputs in `run/network-145609/network-census/`. It rechecks saved correspondence
  edges against `nw0-workflow-consolidation-20261121` and records each missing
  transfer, source-study exclusion, pending selection and readout outcome.
