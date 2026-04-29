# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project follows semantic versioning once a `1.0.0` release is tagged.

## [Unreleased]

### Tian (1999) implementation rewrite — 2026-04-29

The previous `tian_parameters` function did not implement the scheme of
Tian (1999), *A Flexible Binomial Option Pricing Model*, JFM 19(7),
817–843, despite being labeled as such. It implemented a custom strict
strike-aligned construction that solved a three-equation system
(strike alignment, risk-neutral pricing, variance matching) for
`(u, d, p)` simultaneously, with a fallback to a closed-form
parameterization on solver failure. The construction is not in
Tian (1999), and a code-level bug in the choice of `a*` (using
`round(N * p_CRR)` rather than the integer-nearest-strike index from
log-space arithmetic) produced parallel-band convergence artifacts
visible in the legacy figures.

This release replaces that implementation with a faithful Tian (1999)
construction and renames the previous closed-form fallback to its
correct attribution, Tian (1993).

#### Added

- `flexible_binomial_parameters(T, N, r, sigma, lam, q)` in
  `src/option_pricing/parameterizations.py`. Implements Tian (1999)
  eq. (6): `u = exp(σ√Δt + λσ²Δt)`, `d = exp(−σ√Δt + λσ²Δt)`. Takes
  the tilt parameter `λ` as input. `λ = 0` recovers CRR.
- `tian_1999_parameters(S, K, T, N, r, sigma, q)` in
  `src/option_pricing/parameterizations.py`. Computes the
  strike-aligning tilt via Tian (1999) eqs. (11) and (13), then
  dispatches to `flexible_binomial_parameters`. Places the terminal
  node `(N, j₀)` exactly at the strike `K`, verified at machine
  precision.
- `richardson_extrapolation(price_N, price_2N, rho)` in
  `src/option_pricing/pricers.py`. Implements Tian (1999) eq. (17):
  `Ĉ(2N) = (ρC(2N) − C(N)) / (ρ − 1)`. Default `ρ = 2.0`, appropriate
  for O(1/N) schemes including the strike-aligned Tian (1999) tree.
- New scheme names `"tian_1993"` and `"tian_1999"` in the calibration
  dispatch (`src/option_pricing/calibration.py`).
- Test class `TestTian1999PaperReproduction` in
  `tests/test_parameterizations.py`. Hard-codes 28 numerical values
  from Tian (1999) Tables I and II as ground truth; all pass to
  4–6 decimal places.
- Six parametrized strike-alignment tests in `TestTian1999Parameters`
  verifying the terminal node coincides with `K` at machine precision
  across `(K, N)` combinations.
- Test class `TestRichardsonExtrapolation` in `tests/test_pricers.py`,
  including reproduction of the worked extrapolation example from
  Tian (1999) p. 829.
- `scripts/generate_tian_smoothness.py` — produces
  `figures/tian_smoothness.pdf`, a signed-error comparison of CRR
  oscillation against the monotone Tian (1999) curve. Reproduces the
  qualitative content of Tian (1999) Figure 2.
- `scripts/generate_tian_extrapolation.py` — produces
  `figures/tian_extrapolation.pdf`, an absolute-error comparison of
  CRR, Tian (1999), and Richardson-extrapolated Tian (1999).
  Reproduces the qualitative content of Tian (1999) Figure 3.

#### Changed

- `tian_closed_form_parameters` renamed to `tian_1993_parameters`.
  Function signature changed: dropped the unused `S` and `K`
  arguments, since the Tian (1993) third-moment-matching construction
  does not depend on spot or strike. The closed-form expression itself
  is unchanged and verified against Kim, Stoyanov, Rachev, and
  Fabozzi (2016, arXiv:1612.01979) and the original paper, Tian
  (1993), JFM 13, 563–577.
- `SchemeName` literal in `src/option_pricing/calibration.py` changed
  from `"crr" | "tian" | "lr"` to
  `"crr" | "tian_1993" | "tian_1999" | "lr"`. The previous `"tian"`
  scheme name is removed; users must specify which Tian scheme.

#### Removed

- `tian_parameters` (the previous strict strike-aligned solver).
  Replaced by `tian_1999_parameters`, which is faithful to the paper.
- Five legacy figure scripts that were built around the previous
  (incorrect) framing of Tian as a per-N accuracy improvement over
  CRR:
  - `scripts/generate_tian_alignment_wins.py`
  - `scripts/generate_tian_alignment_loses.py`
  - `scripts/generate_tian_regime_analysis.py`
  - `scripts/generate_tian_envelope_crossings.py`
  - `scripts/generate_tian_rolling_winrate.py`
- The corresponding PDFs in `figures/`.

#### Test summary

127 tests pass. This includes 28 paper-reproduction tests against
Tian (1999) Tables I and II, 6 strike-alignment tests at machine
precision, and 4 Richardson extrapolation tests including the worked
example from the paper.

### Added — 2026-04-29 (continued)

- `scripts/generate_american_convergence.py` and the resulting
  `figures/american_convergence.pdf`. Convergence comparison for
  American put pricing across CRR, Tian (1999), and Leisen--Reimer.
  Reference price computed by LR at $N = 25{,}001$, verified stable
  via Richardson extrapolation. All three schemes exhibit
  $O(1/N)$ rate for American options at this regime; LR retains a
  constant-factor advantage but loses its $O(1/N^2)$ European rate
  advantage. Used in Section 7.6 of the LaTeX note.