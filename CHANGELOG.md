# Changelog

All notable changes to this project are documented here.

## [0.4.0] - 2026-07-14
### Added
- `uncollapsed.faultbench`: the **two lieutenants benchmark** — the two zeros
  as distributed-systems fault triage (crash vs Byzantine; Lamport's 2f+1 vs
  3f+1 price gap) on real telemetry: the Intel Berkeley Lab 54-mote sensor
  deployment, fetched once from a public GitHub mirror and cached. Real
  signal statistics, canonical injected fault models (standard BFT/sensor-
  fusion methodology): drift = labelled clear-task fault; Byzantine =
  12-hour self-replay, labelled both ways (the traitor's provenance is
  observer-undecidable — that *is* the two lieutenants problem); crash =
  dropped reports, never in training. Peer-relative features with per-mote
  baselines from training days only; temporal split. Result, seeds 0–2:
  triage AUC 1.000 vs entropy 0.00–0.06 (inverted); crash→gather 1.00;
  Byzantine→escalate 0.88–0.94.
- `uncollapsed --demo faults`, `tests/test_faultbench.py` (4 tests,
  auto-skip when the corpus is neither cached nor downloadable).
- Full documentation overhaul (`mkdocs build --strict` verified): rewritten
  Home/Theory/Usage covering the head's objective, the three background
  modes and their failure cases, a tuning guide, and a documented **pole
  attractor** failure mode (over-large `fit_var_weight` on data lacking one
  of the clear poles collapses contested points to a fake lean — signature,
  cause, and remedies included); new Benchmarks page with all protocols,
  every result, reproduction commands, and a "what would falsify this"
  section; Benchmarks added to site nav (its absence failed strict builds).

## [0.3.0] - 2026-07-13
### Added
- `uncollapsed.realbench`: the two zeros protocol on **real data** — sklearn's
  bundled `digits` (no download) and Fashion-MNIST (fetched once from the
  official Zalando GitHub repo, cached). Conflict is *contradictory
  duplication*: every conflict-class sample appears with both labels, so the
  contradiction is in the data and cannot be memorized away. Void is
  held-out-class near-OOD, deliberately the hard kind. Digits: triage AUC
  0.91 vs 0.23 (entropy baseline). Fashion maps the honest boundary of
  input-density methods, with a per-class routing breakdown showing the
  failure is structured (shirt-like classes escalate; boot-like classes read
  as boots).
- `FieldHead` gains `bg_mode="shell"` (background = training points + noise:
  the void tax lands on the neighborhood of the manifold), `bg_mode="shuffle"`
  (marginal shuffle, for raw correlated features only — documented as wrong
  after PCA/whitening), and `fit_var_weight` (explicit evidence-accumulation
  incentive so conflict zones can outbid the void tax; vanishes in clear
  regions). Gradients still hand-derived and verified.
- `uncollapsed --demo real [--dataset digits|fashion]`, `[bench]` extra
  (scikit-learn), `tests/test_realbench.py` (6 tests; fashion auto-skips
  offline).

## [0.2.0] - 2026-07-13
### Added
- `uncollapsed.head.FieldHead`: the drop-in **field-gated readout head** from
  the roadmap. Feed it any feature vector; it produces per-sample four-mass
  accounting and an explicit route: `presence` / `absence` / `hold` /
  `escalate` (loaded contradiction -- a human should look) / `gather` (no
  evidence exists -- collect data). Evidential fit term + annealed
  misleading-evidence tax + background-void tax ("presence must be earned;
  void is the ground state"). Hand-derived gradients verified by
  `grad_check_head` (~1e-7).
- `uncollapsed.bench`: the **two zeros benchmark** -- clear clusters, a
  50/50-label conflict cluster, and an off-manifold void ring, scored against
  a capacity-matched vanilla MLP whose only signal is predictive entropy.
  Headline metric: triage AUC (separating contradiction from ignorance among
  ambiguous points). `uncollapsed --demo triage` runs it.
- `uncollapsed.data.make_two_zeros`, `uncollapsed.viz.render_two_zeros`
  (five-panel mass-map + routing figure), `tests/test_head.py` (10 tests
  asserting the claims, not just the plumbing).
- `.github/workflows/publish.yml`: PyPI trusted publishing on GitHub release.

### Changed
- Version exported as `0.2.0`; `FieldHead` exported from the package root.

## [0.1.0] - 2026-07-08
### Added
- `uncollapsed.field`: the `Field` state (independent presence/absence channels)
  and the four-mass projection `Mass` (belief, disbelief, conflict, voidness).
- Edge-collapse policy with first-class `HOLD`, `ESCALATE`, `ABSTAIN`, `EMPTY`
  outcomes. A genuinely balanced state never defaults to absence.
- `uncollapsed.algebra`: `NOT` (channel swap), `AND`, `OR`, conservative and
  consensus fusion over fields.
- `uncollapsed.net`: a trainable two-channel field network with signed-weight
  "swap" synapses, hand-derived backprop, and a numerical gradient check.
- `uncollapsed.data`, `uncollapsed.viz`, and a `uncollapsed` CLI.

### Fixed (relative to the pre-release prototype)
- `expectation` now projects BOTH conflict and voidness to the base rate, so a
  fully conflicted state reads ~0.5 instead of 0.0 (no negative-zero leak).
- `voidness` is high only when BOTH channels are weak, so a confident-but-quiet
  state is no longer mislabelled as mostly void.
