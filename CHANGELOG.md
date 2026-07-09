# Changelog

All notable changes to this project are documented here.

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
