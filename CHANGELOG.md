# Changelog

## Unreleased

- Rewrite training around explicit collection, replay sampling, learner updates,
  model publication, and checkpoint boundaries.
- Share PUCT and Gumbel inference, value conventions, and replay targets.
- Support general discrete Gymnasium and deterministic alternating OpenSpiel
  environments, with explicit encodings, masks, rewards, and truncation handling.
- Add phased and asynchronous game/position scheduling with recoverable counters.
- Support one-node multi-GPU learning and remote self-play with deduplicated retries.
- Add TensorBoard and immutable live/offline evaluation with restart recovery.
- Separate experiment configurations from hardware configurations.
- Consolidate the implementation under `zero_systems` and the `zero-systems` CLI.
- Preserve loading of full checkpoints written before the package consolidation.

See `ARCHITECTURE.md` and `VALIDATION.md` for contracts and measured limitations.
