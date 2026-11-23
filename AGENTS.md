# AGENTS.md

## Purpose

Build an understandable, reproducible, high-performance MuZero implementation
for public research use and board-size transfer experiments.

## Working rules

- Read this file and the affected code before changing it.
- Trace the complete affected path before editing.
- Prefer the smallest correct implementation.
- Keep the main training workflow visible.
- Reuse Python, NumPy, Numba, and PyTorch facilities before adding dependencies.
- Do not introduce Ray, databases, native extensions, or distributed training
  without a profile demonstrating the need.
- Keep algorithm settings separate from hardware settings.
- Preserve validation, failure handling, accessibility, and reproducibility.
- Add one small runnable check for non-trivial logic.
- Treat reference repositories as read-only unless their licenses permit reuse.
- Use Gymnasium and OpenSpiel through thin adapters
- Gate optional features on measured system cost and learning benefit.

## Research standard

Every meaningful change should state:

1. What changed.
2. Why it may matter.
3. How it was checked.
4. What limitation remains.

Performance claims require fixed algorithm settings, warm-up, repeated
measurements, and recorded hardware/software versions.

## Context handling

Use context-mode only for bulky generated artifacts over 100 lines or 20 KB,
such as logs, test output, traces, datasets, or broad command output. Read
source files normally.

## Git

Use git for inspection unless the user asks for a git operation. Preserve
unrelated user changes.
