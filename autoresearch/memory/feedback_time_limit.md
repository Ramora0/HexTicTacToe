---
name: evaluation time limit
description: Use 500ms (0.5s) per move for evaluations, not the default 1s
type: feedback
---

Run evaluations with time_limit=0.5, not time_limit=1.0.

**Why:** User preference — 1s per move makes evaluations too slow.
**How to apply:** In all evaluate() calls, use `time_limit=0.5` for both bots.
