# Benchmark report — fake

- model: `fake/model` (sha `None`)
- arm B implementation: `systemone`
- reps: 3, test items: 360
- server_id: `fake-server`

## Quality and latency (rep 1)

| arm | micro-F1 | macro-F1 | exact match | fmt err | err | label mass | p50/p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1.0000 | 0.0833 | 1.0000 | 0.0% | 0.0% | — | 0.2 / 0.2 |
| B | 0.9524 | 0.0833 | 0.9000 | 0.0% | 0.0% | 0.9 | 0.2 / 0.2 |

## Paired bootstrap (1,000 resamples by group, 95% CI)

| metric | value | CI 95% |
| --- | --- | --- |
| Δ micro-F1 (B − A) | -0.0476 | [-0.1111, +0.0000] |
| Δ macro-F1 (B − A) | +0.0000 | [+0.0000, +0.0000] |
| Δ exact match (B − A) | -0.1000 | [-0.2500, +0.0000] |
| p50 ratio (A / B) | 0.731 | [0.623, 0.779] |

## Stability across reps

| arm | same set | max P(yes) swing |
| --- | --- | --- |
| A | 100.0% | — |
| B | 100.0% | 0.0 |

## Per-collection micro-F1 (rep 1)

| collection | A | B |
| --- | --- | --- |
| cognitive_psychology | 1.0 | 0.9524 |
