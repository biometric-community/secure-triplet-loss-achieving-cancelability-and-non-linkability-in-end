# FIDELITY_AUDIT — Secure Triplet Loss

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Keyed embedder + FiLM | 94% | ResNet-18 + bipolar keys |
| 2 | Secure triplet (cancel/impostor) | 93% | d_cancel ≫ d_pos |
| 3 | Non-linkability term | 90% | gap-to-impostor proxy |
| 4 | YTF / ECG / KL unlink | 72% | D1/D2/D4 |
| 5 | Size gate + REPORT + publish | 95% | Stop |

Confidence vs paper structure: **~89%**.

## Measured run (full)

- CelebA+LFW+CALFW+CPLFW ≈ **4.09 GiB** → `full_train`; 8 epochs (**352 s**)
- LFW same-key **74.15%**; cancel mean cosine **0.388**; link |gap| **0.418**
