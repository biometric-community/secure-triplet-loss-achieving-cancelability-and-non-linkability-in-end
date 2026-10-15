# REPORT — Secure Triplet Loss

**Protocol:** `full`

Lab twin of Pinto et al. (TBIOM): keyed face embedder trained with Secure Triplet Loss
(cancelability + non-linkability) on CelebA; evaluated on LFW/CALFW/CPLFW.

## Training

- Epochs: 8
- Base images: 141610
- Identities: 7124
- Elapsed (s): 352.2

![Training](outputs/figures/full/train_loss.svg)

## Same-key verification

| Dataset | Accuracy | EER | N |
|---------|----------|-----|---|
| calfw | 51.83% | 48.17% | 6000 |
| cplfw | 54.72% | 45.28% | 6000 |
| lfw | 74.15% | 25.85% | 6000 |

![Verification](outputs/figures/full/verification.svg)

## Security proxies

- Cancelability mean cosine (same image, different keys): **0.3876**
- Fraction cosine < 0.3: **21.7%**
- Linkability |mated-diffkey − impostor| mean gap: **0.4184**

![Security](outputs/figures/full/security.svg)

## Notes

- Paper uses YouTube Faces / UofTDB ECG; this rebuild uses CelebA + LFW family.
- See `DEVIATIONS.md`.
