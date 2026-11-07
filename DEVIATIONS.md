# DEVIATIONS — Secure Triplet Loss

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Face data | YouTube Faces | CelebA + LFW/CALFW/CPLFW | Local public data |
| D2 | ECG modality | UofTDB | Not included | Face-only lab twin |
| D3 | Architecture | Paper face CNN | ResNet-18 + key MLP | Lab twin stack |
| D5 | Keys | User tokens | Random bipolar vectors | Training surrogate |
