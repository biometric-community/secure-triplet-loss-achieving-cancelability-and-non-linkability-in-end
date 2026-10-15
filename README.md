# Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of **Secure Triplet Loss**: key-conditioned face templates with cancelability and non-linkability regularizers.

**Repository:** https://github.com/biometric-community/secure-triplet-loss-achieving-cancelability-and-non-linkability-in-end

**Paper:** [10.1109/TBIOM.2020.3046620](https://doi.org/10.1109/TBIOM.2020.3046620)

## Setup

```bash
bash scripts/setup_env.sh
```

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
