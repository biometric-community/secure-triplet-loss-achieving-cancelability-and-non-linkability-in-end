# IMPLEMENTATION_PLAN — Secure Triplet Loss

## Paper

Keyed biometric templates; Secure Triplet pulls (ID, key)-mated pairs and pushes cancelability + impostor negatives; non-linkability regularizer.

## Lab twin

| Component | Choice |
|-----------|--------|
| Backbone | ResNet-18 + key MLP fusion |
| Keys | Random bipolar {-1,+1}^64 |
| Loss | Secure triplet (cancel + impostor + γ·unlink) |
| Data | CelebA train; LFW/CALFW/CPLFW eval |
