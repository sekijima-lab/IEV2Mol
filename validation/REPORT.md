# Maintained main CVAE runtime

The historical paper branch `main` is retained. This maintained runtime belongs on `security/modern-runtime` and must not be merged into main. Author checkpoint tensors and model architecture are unchanged.

## Security update and scope

Python 3.7.0 → 3.12.15; PyTorch 1.13.1 → 2.14.1; NumPy 1.21.5 → 2.5.3; scikit-learn 1.0.2 → 1.9.1; RDKit 2020.09.1.0 → 2026.3.6. All 30 external Python distributions are pinned in requirements-main-lock.txt. A fresh isolated installation (31 distributions including this local extension) passes dependency checks. W&B observation is replaced by local logging. The maintained CLI requires restricted tensor checkpoint loading and JSON plus numeric, non-object NPZ inputs. It never falls back to unrestricted pickle loading.

The dated OSV snapshot records 17 matched original-main packages (38 recognized export entries), 3 reference packages (21 entries), and zero maintained main packages (30 entries). This is a PyPI-name/version advisory match, not proof that every historical Conda build is affected or the whole repository is secure. Python itself, native libraries, OS, application code and other environments are outside this query.

## Frozen numerical validation

Reference code is unchanged main commit 36ae3ad4a9a9996a89e65d9822faf14a392bedca. The reference CPU environment uses Python 3.10.19, PyTorch 1.13.1 and RDKit 2022.9.5; it is reconstructed, not the exact original Python 3.7/Linux/CUDA/RDKit 2020.09 environment. Its complete pins and original export are archived here. No full paper reproduction is claimed.

Criteria were frozen before comparison: logits/latent absolute differences ≤1e-5, losses/gradients/one Adam step weights ≤1e-4, vocabulary and seeded generation exact. Three seeds (0,19,73), CPU float32, one thread, the unchanged DRD2 author checkpoint, all 10 supplied DRD2 test molecules and their 189-dimensional interaction vectors were tested.

With actual Dropout enabled, latent variables, interaction conditions, reconstructions, decoder inputs and all decoder logits are exact. Maximum loss difference is 1.49e-8; full gradient difference 9.31e-9; full one-Adam-step weight difference 7.78e-6. All criteria pass. Dropout-disabled diagnostic comparison also passes (weight difference 1.15e-5). Eighteen full conditional SMILES generations (three real conditions × three seeds × two samples, maximum 201 characters) and the 54 ordered vocabulary IDs agree exactly. Fresh-install comparison repeats these results.

Native modern arithmetic initially exceeded the frozen logits limit (5.03e-5). The default compatibility implementation restores scalar expf/tanhf, GRU recurrence order, BatchNorm affine order and SELU `(exp(x)-1)` order. This follows the official [PyTorch 1.13.1 CPU RNN source](https://github.com/pytorch/pytorch/blob/v1.13.1/aten/src/ATen/native/RNN.cpp) and [activation source](https://github.com/pytorch/pytorch/blob/v1.13.1/aten/src/ATen/native/cpu/Activation.cpp). No old PyTorch binary or global monkeypatch is used. Native modern arithmetic remains selectable with `--no-old-compatible`; it does not inherit the compatibility equivalence claim.

Seven tests cover scalar operator oracles, bidirectional packed/dense GRU, 2-D/3-D BatchNorm, state key compatibility, input rejection, safe serialization and prevention of executable pickle loading. Actual fine-tuning CLI smoke tests pass in both modes: two epochs/four Adam steps, restricted checkpoint reload and exact model/optimizer/RNG agreement between uninterrupted and resumed training. Generation is tested to 201 characters in the frozen comparison; smoke training generation uses 32.

## Limits and remaining work

Compatibility is validated for this macOS CPU float32 build, first-order gradients and this DRD2 checkpoint/test set. All three CVAE author checkpoints accept restricted loading, but AA2AR/AKT1 numerical equivalence, Linux/CUDA, full-million-molecule retraining, longer learning trajectories and higher-order derivatives are unverified. The historical training scripts retain their original paths and input format; use runtime_cli.py for the maintained path.

JT-VAE, IFP-RNN, Python 2.7 Vina and proprietary docking workflows are not migrated or verified by this change. They retain historical exports and must use separate environments. There is no claim that IEV2Mol as a whole or the organization maintenance task is complete.

The trusted one-time custom dataset conversion verifies the exact author file SHA256 before unrestricted loading in the reference environment. Do not use it on arbitrary files. Runtime inputs avoid custom object pickles. See REPRODUCE.md and provenance.json.
