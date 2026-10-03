# Reproduce the maintained CPU validation

From the repository root, create a new isolated environment (do not activate or modify a historical environment):

```sh
uv venv --python 3.12.15 .venv
uv pip install --python .venv/bin/python -r requirements-main.txt
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/main_runtime -v
```

The local C extension needs a C compiler. No historical binary is linked. An alternative isolated micromamba environment is described by iev_vae_env.yml; install the editable local package afterward.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python validation/benchmark.py . /tmp/iev2mol-current.npz --all-examples
.venv/bin/python validation/compare.py validation/reference.npz /tmp/iev2mol-current.npz
```

The checked-in compact reference uses explicit indexed samples for large gradients/weights; the reported gate was measured on full arrays. The fixture includes all 10 original test molecules. Add `--stochastic` to repeat the actual-Dropout comparison against a freshly generated original reference with the same flag. Obtain unchanged main in a separate directory and its author DRD2 state, install old-pins.txt using Python 3.10 in a separate reference environment, then use the same benchmark script with that directory as its first argument. Original main modules use W&B; the validation benchmark substitutes a no-op logging observer only, without modifying their math. Add `--rebuild-vocab` to verify the original first-appearance vocabulary from the author million-SMILES zip. Compare both NPZ arrays and JSON vocabulary/generated strings.

Maintained training/generation (numeric fixtures are only a smoke dataset):

```sh
.venv/bin/python MAIN/model/runtime_cli.py --checkpoint MAIN/model/iev2mol_DRD2.pt --vocab validation/fixtures/vocab.json --smiles validation/fixtures/drd2-smiles.json --vectors validation/fixtures/drd2-data.npz --epochs 1 --seed 19 --output /tmp/iev2mol-epoch1
.venv/bin/python MAIN/model/runtime_cli.py --checkpoint /tmp/iev2mol-epoch1/checkpoint.pt --vocab validation/fixtures/vocab.json --smiles validation/fixtures/drd2-smiles.json --vectors validation/fixtures/drd2-data.npz --epochs 1 --output /tmp/iev2mol-epoch2
```

Default maximum generation length is 201. `--epochs 0` generates without training. `--no-old-compatible` selects standard modern math; use the same mode when resuming. The compatibility default supports CPU float32 only. Supply prepared JSON SMILES and `vectors` numeric NPZ from a separate docking/preprocessing environment for actual work. Preserve the author vocabulary order when changing data; do not rebuild it by sorting characters.

`convert_data.py` is a fixed-hash, one-time trusted author-dataset converter, not part of the maintained loading path. Its trusted class reference path must be configured before running it. Do not run it on unknown checkpoints or datasets. Historical train scripts and other model families are outside this validated CLI path.
