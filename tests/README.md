# GitPython security update validation

GitPython **3.1.41 → 3.1.62** addresses reviewed security advisories, including
[CVE-2026-87817 / GHSA-239g-whfq-7xj9](https://github.com/advisories/GHSA-239g-whfq-7xj9).
Tracked files must not be mistaken for the real `.git` directory/configuration.
This is defensive dependency maintenance, not evidence that IEV2Mol has been exploited.

The official PyPI wheel supports Python >=3.7. The dependency is already in
`iev_vae_env.yml`'s pip section; only its pin changes. The existing gitdb,
smmap and typing_extensions pins are compatible and remain unchanged.

Use a **new** Python 3.7 virtual environment:

```sh
python -m pip install -r tests/requirements-gitpython.txt
python tests/test_gitpython_compatibility.py
python -m pip check
```

Validated with **Python 3.7.12 / macOS x86_64 under Rosetta** on 2026-10-02,
using an isolated micromamba environment and two new virtual environments.
The three other listed dependencies are identical before/after. Git metadata,
branch, remote, dirty/untracked state and diffs match the real Git CLI.
Local clone, bare repository and linked worktree checks pass with both
3.1.41 and 3.1.62. Inert tracked `HEAD`, `objects`, `refs`, `gitdir`,
`commondir` and `config` reproduce the wrong directory on 3.1.41 and confirm
correct discovery/config isolation on 3.1.62. No executable hook or external
configuration include is created. All three updated tests and `pip check`
pass; OSV returned no advisories for 3.1.62 on 2026-10-02.

To compare the baseline, replace only the GitPython pin with 3.1.41 in a
second environment. `GITPYTHON_BASELINE=1` skips the expected failing security
test; run without that flag to reproduce the old discovery problem.

## Limits

The original Python **3.7.0** patch level and complete Linux/CUDA environment
were not run. PyTorch/TensorFlow models, predictions, generation, docking and
other IFP-RNN/vina environments are unverified. These are scoped Git API checks,
not a full application validation. Python 3.7 is unsupported and its migration,
along with remaining dependency advisories, requires separate work.
