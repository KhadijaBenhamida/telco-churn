# TP05 — GitHub Actions, dependency audit and secret scanning

Repository: <https://github.com/KhadijaBenhamida/telco-churn> (public, default branch `main`)
Project: `TP1-2 MLOPS/telco-churn` · Python 3.12 · uv 0.13.9 · Ruff 0.17.5 · pytest 9.1.1
pip-audit 2.10.1 · Gitleaks 8.30.1

## 0. Status at a glance

| Check | Where | Result |
|---|---|---|
| Ruff, format and TP04 tests | **GitHub Actions** (`quality`) | **pass** — 26 s ([run](https://github.com/KhadijaBenhamida/telco-churn/actions/runs/44561877608)) |
| Dependency audit (`pip-audit`) | **GitHub Actions** (`dependency-audit`) | **pass** — 14 s, no advisories (same run) |
| Secret scan (Gitleaks 8.30.1) | **GitHub Actions** (`secret-scan`) | **pass** — 7 s, no leaks (same run) |
| Red check → repair | PR #6 (two runs) | **failed then green** (§3) |
| Dependabot | `.github/dependabot.yml` on `main` | **4 PRs opened**, #1 declined with a reason, #2–#4 awaiting checks |
| Local reproduction | `uv sync --locked` + the three checks | **pass** (`reports/tp05_evidence/01_local_checks.txt`) |

The workflow was first blocked: the `gh` token had scopes `gist, read:org, repo` and GitHub
refuses to create `.github/workflows/` without the `workflow` scope (`git push` rejected,
REST API 404). After `gh auth refresh -s workflow` the file was pushed and the checks ran.

Two early CI attempts failed for real reasons and were fixed in the workflow, not bypassed:

| Run | Symptom | Fix |
|---|---|---|
| [44560954002](https://github.com/KhadijaBenhamida/telco-churn/actions/runs/44560954002) | `Unable to resolve action astral-sh/setup-uv@68559e9…` | the SHA is the tag object, not the commit → pinned `astral-sh/setup-uv@v10.3.3` |
| [44561713750](https://github.com/KhadijaBenhamida/telco-churn/actions/runs/44561713750) | `gitleaks: command not found` (exit 127) | `GITHUB_PATH` only applies to *later* steps → `sudo install` the binary into `/usr/local/bin` |


## 1. Local reproduction of the CI checks

`reports/tp05_evidence/01_local_checks.txt`:

```text
$ uv sync --locked
Resolved 111 packages in 3ms / Checked 110 packages in 20ms        exit=0
$ uv run --locked ruff check src/telco_churn tests scripts
All checks passed!                                                 exit=0
$ uv run --locked ruff format --check src/telco_churn tests scripts
18 files already formatted                                         exit=0
$ uv run --locked pytest -q
35 passed in 10.76s                                                exit=0
```

The suite needs neither the Telco CSV (the eight CSV-dependent tests skip) nor an MLflow
server, which is exactly the runner's situation. `--locked` also proves that `uv.lock`
matches `pyproject.toml`: CI fails on a stale lockfile instead of silently resolving new
versions.

**Why a teammate should not rely on the pre-commit hook alone.** Hooks are local: they exist
only after `pre-commit install`, they only see staged files, and they can be skipped with
`--no-verify` or bypassed by an IDE/bot that commits directly. CI runs on a fresh machine
from the committed state for every push and pull request, so it is the shared, reproducible,
auditable record — and the only check a branch-protection rule can require.

## 2. The workflow (`.github/workflows/ci.yml`)

Triggers: `pull_request` targeting `main` and `push` to `main`. `permissions: contents: read`
(no write access, no secrets), `timeout-minutes: 10` per job, three named jobs so a reviewer
sees which check failed. `pull_request` — not `pull_request_target` — is used so contributor
code runs without gaining repository rights.

| Job | Steps | Why |
|---|---|---|
| `quality` | checkout → setup-uv → `uv sync --locked` → `ruff check` → `ruff format --check` → `pytest -q --cov` | reproduces TP04 exactly |
| `dependency-audit` | checkout → setup-uv → `uv sync --locked` → `uv run --locked pip-audit` | audits the installed environment, dev tools included |
| `secret-scan` | checkout → install Gitleaks 8.30.1 (SHA-256 verified) → `git archive` the tracked files → `gitleaks dir --redact --exit-code 1` | scans exactly the committed content |

Actions are pinned to reviewed releases: `actions/checkout` by commit SHA
(`3d3c42e5aac5ba805825da76410c181273ba90b1`, release v7.0.1) and `astral-sh/setup-uv` by its
release tag `v10.3.3` — pinning that tag by SHA does not work, the value `68559e9…` is the tag
object and the action resolver answers *"unable to find version"*, which is what the first run
showed. The Gitleaks release is downloaded during the job and its published SHA-256 is
verified before the binary is executed.


```yaml
name: Telco CI
on:
  pull_request: {branches: [main]}
  push: {branches: [main]}
permissions: {contents: read}
env:
  PYTHON_VERSION: "3.12"
  UV_VERSION: "0.13.9"
  LINT_TARGETS: "src/telco_churn tests scripts"

jobs:
  quality:
    name: Ruff and TP04 tests
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1   # v7.0.1
        with: {persist-credentials: false}
      - uses: astral-sh/setup-uv@v10.3.3                                  # release tag
        with:
          version: ${{ env.UV_VERSION }}
          python-version: ${{ env.PYTHON_VERSION }}
          enable-cache: true
      - run: uv sync --locked
      - run: uv run --locked ruff check ${{ env.LINT_TARGETS }}
      - run: uv run --locked ruff format --check ${{ env.LINT_TARGETS }}
      - run: uv run --locked pytest -q --cov=telco_churn --cov-report=term-missing

  dependency-audit:
    name: pip-audit (locked environment)
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1   # v7.0.1
      - uses: astral-sh/setup-uv@v10.3.3                                  # release tag
        with: {version: "0.13.9", python-version: "3.12", enable-cache: true}
      - run: uv sync --locked
      - run: uv run --locked pip-audit

  secret-scan:
    name: Gitleaks secret scan
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1   # v7.0.1
      - name: Install Gitleaks 8.30.1 (checksum verified)
        env:
          GITLEAKS_VERSION: "8.30.1"
          GITLEAKS_SHA256: "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
        run: |
          curl -sSfL -o gitleaks.tar.gz "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
          echo "${GITLEAKS_SHA256}  gitleaks.tar.gz" | sha256sum -c -
          mkdir -p "${RUNNER_TEMP}/gitleaks"
          tar -xzf gitleaks.tar.gz -C "${RUNNER_TEMP}/gitleaks" gitleaks
          chmod +x "${RUNNER_TEMP}/gitleaks/gitleaks"
          echo "${RUNNER_TEMP}/gitleaks" >> "${GITHUB_PATH}"
      - name: Materialise the tracked files
        run: |
          mkdir -p "${RUNNER_TEMP}/scan"
          git archive --format=tar HEAD | tar -x -C "${RUNNER_TEMP}/scan"
      - run: gitleaks dir --redact --exit-code 1 --no-banner "${RUNNER_TEMP}/scan"
```

The unabridged file is in the working tree at `.github/workflows/ci.yml`, validated by the
`check-yaml` hook; the block above is a reader-friendly copy for this report.

**Why `ruff format --check` reports differences without rewriting files in CI.** `--check`
compares the formatted result with the file on disk and exits non-zero; only `ruff format`
(without `--check`) writes. CI must never modify the checkout: the result has to describe the
commit under review, and a job that rewrites files would either hide the problem or leave
changes nobody committed.

## 3. Red check on a pull request (demonstrated)

Branch `tp05/regression`, PR [#6](https://github.com/KhadijaBenhamida/telco-churn/pull/6):
`OneHotEncoder(handle_unknown="ignore")` → `"error"` in `build_preprocessor()`. The change is
syntactically valid and Ruff-clean, so only a behavioural check can catch it.

**Red run** ([44562157190](https://github.com/KhadijaBenhamida/telco-churn/actions/runs/44562157190)):

```text
Ruff and TP04 tests    fail    24s    ← the only failing job
Gitleaks secret scan   pass     5s
pip-audit              pass    19s

FAILED tests/test_prediction.py::test_unseen_category_and_missing_number_are_accepted
  - ValueError: Found unknown categories ['Three year contract'] in column 0 during transform
FAILED tests/test_prediction.py::test_unknown_category_does_not_leak_into_other_columns
  - ValueError: Found unknown categories ['Three year contract'] in column 0 during transform
2 failed, 25 passed, 8 skipped in 7.50s
```

Diagnosis (three sentences): the encoder now rejects a category it never saw during fitting,
which is exactly the permissive policy the TP02 contract promises. The two failing tests are
the ones written for that risk, and the traceback points at
`sklearn/preprocessing/_encoders.py:221` as expected. The 8 skipped tests are the CSV-dependent
ones — the runner has no `data/raw/telco.csv`, which is why the suite was made CSV-free in TP04.

**Green run after the fix** ([44562327009](https://github.com/KhadijaBenhamida/telco-churn/actions/runs/44562327009),
commit `685416a` "fix: restore the permissive unknown-category policy"):

```text
Ruff and TP04 tests    pass    35s
Gitleaks secret scan   pass     6s
pip-audit              pass    16s
```

No test was touched, no assertion weakened, no `--no-verify`. PR #6 was merged after the green
run, so `main` carries the fixed implementation and both runs stay attached to the PR.


## 4. Security checks

### 4A. Dependency audit — `pip-audit` 2.10.1

Added with `uv add --dev pip-audit` (manifest and `uv.lock` committed). It audits the
**installed** distributions of the locked project environment — dev tools included — not every
platform-specific wheel in the lockfile, and it needs network access to query the advisory
database.

`reports/tp05_evidence/04_pip_audit.txt`:

```text
$ uv run --locked python -m pip_audit     # `uv run --locked pip-audit` in CI
Name        Skip Reason
----------- --------------------------------------------------------------------------
telco-churn Dependency not found on PyPI and could not be audited: telco-churn (0.1.0)
No known vulnerabilities found
exit=0
```

**Outcome: no advisory found**, so no update is proposed. The single `Skip` line is our own
project package: it is not published on PyPI, so there is nothing to compare with an advisory
— a report line, not a finding, and it does not change the exit status. (`pip-audit.exe`
cannot be spawned on this machine because of the same application-control policy that blocks
`mlflow.exe` and `pre-commit.exe`, hence the module form locally; the workflow uses the plain
command on Linux.)

### 4B. Secret scan — Gitleaks 8.30.1

Binary from the official release; the checksum is verified locally and again in CI:

```text
sha256(gitleaks_8.30.1_windows_x64.zip)  = d29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e
published checksum                        = d29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e  -> match
sha256(gitleaks_8.30.1_linux_x64.tar.gz)  = 551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb
```

Clean scan of the committed snapshot (`git archive HEAD` — exactly what CI checks out: no
`.git` objects, no `.venv`, no MLflow store), `reports/tp05_evidence/05_gitleaks_repo.txt`:

```text
$ gitleaks dir --redact --exit-code 1 <clean-checkout>
INF scanned ~13329607 bytes (13.33 MB) in 410ms
INF no leaks found
exit=0
```

Synthetic detection with the classroom rule (config kept **outside** the repository, extends
the default rules), `reports/tp05_evidence/06_gitleaks_demo.txt`:

```toml
[extend]
useDefault = true

[[rules]]
id = "tp05-demo-marker"
description = "Synthetic classroom marker"
regex = '''TP05_DEMO_[A-Z0-9]{12}'''
```

```text
$ gitleaks dir --config gitleaks-demo.toml --redact --exit-code 1 <demo-dir>
INF scanned ~93 bytes in 2.54ms
WRN leaks found: 1                              exit=1
$ rm <demo-dir>/notes.txt                      # synthetic marker removed
$ gitleaks dir --config gitleaks-demo.toml --redact --exit-code 1 <demo-dir>
INF no leaks found                             exit=0
```

The marker (`TP05_DEMO_ABCDEFGH1234`) never entered the repository and the default rules were
not replaced. `--redact` keeps the value out of the log, so a real finding could be shared
without leaking it.

**Scope, and why deleting a secret is not enough.** `gitleaks dir` scans a directory snapshot:
it proves the current tree is clean, not that history is. A credential that was committed stays
in the Git objects (and in clones, forks and CI caches) after the file is fixed, so it must be
**revoked or rotated** first; rewriting history is a separate, disruptive operation that cannot
reach copies other people already pulled.

## 5. Dependabot (bot PRs opened)

`.github/dependabot.yml` (on `main`) requests **version** updates weekly for two ecosystems:
`uv` at `/` (keeps `pyproject.toml` + `uv.lock` current) and `github-actions` at `/` (keeps the
action pins fresh). Each entry sets a commit-message prefix, labels and a five-PR limit.
Version updates are separate from Dependabot *alerts* and *security* updates, which depend on
repository settings.

The bot acted on the configuration within the same session and opened three proposals:

| PR | Update | Type |
|---|---|---|
| [#1](https://github.com/KhadijaBenhamida/telco-churn/pull/1) | `sqlalchemy` 2.0.58 → 2.1.8 | runtime dependency |
| [#2](https://github.com/KhadijaBenhamida/telco-churn/pull/2) | `pytest` 9.1.1 → 9.2.0 | dev dependency |
| [#3](https://github.com/KhadijaBenhamida/telco-churn/pull/3) | `uv-build` `>=0.13.6,<0.14.0` → `>=0.13.8,<0.14.0` | build backend |
| [#4](https://github.com/KhadijaBenhamida/telco-churn/pull/4) | `pandas` 3.1.0 → 3.1.1 | runtime dependency |

They are left open on purpose: the lab asks how I would review them, not to merge blindly.
Review procedure — read the release notes/changelog between the two versions; check that the
diff only bumps the version and the `uv.lock` hashes (no new transitive dependency);
confirm the three jobs are green on that PR **for that commit**; scan the pytest output for
behaviour changes (a pandas minor can alter dtype handling, which is exactly what the
`TotalCharges` parsing tests guard); then merge. For a major bump I would additionally run
`uv sync --locked` and the full suite locally, and for a `github-actions` update I would check
the new SHA against the upstream release tag, since that is what the pin documents.

PR #1 is the concrete example of why CI green is not sufficient: `sqlalchemy 2.1` ships
compiled `*_cy.pyd` extensions that this Windows machine's application-control policy blocks,
which is exactly why `pyproject.toml` pins `>=2.0,<2.1` (TP03 note). The Linux runner has no
such policy, so the PR would very plausibly be green — and the project would still break on the
author's laptop. That is a case for a manual `uv sync --locked` check on the target platform,
not an auto-merge. **PR #1 was therefore closed with that reasoning** instead of merged.

The other three were opened *before* the workflow existed on `main`, so GitHub shows
`no checks reported` on them — a workflow is not applied retroactively to an existing PR head.
I asked Dependabot to rebase them (`@dependabot rebase`); once it pushes a new commit the
three jobs run on that PR and the procedure above applies as written. Until then they stay
open: merging an unvalidated dependency bump would be exactly the behaviour this TP argues
against. If a rebased PR fails `quality`, it gets closed with the failing run link rather than
`--admin`-merged.



## 6. Recorded GitHub Actions runs

| Event | Workflow / jobs | Run |
|---|---|---|
| PR [#5](https://github.com/KhadijaBenhamida/telco-churn/pull/5) — add the pipeline (merged) | quality ✅ · pip-audit ✅ · secret-scan ✅ | [44561877608](https://github.com/KhadijaBenhamida/telco-churn/actions/runs/44561877608) |
| PR #6 — regression (red) | quality ❌ · pip-audit ✅ · secret-scan ✅ | [44562157190](https://github.com/KhadijaBenhamida/telco-churn/actions/runs/44562157190) |
| PR #6 — after the fix (green, merged) | quality ✅ · pip-audit ✅ · secret-scan ✅ | [44562327009](https://github.com/KhadijaBenhamida/telco-churn/actions/runs/44562327009) |

The two earlier failures (setup-uv SHA, Gitleaks PATH) are listed in §0 together with their
fixes; they are part of how the pipeline was made reproducible rather than something to hide.


**Limits of these checks.** CI proves that the checks ran on the committed code, not that the
model is good: the suite protects the input/output contract, Ruff protects style and obvious
defects, `pip-audit` only knows published advisories (no OS packages, no model supply chain),
and Gitleaks only matches patterns it has rules for. A green run means "nothing we know how to
look for was found".
