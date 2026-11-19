# TP05 — GitHub Actions, dependency audit and secret scanning

Repository: <https://github.com/KhadijaBenhamida/telco-churn> (public, default branch `main`)
Project: `TP1-2 MLOPS/telco-churn` · Python 3.12 · uv 0.13.9 · Ruff 0.17.5 · pytest 9.1.1
pip-audit 2.10.1 · Gitleaks 8.30.1

## 0. Status at a glance

| Check | Where | Result |
|---|---|---|
| Ruff + TP04 tests (35 tests) | local reproduction of the workflow | **pass** (`reports/tp05_evidence/01_local_checks.txt`) |
| Dependency audit (`pip-audit`) | local, locked environment | **pass** — no known vulnerabilities (`04_pip_audit.txt`) |
| Secret scan (Gitleaks 8.30.1) | local, clean checkout + synthetic marker | **pass** / detection demonstrated (`05`, `06`) |
| Dependabot | `.github/dependabot.yml` on `main` | **in place** (`uv` + `github-actions`, weekly) |
| GitHub Actions runs | `.github/workflows/ci.yml` | **blocked by a token permission** — see §6 |

The workflow file is complete and valid (`check-yaml` in the pre-commit suite passes on it),
but it cannot be pushed from this machine: the GitHub credential available here (the `gh`
OAuth token, scopes `gist, read:org, repo`) lacks the `workflow` scope, and GitHub refuses
any `git push` or REST call that creates a file under `.github/workflows/`:

```text
! [remote rejected] tp05/ci -> tp05/ci (refusing to allow an OAuth App to create or update
  workflow `.github/workflows/ci.yml` without `workflow` scope)
error: failed to push some refs to 'https://github.com/KhadijaBenhamida/telco-churn.git'
```

Everything that does **not** need that scope is already merged on `main` (Dependabot
configuration, `pip-audit` dev dependency + lockfile, all local evidence). §6 gives the
single command that finishes the remaining part.

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

Actions are pinned by commit SHA (with the release in a trailing comment):
`actions/checkout` v7.0.1 → `3d3c42e5aac5ba805825da76410c181273ba90b1`,
`astral-sh/setup-uv` v10.3.3 → `68559e99cc807c1eb7e9238fd43a15bba4600eec`. The Gitleaks
release is downloaded during the job and its published SHA-256 is verified before the binary
is executed.

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
      - uses: astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba4600eec  # v10.3.3
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
      - uses: astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba4600eec  # v10.3.3
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

## 3. Red check on a pull request

The regression is prepared but **cannot be demonstrated until §6 is unblocked**, because no
workflow can run before the workflow file is on GitHub. The identical change was already
proven to fail the suite locally in TP04
(`reports/tp04_evidence/05_regression_failure.txt`):

```text
$ uv run pytest tests/test_prediction.py -q     # with handle_unknown="error"
FAILED tests/test_prediction.py::test_unseen_category_and_missing_number_are_accepted
FAILED tests/test_prediction.py::test_unknown_category_does_not_leak_into_other_columns
E   ValueError: Found unknown categories ['Three year contract'] in column 0 during transform
2 failed, 5 passed in 36.71s
```

Procedure the finishing script executes (branch `tp05/regression`, PR against `main`):

1. `OneHotEncoder(handle_unknown="ignore")` → `"error"` in `build_preprocessor()`
   (syntactically valid and Ruff-clean, so only a behavioural check can catch it).
2. Push and open the PR → the `quality` job fails on
   `test_unseen_category_and_missing_number_are_accepted`; the other jobs stay green, which
   is exactly the "which check failed" signal the job names provide.
3. Read the traceback in the Actions log, restore `handle_unknown="ignore"`, push again →
   the same job turns green.
4. Merge the PR and keep the branch; both runs stay linked to the PR.

What will be quoted here once it runs: the failing run URL, the passing run URL and the two
or three sentences of diagnosis.

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

## 5. Dependabot

`.github/dependabot.yml` (on `main`) requests **version** updates weekly for two ecosystems:
`uv` at `/` (keeps `pyproject.toml` + `uv.lock` current) and `github-actions` at `/` (keeps the
SHA pins fresh). Each entry sets a commit-message prefix, labels and a five-PR limit. Version
updates are separate from Dependabot *alerts* and *security* updates, which depend on
repository settings.

How I would review an update PR: read the release notes and the changelog between the two
versions; check that the diff only bumps the version and the lockfile hashes; confirm the
`quality`, `dependency-audit` and `secret-scan` jobs are green on that PR; inspect the pytest
output for behaviour changes; then merge. For a major bump I would run `uv sync` plus the full
suite locally first. For a `github-actions` update I would check the new SHA against the
upstream release tag, because that SHA is what the pin documents.

## 6. Finish in one step (needs a token with the `workflow` scope)

```powershell
gh auth refresh -s workflow     # approve in the browser, then:
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\khadi\mlops_stage\finish_tp05.ps1
```

The script pushes `.github/workflows/ci.yml`, opens PR #1 (CI runs), then creates the
regression branch and PR #2, captures the red run, restores the implementation, captures the
green run, merges both PRs and finally writes the run links into this report. Until then the
repository holds the workflow content above, the Dependabot configuration and all local
evidence, but no Actions runs.

**Limits of these checks.** CI proves that the checks ran on the committed code, not that the
model is good: the suite protects the input/output contract, Ruff protects style and obvious
defects, `pip-audit` only knows published advisories (no OS packages, no model supply chain),
and Gitleaks only matches patterns it has rules for. A green run means "nothing we know how to
look for was found".
