# TP04 — Testing and code quality for the churn project

Project: `TP1-2 MLOPS/telco-churn` (TP02 pipeline + TP03 tracking, unchanged behaviour)
Dataset: IBM Telco Customer Churn · Python 3.12 · pytest 9.1.1 · pytest-cov 7.1.0 ·
Ruff 0.17.5 · pre-commit 4.7.0 · coverage 7.17.3

The goal of this TP is not a better model: it is evidence that the preprocessing and
prediction contract from TP02/TP03 still holds after a change, and that cheap local
checks catch what the tests cannot.

## 1. The contract under test

| Contract item | Where it is documented | Where it is protected |
|---|---|---|
| Six features required: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod` | `pipeline.FEATURES` | `tests/test_input_contract.py` |
| `Churn` and `customerID` are never features | `prepare_features()` drops them | `test_target_and_identifier_are_not_model_inputs` |
| Numeric parsing: blank cell = missing, malformed text = error | `clean_telco_numeric_columns()` | `test_blank_total_charges_is_treated_as_missing_and_imputed`, `test_malformed_numeric_value_still_raises` |
| Missing numbers imputed (median), unknown categories ignored | `build_preprocessor()` | `test_unseen_category_and_missing_number_are_accepted` |
| One prediction row per customer, labels in `{0, 1}` | `predict_customers()` | `test_prediction_returns_one_row_per_customer` |
| `Churn = Yes` is class `1`; class order read from `classes_` | `prepare_target()`, `positive_class_scores()` | `test_positive_class_score_uses_classes_order` |
| Probabilities are valid (shape, bounds, rows sum to 1) | sklearn contract + our threshold policy | `test_probability_contract` |
| Saving and reloading keeps predictions and fitted statistics | `prediction.load_model()` / joblib | `tests/test_persistence.py` |
| Model metadata keeps features, positive class, threshold, source hash | `training.save_selected_model()` | `test_saved_model_metadata_records_the_contract` |
| Split keeps customers disjoint | `training.make_split()` | `tests/test_tp02.py` (needs the real CSV) |

## 2. The four required tests (plus the extras that came out of the coverage gap)

Fixtures live in `tests/conftest.py`: twelve invented customers with the six columns
(raw string `TotalCharges`, mixed contracts/services/payment methods), a balanced binary
target, and the *real* pipeline builder fitted on that fixture. No CSV read, no network,
no MLflow, no `fit()` of a large model — the whole TP04 block runs in ~1 s.

| # | Test | Real defect it would catch |
|---|---|---|
| A | `test_missing_required_feature_is_rejected` | a refactor that stops validating the feature list (the boundary would silently score a column-shifted batch) |
| A+ | `test_every_documented_feature_is_required` (parametrized ×6) | the same check drifting for only one feature |
| B | `test_unseen_category_and_missing_number_are_accepted` | tightening `OneHotEncoder(handle_unknown="error")` or removing the imputer — the permissive policy is intentional |
| C | `test_probability_contract` | an estimator/scaler change that produces NaN, out-of-range or non-normalised probabilities |
| D | `test_reload_preserves_predictions` | saving only the classifier, or a joblib/pickle incompatibility, changing predictions |
| + | `test_reload_keeps_the_learned_preprocessing_statistics` | a reload path that quietly refits or loses the imputer medians |
| + | `test_saved_model_metadata_records_the_contract` | dropping the provenance JSON write (features, positive class, threshold, source hash) |
| + | `test_find_latest_model_*` (2 tests) | a model-discovery helper that raises `StopIteration` or picks an arbitrary file |

`Test A` uses `pytest.raises(KeyError, match="Contract")`: the expected exception *and* a
stable fragment of the message, so a different crash cannot pass as success.
`Test D` uses `tmp_path`, so the repository's `models/` artifact is never overwritten, and
compares class labels exactly plus probabilities with `atol=1e-12`.

**Does a successful prediction on an unseen category prove the prediction is accurate?**
No. It only proves the encoder tolerated the value (`handle_unknown="ignore"` maps it to an
all-zero block). Accuracy would need labelled data and a metric; the test checks
robustness of the input contract, not quality.

## 3. Regression demonstration (real failure, real fix)

The defect: a colleague "hardens" preprocessing and switches
`OneHotEncoder(handle_unknown="ignore")` to `handle_unknown="error"` in
`build_preprocessor()`. Training still runs on the CSV (the training categories are all
known); only inference on a new category breaks.

```diff
-            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
+            ("encoder", OneHotEncoder(handle_unknown="error", sparse_output=False)),
```

`uv run pytest tests/test_prediction.py -q` (`reports/tp04_evidence/05_regression_failure.txt`):

```text
FAILED tests/test_prediction.py::test_unseen_category_and_missing_number_are_accepted
FAILED tests/test_prediction.py::test_unknown_category_does_not_leak_into_other_columns
E   ValueError: Found unknown categories ['Three year contract'] in column 0 during transform
2 failed, 5 passed in 36.71s
```

The traceback points at `sklearn/preprocessing/_encoders.py` `_check_unknown`, i.e. exactly
the policy the tests describe, and the message names the offending value, so the diagnosis
needs no guessing. The test that failed is the one written for this risk, not a generic
smoke test.

Fix: restore `handle_unknown="ignore"` (the documented permissive policy) and rerun
(`reports/tp04_evidence/06_regression_fixed.txt`):

```text
$ uv run pytest tests/test_prediction.py -q
7 passed in 7.41s
$ uv run pytest -q
35 passed in 31.81s
```

Nothing else was changed to make the test pass — no assertion weakened, no test deleted, no
`--no-verify`.

## 4. Ruff — inspect, fix, verify

Configuration (in `pyproject.toml`):

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]   # import style, statement style, runtime errors, Pyflakes, isort
```

Scope: `src/telco_churn tests scripts` (package, tests, CLI entry points). `mlartifacts/`,
`mlruns/` and `.venv/` are never scanned.

Before (`reports/tp04_evidence/02_ruff_before.txt`): **11 lint errors** — `I001` unsorted
import blocks and `F401` unused imports (`pandas` and `compare_models` in
`scripts/train_model.py`, `PROJECT_ROOT` in `pipeline.py`) — plus **8 files the formatter
would rewrite**.

After reviewing the diff, the automatic fixes were accepted
(`reports/tp04_evidence/03_ruff_fix.txt`):

```text
$ uv run ruff check --fix src/telco_churn tests scripts
Found 11 errors (11 fixed, 0 remaining).
$ uv run ruff format src/telco_churn tests scripts
8 files reformatted, 9 files left unchanged
$ uv run ruff check src/telco_churn tests scripts
All checks passed!
$ uv run ruff format --check src/telco_churn tests scripts
17 files already formatted
```

The diff was reviewed before accepting: import reordering, removal of two genuinely unused
imports, wrapping to 100 columns. No rule was ignored, no package directory excluded, and
the suite was re-run afterwards (35 passed) — the evidence that reformatting did not change
behaviour. After the last test was added, `ruff format --check` reports
`18 files already formatted` (`reports/tp04_evidence/10_final_verification.txt`).

## 5. Coverage — result, gap and a proposed check

```text
$ uv run pytest -q --cov=telco_churn --cov-report=term-missing --cov-report=xml:coverage.xml
Name                            Stmts   Miss Branch BrPart  Cover   Missing
src\telco_churn\__init__.py         6      0      0      0   100%
src\telco_churn\dataset.py         56     20      8      2    62%   27->26, 30, 56-102, 106-119
src\telco_churn\pipeline.py        44      2     12      0    96%   83-84
src\telco_churn\prediction.py      23      1      4      1    93%   31
src\telco_churn\tracking.py       173     65     26      1    58%   82-89, 94-97, 125-126, 140-141, 151-152, 258-260, 336-340, 347-351, 370-447, 473-490, 505, 510-515
src\telco_churn\training.py        88      3      8      1    94%   54-57
TOTAL                             390     91     58      5    74%
35 passed in 31.51s
```

Coverage went from 69 % to 74 % because behaviours that were never executed are now tested
(`compare_models`, the metadata contract of `save_selected_model`, both `find_latest_model`
paths). The percentage is not the objective: 74 % with weak assertions would be worse than
70 % with the four contract tests above.

**Uncovered behaviour chosen: `dataset.build_report()`** (62 % on `dataset.py`, lines 56-102).
It is the TP01 provenance report — dataset shape, target counts, duplicate IDs, whitespace
counts, source SHA-256 — and it produced `reports/data_overview.json`, which TP03's
provenance story relies on. No test executes it, so a refactor could silently drop a field.
Proposed test (not written here, to keep this TP's suite focused):

```python
def test_build_report_describes_a_small_dataset(tmp_path):
    source = tmp_path / "tiny.csv"
    source.write_text(
        "customerID,tenure,MonthlyCharges,TotalCharges,Contract,"
        "InternetService,PaymentMethod,Churn\n"
        "0001-A,1,29.85,29.85,Month-to-month,DSL,Mailed check,No\n"
        "0002-B,2,56.95,113.90,One year,Fiber optic,Electronic check,Yes\n",
        encoding="utf-8",
    )
    report = dataset.build_report(source, tmp_path / "overview.json")

    assert report["dataset"]["rows"] == 2
    assert report["target"]["counts"] == {"No": 1, "Yes": 1}
    assert report["duplicate_customer_ids"] == 0
    assert len(report["source"]["sha256"]) == 64
    assert json.loads((tmp_path / "overview.json").read_text())["dataset"]["columns"] == 8
```

**What does an executed line tell you?** Only that the interpreter reached it. It says
nothing about whether the result was right — the line can run with the wrong branch, the
wrong input or a silently wrong value. **Can a test execute a line without asserting the
right behaviour?** Yes: calling `predict_customers(...)` and ignoring the result adds
coverage and proves nothing, which is why every test above asserts a value, a type, a shape
or an exception. **Third-party code outside scope:** `site-packages` (scikit-learn, pandas,
MLflow), `scripts/` and the generated MLflow stores — `--cov=telco_churn` measures only the
package. The 58 % on `tracking.py` is expected: logging and registration need a live
tracking server, which TP03 demonstrates as runtime evidence rather than unit tests.

## 6. Pre-commit hooks

`uv run pre-commit` cannot spawn `pre-commit.exe` on this machine (the same
application-control policy that blocks `mlflow.exe`), so the commands use
`uv run python -m pre_commit ...`. The installed Git hook itself calls
`python -m pre_commit`, so `git commit` works normally.

`reports/tp04_evidence/07_pre_commit_setup.txt` (first run: two hooks from the
`pre-commit-hooks` environment intermittently hit the same application-control policy,
`[WinError 4551]`; the retry in `07b_pre_commit_all_files.txt` passes):

```text
$ uv run python -m pre_commit validate-config
exit=0
$ uv run python -m pre_commit install
pre-commit installed at .git\hooks\pre-commit
$ uv run python -m pre_commit run --all-files           # first run: hooks fix files
fix end of files.........................................................Failed
- files were modified by this hook
Fixing reports/tp03_summary.md ... Fixing README.md ... Fixing pyproject.toml
ruff check...............................................................Passed
ruff format..............................................................Passed
$ uv run python -m pre_commit run --all-files           # second run
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check for added large files..............................................Passed
ruff check...............................................................Passed
ruff format..............................................................Passed
exit=0
```

`reports/tp04_evidence/08_pre_commit_blocked.txt` — a deliberately mis-formatted line is
staged and committed:

```text
$ git commit -m "tp04 pre-commit demonstration"
trim trailing whitespace.................................................Failed
- files were modified by this hook
fix end of files.........................................................Failed
- files were modified by this hook
ruff check...............................................................Passed
ruff format..............................................................Failed
- hook id: ruff-format
- files were modified by this hook
2 files reformatted, 13 files left unchanged
exit=1

$ git diff -- scripts/tp03_smoke_run.py
-        run_id    =    run.info.run_id
+        run_id = run.info.run_id
$ git log --oneline -1
abcc9ae Merge origin/main ...          <- no new commit: the hook refused it
```

The commit is refused, the file on disk is the formatter's version, `git diff` shows exactly
that cosmetic change, and after `git add` the retried commit succeeds
(`reports/tp04_evidence/09_pre_commit_success.txt`). No `--no-verify` was used and no hook
was disabled.

Limits: hooks are local and can be bypassed (`--no-verify`), they only see staged files, and
they run fast checks only — no training, no MLflow server. That is exactly why TP05 moves the
same checks to CI, where they cannot be skipped.


## 7. Quality Gate discussion

| Layer | What it observes | What it cannot tell you |
|---|---|---|
| pytest | executed behaviour against assertions | whether the oracle/expectation is right, or uncovered behaviour |
| Ruff (static) | syntax, imports, unused names, style | runtime values, logic errors, data problems |
| Coverage | which lines/branches ran | whether the results were correct |
| Quality Gate (policy) | the analysis results against configured thresholds and a "new code" definition | anything the underlying tools did not measure |

**If the tests pass, what can still be wrong?** Everything that is not asserted: the
`build_report` provenance path above, the tracking/registration paths (58 %), the model's
actual quality, calibration, fairness across `Contract`/`InternetService` groups, data
drift (a new payment method appearing in production), dependency vulnerabilities, and the
fact that the artifact in the registry was trained by code that may no longer match `HEAD`.
Tests also pass on the wrong split: nothing here proves the model is good, only that the
contract holds.

**If a Quality Gate passes, does it prove the model is useful, fair or secure?** No. A gate
is a policy over analysis results — typically "no blocker issues" and "coverage on new code
above X %". It is silent about predictive quality, group fairness, privacy and adversarial
robustness unless those checks exist and are wired in. Analysis completing and the gate
passing are also two different events: the scanner can succeed while the gate fails on a
threshold, and a green gate only means the rules were met.

**SonarQube Community Build (optional, not run here).** Running it would add: one concrete
issue with file/line (for example an unused variable or the cognitive complexity of
`log_candidate_run`), the distinction between an issue and coverage, the gate result with the
failing condition responsible, and the difference between "analysis succeeded" and "gate
passed". Coverage must be imported
(`sonar.python.coverage.reportPaths=coverage.xml`) because SonarQube does not run pytest,
and the source/test paths must match `src/telco_churn` and `tests`. Community Build covers
main-branch analysis, coverage import and the gate; pull-request-specific analysis is a
commercial feature, and a GitHub-hosted runner cannot reach a server on `localhost` — so
TP05 keeps Ruff and pytest as the required PR checks and treats SonarQube as an extension.

## 8. Files and how a peer reproduces this

| Path | Purpose |
|---|---|
| `tests/conftest.py` | synthetic customers, target, fitted pipeline fixtures (no CSV, no network) |
| `tests/test_input_contract.py` | missing feature, `Churn`/`customerID` exclusion, numeric parsing rules |
| `tests/test_prediction.py` | permissive variation, probability contract, `classes_` order, threshold |
| `tests/test_persistence.py` | joblib round trip, imputer statistics, model discovery, metadata contract |
| `tests/test_comparison.py` | `compare_models` table contract |
| `pyproject.toml` | dev group + pytest/ruff/coverage configuration |
| `.pre-commit-config.yaml` | pinned hooks (ruff 0.17.5, pre-commit-hooks v6.0.0) |
| `reports/tp04_evidence/*.txt` | raw logs quoted above |

```powershell
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"
uv sync
uv run pytest -q
uv run ruff check src/telco_churn tests scripts
uv run ruff format --check src/telco_churn tests scripts
uv run pytest --cov=telco_churn --cov-report=term-missing --cov-report=xml:coverage.xml
uv run python -m pre_commit install
uv run python -m pre_commit run --all-files
```

The eight tests that need the real Telco CSV (`tests/test_tp02.py`, `tests/test_tp03.py`)
skip automatically when `data/raw/telco.csv` is absent, because the CSV is git-ignored; add
it (see `references/data_source.md`) to run the full regression. The TP04 behaviour tests
never need it.
