# TP03 — Track, compare and recover ML experiments with MLflow

Project: `TP1-2 MLOPS/telco-churn` · Dataset: IBM Telco Customer Churn
(`datasets.csv`, SHA-256 `62ca518b1ebbcd4c832c18c1a21629b03abe9033fef66fe2e38d8d030658a20c`)
Experiment: **`telco-churn-tp03`** (id `1`) · Tracking server: `http://127.0.0.1:5000`
Final evidence runs were produced from Git commit **`91da09b8d5f9612f61b47768eabe80252133df54`**
with a clean worktree (`git_worktree_dirty=false` in every run's `provenance.json`).

TP02 was reused unchanged: same six features (`tenure`, `MonthlyCharges`,
`TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`), same stratified
60/20/20 customer allocation (`random_state=42`), preprocessing fitted inside the
pipeline on the training split only, evaluation on validation, test set reserved.
`Churn = Yes → 1`, threshold `0.5` fixed before the comparison, primary metric
average precision from continuous class-1 scores.

## 0. Run it again (three terminals)

```powershell
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"

# terminal 1 — tracking server (keep running)
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

# terminal 2 — section 1 check, then the tracked comparison
uv run python scripts/tp03_smoke_run.py
uv run python scripts/train_mlflow.py --source "C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv"

# terminal 3 (fresh process) — reload through the alias
uv run python scripts/predict_mlflow.py
```

Windows notes for this machine (recorded because they change the commands, not
the design):

- `sqlalchemy` is pinned to `>=2.0,<2.1`: the 2.1 wheels ship compiled
  `*_cy.pyd` extensions that the local application-control policy blocks, while
  2.0.58 falls back to its pure-Python implementation.
- MLflow's end-of-run message contains an emoji; `PYTHONUTF8=1` (or a UTF-8
  console) is required, otherwise `end_run` raises `UnicodeEncodeError` on a
  cp1252 console. The scripts also reconfigure `stdout` to UTF-8 defensively.
- On this machine `uv run mlflow ...` occasionally cannot spawn `mlflow.exe`
  (same policy) so the helper `scripts/start_tracking_server.ps1` uses the
  equivalent `uv run python -m mlflow server ...`.

## 1. Section 1 — experiment, run and storage

Smoke run (one parameter, one metric) proving client ↔ server connectivity:

| Item | Value |
|---|---|
| Experiment | `telco-churn-tp03`, id `1` (created by the first client call) |
| Run name / id | `smoke-check` / `a6329bb4747d4a6badd9cac175b3897b` |
| Recorded | `purpose="tracking connectivity check"`, `threshold=0.5`, `smoke_value=1.0` |
| Backend store | `mlflow.db` (SQLite) — created next to `pyproject.toml` |
| Artifact store | `mlartifacts/` — created when the first model/artifact was logged |

**Experiment vs run:** the experiment is the named container (a topic/project,
here `telco-churn-tp03`); a run is one execution inside it with its own id,
parameters, metrics, tags, artifacts, dataset inputs and logged models. The
browser page does **not** execute the training code: it only queries the
tracking server, which reads `mlflow.db` and the artifact store. Training runs in
the terminal process and pushes records over HTTP.

**If a colleague receives only `mlflow.db`:** they get the metadata (runs,
params, metrics, tags, dataset records, registry entries) but not the model
files, confusion matrices or `provenance.json`, because those live in the
artifact store. They also need a tracking server or the same tracking URI, the
project code and a compatible environment. Loading
`models:/telco-churn@candidate` would fail: the registry row exists, the
artifact bytes do not.

## 2. Section 2 — what produced each result

Every candidate run records (see any run page in the UI):

| Category | Recorded content |
|---|---|
| Parameters | `candidate`, `model_type`, real hyperparameters as `hp_*` (`hp_C`, `hp_max_iter`, `hp_max_depth`, `hp_n_estimators`, `hp_min_samples_leaf`, `hp_strategy`), `random_state=42`, `threshold=0.5`, `positive_class=1`, `train_rows=4225`, `valid_rows=1409`, `n_features=6` |
| Metrics | `val_average_precision` (primary), `val_precision`, `val_recall`, `val_f1`, `val_accuracy` |
| Tags | `git_commit`, `git_worktree_dirty`, `data_sha256`, `data_source`, `split_seed`, `split_protocol`, `primary_metric`, `threshold_policy`, `positive_class`, `candidate_family`, `test_set_usage` |
| Artifacts | `validation/<run>_confusion_matrix.png`, `provenance.json`, `validation/validation_metrics.json` |
| Dataset inputs | `telco_train` (`training`) and `telco_validation` (`validation`) via `mlflow.data.from_pandas(..., targets="Churn")` |
| Model | the complete fitted `Pipeline(preprocessor → classifier)` logged as `churn_pipeline` in **every** candidate run |

Notes kept in the report because they are easy to get wrong:

- The class-1 score is selected through `classes_` (`classes_.index(1)`), not by
  assuming column 1; average precision uses those continuous scores, while
  precision/recall/F1 use the `score >= 0.5` decision.
- Confusion-matrix ordering is stated on the figure and in the JSON:
  **rows = true class, columns = predicted class**, label `0 = No churn`,
  `1 = Yes churn`. For the selected run: `[[926, 109], [196, 178]]`.
- **Metric vs artifact:** a metric is a numeric time series stored in the backend
  store, used to sort/compare runs; an artifact is a file (PNG, JSON, model)
  stored in the artifact store and fetched by path.
- Dataset records store name, digest and schema. They do not version-control the
  DataFrame and the digest is not the CSV SHA-256, so the original file hash and
  the per-split `customer_id_sha256` fingerprints are kept as well.

## 3. Section 3 — three fair candidate runs and the decision

All four runs share the same six features, the same customer allocation, the same
preprocessing definition, the same 0.5 threshold and the same validation
population; only the estimator (and its hyperparameters) changes. Each candidate
was fitted on the training split only and evaluated on validation.

| Run ID | Candidate / changed setting | Validation AP | Validation recall | Validation F1 | Decision |
|---|---|---:|---:|---:|---|
| `eb5c7ba6dcd64484996c312cd9dfeba3` | `forest-leaf5` — RandomForest(n_estimators=300, min_samples_leaf=5) | **0.652617** | 0.475936 | 0.538578 | **selected** (highest validation AP) |
| `5d3bc9a7f2dd47aaaa30d9ef4f39a874` | `logistic-C1` — LogisticRegression(C=1.0, max_iter=2000) | 0.647466 | 0.529412 | 0.589286 | rejected |
| `d8d2a9b547f54ae09117910bbff230bb` | `tree-depth6` — DecisionTree(max_depth=6) | 0.603320 | 0.497326 | 0.551929 | rejected |
| `f53fab065b2c45c6b598fdd2bdc7a1eb` | `baseline-majority` — DummyClassifier(most_frequent) | 0.265436 | 0.000000 | 0.000000 | rejected |

Precision at the same threshold: forest 0.6202, logistic 0.6644, tree 0.6200,
baseline 0.0000. Full machine-readable table: `reports/tp03_comparison.csv`.

Screenshots (in `evidence/tp03/`): `01_experiment_runs.png` and
`03_runs_table_with_metrics.png` (all five runs of the experiment, each candidate
row carrying both its registered model and its logged `churn_pipeline`),
`02_compare_metrics.png` (runs table sorted by validation average precision) and
`07_candidate_comparison.png` (grouped bars of AP / precision / recall / F1 built
from `mlflow.search_runs`, i.e. the same run-search evidence as the UI).

**Selection reason (validation only).** Average precision was fixed as the
primary metric before the runs, and the random forest reaches the highest
validation AP (0.6526 vs 0.6475 for logistic regression, a 0.005 gap, roughly
0.8 % relative). Because the rule was registered in advance, the forest is
selected; switching to logistic regression *after* seeing that it has better
recall/F1 and fits in 0.12 s instead of 3.9 s would be metric shopping on the
same validation split. That said, the gap is small compared with the spread one
would expect from a different split, so the operational comparison matters for
any real deployment: logistic regression is ~30× cheaper to fit, has higher
recall and F1 at threshold 0.5, and is easier to explain — if the team weighted
recall or interpretability above ranking quality, `logistic-C1` would be a
legitimate choice. The test set was not used at any point in this decision.

**Why different validation populations are harder to compare.** Average
precision depends on the class prevalence and on which hard negatives happen to
be in the evaluation set: a split with more churners or "easier" non-churners can
move AP by several points without any change in the model. If candidates were
scored on different populations, a difference would confound model quality with
dataset difficulty, so the comparison would no longer be fair.

**Why not select on the test set repeatedly.** Every look at the test set leaks
information: choosing the configuration that maximises test AP turns the test set
into a second validation set, and the reported score becomes optimistically
biased with no honest estimate left for the final report. That is why selection
is done on validation and the test set is reserved.

## 4. Section 4 — logged pipeline, registered version and alias

Each candidate run logs its fitted pipeline with `mlflow.sklearn.log_model`
(`name="churn_pipeline"`, `input_example=X_train.head(3)` — the six unencoded
feature columns without `Churn`/`customerID`, `serialization_format="cloudpickle"`).
The stored signature is therefore
`tenure: long, MonthlyCharges: double, TotalCharges: double, Contract: string,
InternetService: string, PaymentMethod: string`; it describes the *input* contract
(names and types) that the model expects before its own preprocessing runs.

The selected run was registered as a new version of `telco-churn` and the
`candidate` alias was moved onto it:

| Identifier | Value |
|---|---|
| Source run ID | `eb5c7ba6dcd64484996c312cd9dfeba3` |
| Logged model URI (returned by `log_model`) | `models:/m-477af62b21674c358b732911532f7a50` |
| Run-scoped model URI | `runs:/eb5c7ba6dcd64484996c312cd9dfeba3/churn_pipeline` |
| Registered name + version | `telco-churn` version **`1`** (status `READY`) |
| Alias | `candidate` → version `1` |
| Load URI used by consumers | `models:/telco-churn@candidate` |
| Git commit | `91da09b8d5f9612f61b47768eabe80252133df54` |
| Original CSV SHA-256 | `62ca518b1ebbcd4c832c18c1a21629b03abe9033fef66fe2e38d8d030658a20c` |

Run ID and logged model URI identify different objects: the run is the execution
with its evidence, the logged model is the serialized artifact that can be
loaded. The registered version ties a model to the managed name, and the alias is
a *mutable* pointer to the version that currently fills the `candidate` role. The
alias is not a copy and does not deploy anything.

Screenshots: `03_selected_run_overview.png` (parameters + validation metrics),
`04_selected_run_artifacts.png` (datasets/artifacts/model in the run),
`05_models_list.png` and `06_registered_model_telco_churn.png` (registry, version
`1`, alias `candidate`).

## 5. Section 5 — fresh-process reload and agreement

Exact command (new terminal, tracking server running, no training imported or
executed):

```powershell
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"
uv run python scripts/predict_mlflow.py --model-uri "models:/telco-churn@candidate"
```

Observed output:

```text
Tracking URI : http://127.0.0.1:5000
Model URI    : models:/telco-churn@candidate
Alias        : candidate -> version 1
Loaded       : Pipeline

customerID  recorded_prediction  reloaded_prediction  agree  recorded_probability  reloaded_probability  absolute_delta
0002-ORFBO                    0                    0   True              0.189672              0.189672             0.0
0004-TLHLJ                    1                    1   True              0.578996              0.578996             0.0
0013-EXCHZ                    0                    0   True              0.498506              0.498506             0.0
0017-IUDMW                    0                    0   True              0.018139              0.018139             0.0
0023-XUOPT                    1                    1   True              0.640818              0.640818             0.0

Class predictions agree : True
Max |probability delta| : 0.000e+00 (tolerance 1e-09)
RESULT: reloaded pipeline reproduces the selected run on the same batch.
```

The five customers are the same records for both runs: they are stored in
`reports/tp03_reference_predictions.json`, written by the selected training run
with its in-memory pipeline, and the reloaded process only parses them, loads the
model and predicts. Class predictions must match exactly; probabilities are
compared with tolerance `1e-9` (observed maximum delta `0.0`). Evidence file:
`reports/tp03_inference_check.json`.

**Why this is stronger evidence than predicting again in the training notebook.**
The notebook process still holds the fitted object, the fitted preprocessors and
every module in memory, so a prediction there says nothing about the artifact.
The fresh process has to resolve `candidate` in the registry, download the
serialized pipeline from the artifact store, import the right library versions
and rebuild the object from disk; only then can it predict. That exercises the
packaging, the tracking URI, the stored signature and the documented input
parsing at the same time — exactly the operations a consumer would perform.

**If the alias is later moved to a new version,** an already-loaded Python object
does **not** change: it is a materialized object in that process's memory. A new
load (`load_model("models:/telco-churn@candidate")`) is needed in that process to
pick up the new version; the alias only affects future resolutions.

## 6. Section 6 — restart, persistence and what each store preserves

**Before restart** the UI at <http://127.0.0.1:5000> showed experiment
`telco-churn-tp03` with five runs (`smoke-check` + four candidates) and the
registered model `telco-churn` v1 with alias `candidate`.

**After stopping the server** (all server processes killed) `GET /health` failed
with a connection error and the UI was unreachable, while `mlflow.db` and
`mlartifacts/` stayed untouched on disk. Nothing was deleted or renamed.

**After restarting from the same project root with the same command and options**
(`uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri
sqlite:///mlflow.db --artifacts-destination ./mlartifacts`), a fresh client
process reported:

```text
experiments after restart: [('telco-churn-tp03', '1'), ('Default', '0')]
runs after restart:
   forest-leaf5      eb5c7ba6dcd64484996c312cd9dfeba3  0.6526172282096577
   tree-depth6       d8d2a9b547f54ae09117910bbff230bb  0.6033197675871691
   logistic-C1       5d3bc9a7f2dd47aaaa30d9ef4f39a874  0.6474664703085159
   baseline-majority f53fab065b2c45c6b598fdd2bdc7a1eb  0.2654364797728886
   smoke-check       a6329bb4747d4a6badd9cac175b3897b  None
alias candidate -> 1 READY
registered versions: [('1', 'READY')]
models:/telco-churn@candidate -> [0, 1, 0, 0, 1] (match=True)
runs:/eb5c7ba6dcd64484996c312cd9dfeba3/churn_pipeline -> [0, 1, 0, 0, 1] (match=True)
models:/telco-churn/1 -> [0, 1, 0, 0, 1] (match=True)
expected predictions: [0, 1, 0, 0, 1]
```

So the runs, the registered version, the alias and the model bytes are all
durable: stopping the *server process* does not delete the *stored data*.
Restarting matters only because it must point at the same SQLite file and the
same artifact directory, from the same project root.

| Store | Preserves |
|---|---|
| `mlflow.db` (backend store) | experiments, run ids, status, params, metrics, tags, dataset records/digests, registry model rows, versions, aliases, model-version tags |
| `mlartifacts/` (artifact store) | every file body: `model.pkl`, `MLmodel`, `conda.yaml`/`requirements.txt`, `provenance.json`, `validation_metrics.json`, confusion-matrix PNGs |

**If the database survived but the model files were lost:** the UI would still
list the experiment, every run, all params/metrics/tags and the registry version
with its alias, but every artifact download and every `models:/...` load would
fail — the confusion matrices and `provenance.json` would be gone and the fitted
pipeline could not be recovered. Re-creating it means re-running training from
the recorded parameters/commit/CSV (a *new*, not identical, fitted object).

**If only the model files survived:** the bytes (`model.pkl`, `MLmodel`, …) would
exist somewhere in the artifact tree, but nothing links them to an experiment,
run, registered name, version or alias; run ids would not resolve and the UI
would show no trace. The only remaining provenance would be the `provenance.json`
stored next to the model, and a consumer would have to locate `model.pkl` by hand
and deserialize it with the right library versions.

**What a colleague on another machine must obtain separately:** a run ID or a
local `models:/` URI is useless to them. They need (1) the SQLite database **and**
the artifact directory — or access to a shared tracking server that holds both,
(2) the code at commit `91da09b`, (3) a compatible environment (`uv sync
--locked`: MLflow 3.18, scikit-learn 1.9.1, pandas 3.1.0, SQLAlchemy 2.0.58 …),
and (4) the original CSV (`62ca518b…`) if they want the same splits or a refit.
`http://127.0.0.1:5000` only works on this machine.

## 7. Short answers to the study questions

| Question | Answer |
|---|---|
| Experiment vs run | Experiment = named container for a line of work; run = one execution with its own id, params, metrics, tags, artifacts, dataset inputs and logged models. |
| Is the browser executing the training code? | No. The UI only queries the tracking server, which reads `mlflow.db` and the artifact store. |
| Metric vs artifact | A metric is a numeric series stored in the backend store (used for sorting/comparing); an artifact is a file stored in the artifact store (used by downloading it). |
| Does a seed guarantee identical results? | No. It fixes the pseudo-random draws, but library versions, BLAS/threading, platform and data ordering can still change the numbers. Necessary, not sufficient. |
| Can the training data be recovered from its hash? | No. SHA-256 is a one-way fingerprint of the bytes; you still need the file (or a copy matching the hash). |
| What if you log a commit while training from uncommitted edits? | The SHA describes the wrong code. Traceability requires recording that the worktree was dirty (and ideally the diff). This project logs `git_worktree_dirty` and the uncommitted paths; the final runs were produced after committing, so the flag is `false`. |
| What distinguishes two allocations from the same CSV? | The per-split `customer_id_sha256` fingerprints (plus sizes, positive rates and the split protocol/seed) in `provenance.json`; the CSV hash alone cannot distinguish them. |

## 8. Files produced

| Path | Content |
|---|---|
| `src/telco_churn/tracking.py` | tracking, provenance, evaluation and registry helpers (TP02 pipeline reused unchanged) |
| `scripts/train_mlflow.py` | one run per candidate, comparison export, selection, registration, reference batch |
| `scripts/predict_mlflow.py` | fresh-process reload and agreement check |
| `scripts/tp03_smoke_run.py` | section-1 connectivity check |
| `scripts/start_tracking_server.ps1`, `scripts/stop_tracking_server.ps1` | background server control |
| `reports/tp03_comparison.csv`, `reports/tp03_selection.json`, `reports/tp03_reference_predictions.json`, `reports/tp03_inference_check.json` | machine-readable evidence |
| `evidence/tp03/*.png` | UI screenshots and the comparison chart |
| `tests/test_tp03.py` | protocol, provenance and registry-contract tests (no server required) |
