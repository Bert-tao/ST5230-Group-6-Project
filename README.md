# ST5230 Group 6 Project

Evaluating the validity of LLM-as-a-Judge for literary translation preference using the PAR3 human annotation subset.

## Research design

### Main experiment (validity)
We use the cleaned `hum_gt` human annotation subset as the reference signal. Each item contains:
- a source paragraph
- two candidate translations (`text1`, `text2`)
- a human preference label (`choice`)

The main outcome is whether an LLM judge selects the same preferred translation as the human annotation.

### Stability experiment
On a fixed subset of the main experiment items, we apply:
- answer-order swap
- prompt variation

This tests whether model judgments remain stable under superficial perturbations.

## Important note on stratification
The current cleaned annotation file does **not** contain a `book` identifier. Therefore, the executable pipeline in this repository currently supports:
- full-sample evaluation on all high-confidence `hum_gt` items
- optional stratification by `lang_code`

If book identifiers are later recovered by matching annotations back to `par3.pkl`, the same scripts can be extended to support book-level stratified analysis.

## Repository structure

```text
ST5230-Group-6-Project/
  config/
    models.yaml
  data/
    raw/
    processed/
    experiments/
  prompts/
    judge_main.txt
    judge_style_fidelity.txt
    judge_minimal.txt
  results/
    raw/
    parsed/
    metrics/
  scripts/
    01_build_main_experiment_input.py
    02_run_openrouter_judges.py
    03_compute_metrics.py
  requirements.txt
  .env.example
  .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy your cleaned file into:

```text
data/processed/par3_clean_hum_gt_highconf.csv
```

Then create a `.env` file from `.env.example` and set your OpenRouter API key.

## Step 1: Build experiment inputs

This creates:
- `main_experiment.jsonl`
- `stability_subset.jsonl`

```bash
python scripts/01_build_main_experiment_input.py \
  --input_csv data/processed/par3_clean_hum_gt_highconf.csv \
  --main_output data/experiments/main_experiment.jsonl \
  --stability_output data/experiments/stability_subset.jsonl \
  --stability_n 45 \
  --seed 42
```

## Step 2: Run judges through OpenRouter

Example: run the main experiment on all models using the main prompt.

```bash
python scripts/02_run_openrouter_judges.py \
  --input_jsonl data/experiments/main_experiment.jsonl \
  --models config/models.yaml \
  --prompt_file prompts/judge_main.txt \
  --prompt_version main \
  --output_jsonl results/raw/main_mainprompt.jsonl
```

Example: run the stability subset with answer-order swap.

```bash
python scripts/02_run_openrouter_judges.py \
  --input_jsonl data/experiments/stability_subset.jsonl \
  --models config/models.yaml \
  --prompt_file prompts/judge_main.txt \
  --prompt_version main \
  --swap_order \
  --output_jsonl results/raw/stability_swap_mainprompt.jsonl
```

Example: run a prompt variant on the stability subset.

```bash
python scripts/02_run_openrouter_judges.py \
  --input_jsonl data/experiments/stability_subset.jsonl \
  --models config/models.yaml \
  --prompt_file prompts/judge_style_fidelity.txt \
  --prompt_version style_fidelity \
  --output_jsonl results/raw/stability_prompt_stylefidelity.jsonl
```

## Step 3: Compute metrics

```bash
python scripts/03_compute_metrics.py \
  --main_results results/raw/main_mainprompt.jsonl \
  --stability_results results/raw/stability_swap_mainprompt.jsonl results/raw/stability_prompt_stylefidelity.jsonl \
  --output_dir results/metrics
```

## Statistical analysis used

### Main experiment
- agreement rate with human preference
- 95% bootstrap confidence interval
- one-sided binomial test against 0.5
- pairwise model comparison via bootstrap confidence intervals on agreement differences

### Stability experiment
- swap consistency rate
- prompt consistency rate
- 95% bootstrap confidence intervals
- optional Cohen's kappa for consistency between conditions

## Suggested reporting

### Main experiment
Report, for each model:
- agreement rate
- 95% CI
- p-value vs random baseline

### Stability experiment
Report, for each model:
- swap consistency rate
- prompt consistency rate
- reversal rate
- optional kappa

### Error analysis
Use a small diagnostic subset where you allow a short reason field. Recommended targets:
- incorrect cases
- large inter-model disagreement cases
- instability cases

