# ST5230 Group 6 Project

## Project title
**Validity and Stability of LLM-as-a-Judge in Literary Translation Comparison**

## Project goal
This project studies whether LLM judges can reliably compare two English translations of the same literary source paragraph.

We focus on two questions:

- **RQ1 (Validity):** Does the judge agree with the **human-preferred direction**?
- **RQ2 (Stability):** Are judgments robust to **answer-order swaps** and **prompt wording changes** (`main` vs `minimal`)?

We also run a **comment analysis** sub-experiment to compare what humans and LLMs focus on when explaining translation preferences.

---

## 1. Repository structure

```text
ST5230-Group-6-Project/
├─ config/                       # model config files (.yaml)
├─ prompts/                      # prompt templates used in the experiments
├─ scripts/                      # all runnable Python scripts
├─ data/
│  ├─ raw/                       # raw files (original human annotation csv / original PAR3 resources)
│  ├─ processed/                 # cleaned csv files for experiments
│  └─ experiments/               # jsonl files used as direct experiment inputs
├─ results/
│  ├─ raw/                       # raw jsonl outputs returned by models
│  ├─ metrics/                   # computed metrics (validity / swap / prompt)
│  └─ analysis/                  # merged comment tables and downstream analysis outputs
├─ docs/                         # optional reports / proposal / slide assets
├─ README.md
└─ requirements.txt
```

If your local project is not fully organized yet, the file roles below still apply.

---

## 2. Important files and what they do

### 2.1 Data files

#### Raw / original data
- `human_annot_par3.csv`  
  Original human annotation file from PAR3.

#### Cleaned data
- `par3_clean_hum_gt.csv`  
  Cleaned `hum_gt` subset.
- `par3_clean_hum_gt_highconf.csv`  
  High-confidence `hum_gt` subset used as the main reference dataset.
- `par3_flagged_conflicts.csv`  
  Rows with `choice != gold` or other issues, kept for inspection.

#### Experiment inputs
- `main_experiment.jsonl`  
  Main experiment input set.
- `stability_subset.jsonl`  
  Fixed 45-item subset used for swap and prompt stability experiments.
- `comment_subset_*.jsonl`  
  Model-specific diagnostic subsets used for comment analysis.

---

### 2.2 Prompt files

- `prompts/judge_main.txt`  
  Main judge prompt for pairwise comparison.
- `prompts/judge_minimal.txt`  
  Minimal prompt used to test prompt stability.
- `prompts/judge_comment_free.txt`  
  Free-comment prompt: returns `winner + comment` without explicitly forcing evaluation dimensions.

---

### 2.3 Model config files

Single-model config files are used to run each judge separately:

- `config/models_deepseek_only.yaml`
- `config/models_gemini_only.yaml`
- `config/models_grok_only.yaml`
- `config/models_claude_only.yaml`
- `config/models_gpt54_only.yaml`

Comment-analysis config files:

- `config/models_comment_deepseek_only.yaml`
- `config/models_comment_gemini_only.yaml`
- `config/models_comment_grok_only.yaml`
- `config/models_comment_claude_only.yaml`
- `config/models_comment_gpt54_only.yaml`

---

### 2.4 Core scripts

#### Data preparation
- `scripts/01_build_main_experiment_input.py`  
  Builds `main_experiment.jsonl` and `stability_subset.jsonl` from cleaned csv.

#### Main experiment runs
- `scripts/02_run_openrouter_judges.py`  
  Calls OpenRouter and runs pairwise judging experiments.

#### Basic summary metrics
- `scripts/03_compute_metrics.py`  
  Computes main metrics from raw outputs. Useful for basic summaries.

#### Swap stability
- `scripts/04_compute_swap_consistency_from_main_and_swap.py`  
  Computes swap consistency by matching `main` and `swap` results.

#### Comment subset construction
- `scripts/05_build_comment_subset.py`  
  Builds a shared diagnostic subset.
- `scripts/05b_build_model_specific_comment_subset.py`  
  Builds **model-specific** diagnostic subsets (recommended).

#### Comment generation
- `scripts/06_run_openrouter_comment_analysis.py`  
  Runs the free-comment analysis and returns `winner + comment`.

#### Prompt stability
- `scripts/07_compute_prompt_consistency_from_main_and_minimal.py`  
  Computes consistency between `main` and `minimal` prompt results.

#### Comment merging / post-processing
- `scripts/08_merge_human_and_llm_comments.py`  
  Merges human comments with DeepSeek / Gemini comment outputs.
- `scripts/08_merge_human_and_llm_comments_multimodel.py`  
  Multi-model version; merges comments from all judge models.

---

## 3. Environment setup

### 3.1 Install dependencies
```powershell
pip install -r requirements.txt
```

### 3.2 Set OpenRouter API key
```powershell
$env:OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
```

---

## 4. How to run the project

## Step 1. Build experiment inputs
```powershell
python scripts/01_build_main_experiment_input.py `
  --input_csv data/processed/par3_clean_hum_gt_highconf.csv `
  --main_output data/experiments/main_experiment.jsonl `
  --stability_output data/experiments/stability_subset.jsonl `
  --stability_n 45 `
  --seed 42 `
  --stratify_by lang_code
```

---

## Step 2. Run the main experiment
Example: DeepSeek
```powershell
python scripts/02_run_openrouter_judges.py `
  --input_jsonl data/experiments/main_experiment.jsonl `
  --models config/models_deepseek_only.yaml `
  --prompt_file prompts/judge_main.txt `
  --prompt_version main `
  --output_jsonl results/raw/main_deepseek_main.jsonl `
  --sleep 2
```

Repeat by replacing the model config and output file for:
- Gemini
- Grok
- Claude
- GPT-5.4

---

## Step 3. Compute validity (RQ1)
Example: Grok
```powershell
python scripts/03_compute_metrics.py `
  --main_results results/raw/main_grok_main.jsonl `
  --output_dir results/metrics/grok_main
```

---

## Step 4. Run swap stability
Example: Gemini
```powershell
python scripts/02_run_openrouter_judges.py `
  --input_jsonl data/experiments/stability_subset.jsonl `
  --models config/models_gemini_only.yaml `
  --prompt_file prompts/judge_main.txt `
  --prompt_version main `
  --output_jsonl results/raw/stability_gemini_swap.jsonl `
  --swap_order `
  --sleep 2
```

Then compute swap consistency:
```powershell
python scripts/04_compute_swap_consistency_from_main_and_swap.py `
  --main_results results/raw/main_gemini_main.jsonl `
  --swap_results results/raw/stability_gemini_swap.jsonl `
  --output_dir results/metrics/gemini_swap_manual
```

---

## Step 5. Run prompt stability (`main` vs `minimal`)
Example: Claude
```powershell
python scripts/02_run_openrouter_judges.py `
  --input_jsonl data/experiments/stability_subset.jsonl `
  --models config/models_claude_only.yaml `
  --prompt_file prompts/judge_minimal.txt `
  --prompt_version minimal `
  --output_jsonl results/raw/stability_claude_minimal.jsonl `
  --sleep 2
```

Then compute prompt consistency:
```powershell
python scripts/07_compute_prompt_consistency_from_main_and_minimal.py `
  --main_results results/raw/main_claude_main.jsonl `
  --minimal_results results/raw/stability_claude_minimal.jsonl `
  --output_dir results/metrics/claude_prompt_manual
```

**Important note for Claude:** current main/swap/prompt effective `n` are lower than the designed 99/45/45 because some outputs failed to parse or did not match across conditions. If needed, rerun Claude on the 45-item stability subset for `main` and `swap` to improve matching coverage.

---

## Step 6. Build model-specific diagnostic subsets for comment analysis
Example: GPT-5.4
```powershell
python scripts/05b_build_model_specific_comment_subset.py `
  --experiment_items data/experiments/main_experiment.jsonl `
  --main_results results/raw/main_gpt54_main.jsonl `
  --swap_results results/raw/stability_gpt54_swap.jsonl `
  --output_jsonl data/experiments/comment_subset_gpt54.jsonl `
  --model_name openai/gpt-5.4-pro `
  --n_incorrect 12 `
  --n_unstable 8 `
  --n_control 6 `
  --seed 42
```

Recommended diagnostic subset size for each model:
- incorrect: up to **12**
- unstable: up to **8**
- control: up to **6**

This keeps comment-analysis subsets comparable across models.

---

## Step 7. Run free-comment analysis
Example: Grok
```powershell
python scripts/06_run_openrouter_comment_analysis.py `
  --input_jsonl data/experiments/comment_subset_grok.jsonl `
  --models config/models_comment_grok_only.yaml `
  --prompt_file prompts/judge_comment_free.txt `
  --prompt_version comment_free `
  --output_jsonl results/raw/comment_analysis_grok_self.jsonl `
  --sleep 2
```

---

## Step 8. Merge human comments and LLM comments
Multi-model version:
```powershell
python scripts/08_merge_human_and_llm_comments_multimodel.py `
  --main_experiment data/experiments/main_experiment.jsonl `
  --human_csv data/processed/par3_clean_hum_gt_highconf.csv `
  --comment_files `
    results/raw/comment_analysis_deepseek_self.jsonl `
    results/raw/comment_analysis_gemini_self.jsonl `
    results/raw/comment_analysis_grok_self.jsonl `
    results/raw/comment_analysis_claude_self.jsonl `
    results/raw/comment_analysis_gpt54_self.jsonl `
  --output_csv results/analysis/comment_comparison_merged_all_models.csv
```

---

## 5. Main project outcomes

## 5.1 RQ1: Validity vs human preference
Human reference signal: **high-confidence `hum_gt` subset** from PAR3.

### Overall validity results
| Model | Validity (%) | 95% CI | Main n |
|---|---:|---|---:|
| DeepSeek | 89.9 | 83.8–96.0 | 99 |
| Gemini | 87.9 | 80.8–93.9 | 99 |
| Grok | 94.9 | 89.9–99.0 | 99 |
| Claude | 92.0 | 86.4–96.6 | 88 |
| GPT-5.4 | 88.9 | 82.8–94.9 | 99 |

**Interpretation:** All tested models perform far above the random baseline on this task. Grok currently has the highest observed agreement with the human-preferred direction, while DeepSeek, Gemini, and GPT-5.4 also show strong validity.

---

## 5.2 RQ2: Stability under perturbations
### Swap stability (`main` vs answer-order swap)
| Model | Swap Stability (%) | 95% CI | Effective n |
|---|---:|---|---:|
| DeepSeek | 77.8 | 64.4–88.9 | 45 |
| Gemini | 91.1 | 82.2–97.8 | 45 |
| Grok | 95.6 | 88.9–100.0 | 45 |
| Claude | 89.2 | 78.4–97.3 | 37 |
| GPT-5.4 | 88.9 | 77.8–97.8 | 45 |

### Prompt stability (`main` vs `minimal`)
| Model | Prompt Stability (%) | 95% CI | Effective n |
|---|---:|---|---:|
| DeepSeek | 97.8 | 93.3–100.0 | 45 |
| Gemini | 93.3 | 84.4–100.0 | 45 |
| Grok | 97.8 | 93.3–100.0 | 45 |
| Claude | 100.0 | 100.0–100.0 | 40 |
| GPT-5.4 | 97.8 | 93.3–100.0 | 45 |

**Interpretation:** Prompt stability is generally very high across models. Swap stability is more discriminative: DeepSeek is more sensitive to answer order, while Gemini, Grok, Claude, and GPT-5.4 are more stable under answer-order perturbation.

---

## 5.3 Comment analysis outcomes
We compared **human comments** and **LLM self-comments** across the following 8 dimensions:

1. Faithfulness / adequacy  
2. Fluency / naturalness  
3. Style / tone  
4. Literalness  
5. Mistranslation / meaning error  
6. Word choice / expression  
7. Awkwardness / grammar  
8. Context sensitivity

### Main findings so far
- Human comments strongly emphasize **word choice / expression**, **literalness**, **awkwardness / grammar**, and **faithfulness**.
- DeepSeek comments frequently emphasize **fluency**, **style**, and **word choice**, but rarely mention **mistranslation / meaning error**.
- Gemini comments also mention **fluency**, but comparatively attend more than DeepSeek to **mistranslation / meaning error**.
- Grok comments show strong emphasis on **word choice / expression** and **fluency**.
- Claude and GPT-5.4 comments have been collected and merged into the multi-model comment table, enabling direct comparison across all judges.

---

## 6. Key generated outputs

### Comment-analysis merged tables
- `comment_comparison_merged.csv`
- `comment_comparison_merged_all_models.csv`
- `updated_comment_flags_final/comment_comparison_merged_all_models_with_category_flags.xlsx`
