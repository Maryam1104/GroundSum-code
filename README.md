# GroundSum: A Grounded Extraction–Abstraction Framework for Factually Consistent Long-Document Summarization

## Abstract

Abstractive summarization of long scientific documents, such as ArXiv
preprints and PubMed biomedical articles remains a challenging open problem
in Natural Language Processing. State-of-the-art sequence-to-sequence models
frequently produce summaries that are fluent yet factually inconsistent with
their source documents, a phenomenon known as *hallucination*. This repository
implements a complete, reproducible research pipeline that directly attacks
this problem through a two-stage hybrid framework:

1. **Extractive grounding**: a pretrained Longformer encoder processes
   scientific documents of up to 4,096 tokens. A learnable global query
   attends over the Longformer token representations through multi-head
   cross-attention to aggregate document-level context. This context is
   incorporated into the token representations through a residual
   connection, followed by binary token-relevance classification.
   Token-level predictions are aggregated into sentence importance
   scores to select salient sentences for an extractive summary.
   A DistilBERT sentence-level classifier serves as a comparative baseline.
2. **Controlled abstractive refinement**: a four-stage GPT self-critique loop
   (fact extraction → guided rewrite → self-critique → automated revision)
   rewrites each extractive chunk into fluent academic prose while being
   explicitly anchored to extracted (subject, predicate, object) fact triples,
   suppressing the introduction of unsupported content. A T5 seq-to-seq model
   serves as a second abstractive baseline for comparison.

Factual consistency is evaluated quantitatively via BERTScore semantic
similarity and a GPT-based FactCC probe (SUPPORTS / NOT\_SUPPORTS with a
continuous confidence score), supplemented by standard ROUGE-1/2/L metrics for
coverage.

## Project Highlights

- Novel hybrid extractive–abstractive summarization framework
- Longformer-based extraction with learnable global-query cross-attention, residual context fusion, and token-level relevance classification
- GPT-4o-mini self-critique rewriting pipeline for hallucination mitigation
- DistilBERT and T5 baselines for comparison
- Evaluated on ArXiv and PubMed scientific summarization benchmarks
- Automated evaluation using ROUGE, BERTScore, and GPT-based FactCC

## Overall System Architecture

The proposed framework consists of a two-stage summarization pipeline. A custom Longformer-based extractive model first identifies salient evidence sentences from long scientific documents. These extracted summaries are subsequently refined by a GPT-based self-critique rewriting pipeline to improve fluency while preserving factual consistency. DistilBERT and T5 are implemented as baseline models for comparative evaluation.

<p align="center">
  <img src="images/architecture_approach.png" width="650">
</p>

<p align="center">
  <em>Figure 1. Overall architecture of the proposed framework.</em>
</p>


### Token-Level Oracle Supervision

The extraction model is trained using TF-IDF-selected oracle sentences from the source documents. Training labels are
constructed by aligning oracle sentence spans with the Longformer tokenizer's character-level offset mappings.

All valid tokens overlapping an oracle sentence are assigned label `1`, while other valid document tokens receive label `0`.
Special tokens and padding positions receive label `-100` and are ignored during cross-entropy loss computation.

The same label-generation procedure is applied to the training and validation splits, ensuring consistent supervision.
Only tokens within the maximum 4,096-token input window participate in training.
     
---

## Repository Structure

```
hallucination_mitigation_nlp/
├── main.py                  ← Single entry point; orchestrates the complete pipeline
├── requirements.txt         ← Python dependencies
├── README.md                ← Project documentation
├── images/                  ← Architecture and workflow figures used in the README
│   ├── architecture_approach.png
│   └── abstractive_pipeline.png
└── src/
    ├── __init__.py
    ├── utils.py             ← Configuration, logging, path helpers, and I/O utilities
    ├── data_processing.py   ← Dataset downloading, preprocessing, TF-IDF oracle generation, and BERTScore evaluation
    ├── models.py            ← Longformer, DistilBERT, T5, GPT self-critique pipeline, and training workflows
    └── evaluation.py        ← ROUGE, BERTScore, FactCC evaluation frameworks
```

### `src/` Module Responsibilities

- **`src/utils.py`**
  - Central configuration constants (`TRAIN_SIZE`, `VAL_SIZE`, `TEST_SIZE`,
    `GPT_MODEL`, `CHUNK_TOKEN_SIZE`, etc.).
  - `build_paths(base_dir)`: constructs and creates all project directories
    from a single root path, making the codebase environment-agnostic.
  - `setup_logging(log_dir)`: configures a dual file+stdout logger.
  - `chunk_by_tokens(text)`: tiktoken-based sentence-boundary-aware text
    chunker used by the GPT rewrite pipeline.
  - `safe_batch_decode(preds, tokenizer)`: robust conversion from logit or
    token-ID arrays to decoded strings, handling 1D/2D/3D inputs and
    vocabulary clamping.
  - General I/O helpers: `save_json`, `save_jsonl`, `load_jsonl`,
    `inspect_jsonl`.

- **`src/data_processing.py`**
  - `download_datasets(dataset_path)`: downloads ArXiv and PubMed JSONL
    splits from the Hugging Face Hub via `wget`.
  - `load_custom_dataset(dataset_name, dataset_path)`: loads raw JSONL data
    into a typed HuggingFace `DatasetDict` with a fixed feature schema.
  - `tfidf_oracle(text_sentences, top_k)`: TF-IDF sentence scoring and
    top-*k* selection for oracle extractive summary generation.
  - `generate_tfidf_oracle_dataset(dataset_dict)`: in-memory oracle
    annotation of all splits; returns an enriched `DatasetDict`.
  - `generate_and_save_trimmed_split(...)`: production-scale enriched JSONL
    serialisation with configurable sample limits.
  - `build_enriched_datasets(dataset_path)`: end-to-end orchestrator for
    both ArXiv and PubMed enriched dataset creation.
  - `load_cleaned_enriched_split(jsonl_path)`: drops auxiliary schema fields
    that cause type conflicts before BERTScore evaluation.
  - `bertscore_pairwise(oracle, gold)`: per-example BERTScore computation.
  - `evaluate_bert_similarity(dataset_dict, split_name)`: batch BERTScore
    oracle-vs-gold evaluation.
  - `run_bertscore_evaluation(dataset_path)`: full evaluation across all
    splits of both datasets.

### `src/` Module Responsibilities

- **`src/utils.py`**
  - Defines shared configuration constants, including dataset sizes, summary length, model settings, and GPT parameters.
  - `build_paths(base_dir)`: creates project directories using a configurable root path.
  - `setup_logging(log_dir)`: configures file and console logging.
  - `chunk_by_tokens(text)`: splits text into token-bounded chunks for GPT-based rewriting.
  - `safe_batch_decode(preds, tokenizer)`: decodes model predictions into text.
  - Provides JSON/JSONL input-output helpers and dataset inspection utilities.

- **`src/data_processing.py`**
  - `download_datasets(dataset_path)`: downloads arXiv and PubMed JSONL datasets from Hugging Face.
  - `load_custom_dataset(dataset_name, dataset_path)`: loads raw training, validation, and test splits into a Hugging Face `DatasetDict`, preserving split-specific fields.
  - `tfidf_oracle(text_sentences, top_k)`: selects informative source sentences using TF-IDF scoring.
  - `generate_tfidf_oracle_dataset(dataset_dict)`: generates extractive oracle summaries for all dataset splits.
  - `generate_and_save_trimmed_split(...)`: saves oracle-enriched datasets as JSONL files.
  - `build_enriched_datasets(dataset_path)`: prepares arXiv and PubMed enriched datasets.
  - `load_cleaned_enriched_split(jsonl_path)`: prepares enriched data for semantic similarity evaluation.
  - `bertscore_pairwise(oracle, gold)`: calculates BERTScore between oracle and reference summaries.
  - `evaluate_bert_similarity(dataset_dict, split_name)`: evaluates oracle-summary similarity.
  - `run_bertscore_evaluation(dataset_path)`: evaluates oracle quality across both datasets.

- **`src/models.py`**
  - `call_openai_chat(messages, ...)`: OpenAI Chat Completions wrapper with exponential back-off retries.
  - `rewrite_with_self_critique(chunk)`: four-stage GPT-based refinement comprising fact extraction, guided rewriting, self-critique, and automated revision.
  - `run_rewrite_pipeline(base_dir, datasets=None)`: applies fact-anchored abstractive rewriting to extractive summaries for selected datasets.
  - `LongformerWithCrossAttention`: custom PyTorch model combining a pretrained Longformer encoder, learnable global query, multi-head cross-attention over token representations, residual context fusion, and binary token-level relevance classification.
  - `SavePretrainedCallback`: saves custom model weights during Hugging Face Trainer checkpointing.
  - `SummaryGenerator`: computes token relevance probabilities, aligns tokens with source sentences through tokenizer offset mappings, performs mean pooling, and selects the highest-scoring sentences in their original document order.
  - `LongformerExtractor`: manages model initialization, token-level oracle supervision, training, validation, checkpoint resumption, ROUGE evaluation, and model serialization.
  - `_create_token_labels(...)`: assigns binary relevance labels using oracle sentence spans and tokenizer offset mappings. Padding and special tokens receive `-100`.
  - `build_sentence_dataset(ds_dict)`: transforms document-level examples into sentence-level classification examples for DistilBERT.
  - `get_distilbert_trainer(...)`: configures DistilBERT training and ROC-AUC evaluation.
  - `train_distilbert_baselines(arxiv_ds, pubmed_ds, output_dir)`: trains DistilBERT baselines for arXiv and PubMed.
  - `load_and_split_extractive_preds(input_file)`: prepares training, validation, and test subsets from extractive prediction files for T5.
  - `preprocess_for_t5(datasets, tokenizer, ...)`: tokenizes extractive predictions and reference summaries for sequence-to-sequence training.
  - `run_t5_abstractive_pipeline(...)`: performs resource-constrained T5 fine-tuning and evaluation using BERTScore and GPT-based factual-consistency scoring.

- **`src/evaluation.py`**
  - `evaluate_summaries(refs, gens)`: calculates ROUGE-1, ROUGE-2, and ROUGE-L scores.
  - `evaluate_model(dataset_name, model_fn, ...)`: evaluates extractive summaries against TF-IDF oracle summaries.
  - `evaluate_extractive_distilbert(trainer, tokenizer, raw_ds, ...)`: performs DistilBERT sentence ranking, top-*k* extraction, ROUGE evaluation, and prediction saving.
  - `compute_bertscore(refs, hyps)`: calculates corpus-level BERTScore precision, recall, and F1.
  - `compute_factcc_via_gpt(refs, hyps)`: performs GPT-based factual-consistency assessment using SUPPORTS/NOT_SUPPORTS predictions and confidence scores.
  - `load_reference_summaries(path)`: loads human-written reference summaries from JSONL files.
  - `load_generated_summaries(dirpath)`: loads numbered generated-summary files.
  - `load_and_preprocess_test_dataset(dataset_name, drive_path)`: loads and normalizes enriched test examples.
  - `evaluate_all(base_dir, datasets)`: aligns reference and generated summaries by document ID, computes BERTScore and GPT-based factual-consistency metrics, and saves evaluation results.

---

## Dataset Provenance

GroundSum uses long-document scientific summarization datasets covering arXiv research papers and PubMed biomedical articles.

The datasets are retrieved from the following Hugging Face repositories:

- **arXiv:** [nianlong/long-doc-extractive-summarization-arxiv](https://huggingface.co/datasets/nianlong/long-doc-extractive-summarization-arxiv)
- **PubMed:** [nianlong/long-doc-extractive-summarization-pubmed](https://huggingface.co/datasets/nianlong/long-doc-extractive-summarization-pubmed)

The `src/data_processing.py` module downloads the original JSONL splits and generates TF-IDF-based extractive oracle summaries. The enriched datasets retain the source documents, human-written summaries, and oracle annotations.

The configured maximum dataset sizes are:

| Split | Documents per dataset |
|---|---:|
| Training | 20,000 |
| Validation | 2,500 |
| Test | 2,500 |

Actual counts may be lower if fewer records are available or examples are removed during filtering.

The enriched datasets are saved under `dataset/arxiv_enriched/` and `dataset/pubmed_enriched/`.

---

## Setup and Installation

### Prerequisites

- Python 3.10 or later.
- A CUDA-capable GPU is strongly recommended for Longformer training. Memory requirements depend on sequence length, batch size, precision, and hardware configuration.
- Internet access for downloading datasets, pretrained models, and required dependencies.
- An [OpenAI API key](https://platform.openai.com/api-keys) for GPT-based rewriting and factual-consistency evaluation.

### 1. Clone the repository

```bash
git clone https://github.com/Maryam1104/GroundSum-code.git
cd GroundSum-code
```

### 2. Create and activate a virtual environment

Create the environment:

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Set your OpenAI API key

Linux / macOS:

```bash
export OPENAI_API_KEY="your-api-key"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Do not commit API keys or `.env` files to GitHub.

### 5. Download NLTK resources

```bash
python -m nltk.downloader punkt punkt_tab
```

### 6. Verify the installation

```bash
python -m compileall -q main.py src
python main.py --help
```

---

## Running the Pipeline

All commands should be executed from the repository root.

The `--base_dir` argument specifies the directory in which datasets, model checkpoints, generated summaries, and evaluation results will be stored.

### Full end-to-end pipeline

```bash
python main.py --base_dir ./project_data --stages all --datasets arxiv_enriched,pubmed_enriched
```

The full pipeline downloads datasets, creates oracle annotations, trains the extractive models and baselines, generates summaries, performs GPT-based rewriting, and evaluates the outputs.

Running all stages requires substantial computation and an OpenAI API key. For controlled experiments, stages can be executed individually.

### Run individual stages

**1. Download raw datasets**

```bash
python main.py --base_dir ./project_data --stages download
```

**2. Generate TF-IDF enriched datasets**

```bash
python main.py --base_dir ./project_data --stages build_datasets
```

**3. Evaluate TF-IDF oracle quality**

```bash
python main.py --base_dir ./project_data --stages bertscore_eval
```

**4. Train GroundSum's Longformer extractor on arXiv**

```bash
python main.py --base_dir ./project_data --stages longformer --datasets arxiv_enriched
```

**5. Train GroundSum's Longformer extractor on PubMed**

```bash
python main.py --base_dir ./project_data --stages longformer --datasets pubmed_enriched
```

The Longformer training pipeline uses:

- Pretrained `allenai/longformer-base-4096`.
- Maximum input sequence length of 4,096 tokens.
- Learnable global-query multi-head cross-attention.
- Residual fusion of document-level and token-level representations.
- Binary token-level classification with oracle-aligned labels.
- Sentence-level mean pooling and top-five extraction.
- Batch size 1 and three training epochs.
- Checkpoint saving and resumption.

**6. Train DistilBERT sentence-level baselines**

```bash
python main.py --base_dir ./project_data --stages baseline_distilbert
```

This stage trains and evaluates separate classifiers for arXiv and PubMed.

**7. Fine-tune the T5 abstractive baseline**

For arXiv:

```bash
python main.py --base_dir ./project_data --stages baseline_t5 --datasets arxiv_enriched --t5_model t5-base --t5_max_input_length 32 --t5_max_target_length 32 --t5_epochs 1
```

For PubMed:

```bash
python main.py --base_dir ./project_data --stages baseline_t5 --datasets pubmed_enriched --t5_model t5-base --t5_max_input_length 32 --t5_max_target_length 32 --t5_epochs 1
```

The T5 baseline uses resource-constrained training with a maximum input length of 32 tokens, a maximum target length of 32 tokens, batch size 1, and one training epoch.

**8. Run GPT-based abstractive rewriting**

```bash
python main.py --base_dir ./project_data --stages rewrite --datasets arxiv_enriched,pubmed_enriched
```

This stage applies fact extraction, guided rewriting, self-critique, and automated revision to the Longformer-generated extractive summaries.

**9. Evaluate rewritten summaries**

```bash
python main.py --base_dir ./project_data --stages evaluate --datasets arxiv_enriched,pubmed_enriched
```

The evaluation includes BERTScore and GPT-based factual-consistency scoring.

**10. Inspect prediction files**

```bash
python main.py --base_dir ./project_data --stages inspect --datasets arxiv_enriched,pubmed_enriched
```

### Combining stages

To train GroundSum, generate extractive summaries, perform abstractive rewriting, and evaluate both datasets:

```bash
python main.py --base_dir ./project_data --stages longformer,rewrite,evaluate --datasets arxiv_enriched,pubmed_enriched
```

The datasets must already have been downloaded and enriched before executing these stages.

### Experimental Reproducibility

The main pipeline accepts a `--seed` argument, with a default value of `42`.

For reproducible experiments, use consistent dataset splits, random seeds, dependency versions, model configurations, and evaluation procedures.

The T5 baseline uses a separate training/validation/test partition derived from DistilBERT prediction records. Its evaluation split should be distinguished from the original full GroundSum test set when reporting comparative results.

GPU hardware, numerical precision, library versions, and external GPT API responses may affect exact reproducibility.

---

## Key Outputs

After pipeline execution, generated files are stored under the configured `<base_dir>`.

| Path | Contents |
|---|---|
| `dataset/arxiv_enriched/{train,val,test}.jsonl` | TF-IDF oracle-annotated arXiv dataset splits |
| `dataset/pubmed_enriched/{train,val,test}.jsonl` | TF-IDF oracle-annotated PubMed dataset splits |
| `models/<dataset>/final_model/pytorch_model.bin` | Final saved Longformer model weights |
| `checkpoints/<dataset>/checkpoint-*/` | Longformer training checkpoints |
| `summaries/<dataset>/<idx>.txt` | Longformer extractive summaries |
| `abs_summaries/<dataset>/test_summaries/<idx>.txt` | GPT-refined abstractive summaries |
| `abs_summaries/<dataset>/evaluation_results.json` | BERTScore and GPT-based factual-consistency results |
| `baselines/distilbert/<raw_dataset>/test_predictions.jsonl` | DistilBERT extractive predictions and reference summaries |
| `baselines/distilbert/<raw_dataset>/test_rouge.json` | DistilBERT ROUGE scores |
| `abstractive_t5/<raw_dataset>/metrics/test_metrics.json` | T5 BERTScore and GPT-based factual-consistency results |
| `abstractive_t5/<raw_dataset>/summaries/test_abstractive.jsonl` | T5-generated summaries and references |
| `results/<dataset>_results.csv` | GroundSum extractive ROUGE evaluation results |
| `logs/main_pipeline.log` | Main pipeline execution log |

Here, `<dataset>` denotes `arxiv_enriched` or `pubmed_enriched`, while `<raw_dataset>` denotes `arxiv` or `pubmed`.

---

## Citation

This research is currently under review. Citation details will be updated following publication.

---

## License

This project is intended for academic and research use. Refer to the repository's license file, if provided, for applicable permissions and conditions.
