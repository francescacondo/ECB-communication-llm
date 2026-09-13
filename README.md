# Extracting Economic Signals from ECB Communication with Large Language Models

An LLM-based measurement pipeline for European Central Bank speeches,
together with the validation evidence needed before its output can be
used as an economic variable.

Six signals are extracted from each speech under a fixed coding scheme:
binary attention indicators for inflation, real activity, financial
stability and uncertainty, and ternary directional outlook variables for
inflation and growth. The corpus is the ECB's official precompiled
speech dataset, 2,770 speeches from January 1999 to December 2025.

The write-up is `measurement_note.tex`. It treats the extraction as an
instrument and asks what has to be established before its output is
usable: whether it reproduces human coding, how it fails when it fails,
whether the failure can be corrected, whether the resulting series
relate to euro-area macro data as economic reasoning predicts, and what
the language model adds over a keyword dictionary.

## Compiling the note

    pdflatex measurement_note.tex

Run it three times to settle the table of contents and cross-references.
The figures it needs are committed in `figures/`, so no analysis has to
be run first. `measurement_note.pdf` is the compiled result.

Requires `multirow` and `listings` beyond a basic TeX distribution.

## The coding scheme

`docs/coding_rules.md` is the codebook, at **version 2**.
`prompts/extraction_prompt.txt` is the classification prompt at
**version 1**, which is what the committed extraction ran under; it is
preserved byte-for-byte and must not be edited.
`prompts/extraction_prompt_v2.txt` mirrors the revised codebook for a
future extraction. Both prompts are reproduced in full in the note's
appendices, and `docs/adjudication_protocol.md` records how coder
disagreements were resolved into rules.

The model codes therefore predate the six decision rules added at
version 2. Wherever the model is scored against human codes made under
version 2, it is being held to a specification it was not given, and the
note says so at each point.

## Running the pipeline

    python src/download_data.py            # ECB speeches
    python src/audit_data.py               # corpus audit
    python src/clean_text.py               # baseline sample
    python src/prepare_full_sample.py      # extraction excerpts
    python src/extract_signals_openai.py   # LLM extraction (costly; do not re-run)
    python src/validate_extractions.py     # completeness checks
    python analysis/10_panel.py            # speech-level panel
    python analysis/15_validation_stats.py # round-one audit, error rates
    python analysis/12_measures.py         # aggregated measures
    python src/download_macro.py           # euro-area macro series
    python src/download_salience_series.py # CISS and SPF dispersion
    python analysis/13_external_validation.py
    python analysis/16_dictionary_benchmark.py

Validation rounds, drawn once each with `ACTIVE_ROUND` set to the round
being drawn:

    python src/select_validation_sample.py # seeded blind sample and key
    python src/build_coding_sheet.py       # blind sheet and speech texts
    python analysis/17_validation_round2.py
    python analysis/18_coder_reliability.py
    python src/diff_prompts.py             # version 1 to version 2 rule diff
    python analysis/19_round4_ceiling.py
    python analysis/25_salience_validation.py

Order matters in two places. `analysis/12_measures.py` reads the error
rates written by `analysis/15_validation_stats.py` and fails without
them, and `analysis/25_salience_validation.py` consumes output from both
`13_external_validation.py` and `19_round4_ceiling.py`.

Table 16 of the note lists the same order with the tables and figures
each stage produces.

## What is committed and what is not

Committed: the note and its figures, the coding materials, the analysis
code, the human coding in `data/human/`, the seeded validation-sample
keys, and every derived result in `outputs/`.

Not committed: the ECB speech text, in any form. That means the raw
corpus, the extraction excerpts, the model output, and the audit
coding sheets, which carry speech text alongside the codes. The corpus
is downloaded programmatically by `src/download_data.py`; the
extraction is regenerated only by re-running the model, which costs
money and is not something to do incidentally.

Consequently the analysis stages will not run end to end from a fresh
clone. The results they produce are committed instead, so every number
in the note can be checked against `outputs/` without re-running
anything. The four supporting modules — `src/measurement_error.py`,
`src/agreement.py`, `src/timeseries_stats.py` and
`src/dictionary_signals.py` — hold the reusable statistical logic and
are readable on their own.

## Dependencies

    pip install -r requirements.txt

pandas, numpy and matplotlib. The statistical procedures are implemented
in `src/`, so neither scipy nor statsmodels is required.
