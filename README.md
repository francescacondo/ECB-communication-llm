# Extracting Economic Signals from ECB Communication with Large Language Models

This project builds and evaluates an LLM-based pipeline for extracting
structured economic signals from European Central Bank speeches. The
focus is on measurement: how text classifications can be constructed,
validated, and assessed before being used as variables in economic
analysis.

The corpus consists of 2,770 speeches from the ECB's official
precompiled speech dataset, covering January 1999 to December 2025. Six
variables are extracted from each speech under a fixed coding scheme:
binary attention indicators for inflation, real activity, financial
stability, and uncertainty, together with ternary directional outlook
measures for inflation and growth.

The project treats the LLM extraction as a measurement instrument rather
than as observed data. The validation therefore examines how closely the
classifications reproduce independent human coding, where and why
disagreements arise, how classification error could be incorporated into
aggregate measures, whether the resulting communication series behave
as expected against euro-area macroeconomic data, and what contextual
classification adds relative to a keyword-based benchmark.


## The coding scheme

`docs/coding_rules.md` is the codebook, at **version 2**.
`prompts/extraction_prompt.txt` is the classification prompt at
**version 1**, which is what the committed extraction ran under.
`prompts/extraction_prompt_v2.txt` mirrors the revised codebook for a
future extraction.


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
    python analysis/13_external_validation.py  # macro correlations
    python analysis/16_dictionary_benchmark.py # LLM vs keyword dictionary

Supplementary analyses:

    python analysis/21_alternative_inference.py # robustness of external validation
    python analysis/22_local_projections.py     # local projections
    python analysis/23_altavilla_factors.py     # monetary-policy surprise factors
    python analysis/24_event_window_factors.py  # event-window analysis
    python analysis/25_salience_validation.py   # salience-based validation

Validation rounds, drawn once each with `ACTIVE_ROUND` set to the
round being drawn (rounds 2, 3 and 4):

    python src/select_validation_sample.py # seeded blind sample and key
    python src/build_coding_sheet.py       # blind sheet and speech texts
    python analysis/17_validation_round2.py
    python analysis/18_coder_reliability.py
    python src/diff_prompts.py             # version 1 to version 2 rule diff
    python analysis/19_round4_ceiling.py

Order is important in:
`analysis/12_measures.py` reads the error 
rates written by `analysis/15_validation_stats.py` and fails without
them, and `analysis/25_salience_validation.py` consumes output from both
`13_external_validation.py` and `19_round4_ceiling.py`.

The accompanying note (Table 17) lists the same order with the tables
and figures each stage produces.


## Dependencies

    pip install -r requirements.txt

pandas, numpy and matplotlib. The statistical procedures are implemented
in `src/`, so neither scipy nor statsmodels is required.
