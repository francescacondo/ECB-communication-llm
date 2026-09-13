# Prompt versions

Two versions are numbered, because two are analysed. Earlier drafts are
kept for provenance under descriptive names rather than numbers.

`extraction_prompt.txt` is **version 1**. It is the specification the
signals in `outputs/extracted_signals/ecb_signals_gpt5mini_v1.csv` were
generated under, and it is preserved byte-for-byte so that extraction
remains reproducible. Do not edit it.

`extraction_prompt_v2.txt` is **version 2**. It adds the six decision
rules derived from adjudicating disagreements between two independent
coders, and mirrors `docs/coding_rules.md` at version 2. Nothing has
been extracted under it yet.

`src/extract_signals_openai.py` and `src/extract_signals.py` both read
`extraction_prompt.txt`. Switching them to version 2 is a deliberate act
that requires a full re-extraction, which is costly; it should not be
done incidentally.

The consequence to keep in view: human codes made under version 2 are
made against rules the current model codes never saw. Any comparison
between the two is measuring the model against a specification it was
not given, and should say so.

## The predecessor prompt was recovered

`extraction_prompt_predecessor.txt` is the draft immediately preceding
version 1. It was reconstructed on 2026-08-27 from a copy the author
still held, after it was found to be absent from the repository. **It is
a reconstruction, not a recovered artefact**: the substantive text is as
supplied, but indentation, bullet characters and quote marks may not
match the original byte-for-byte. Nothing in the analysis depends on
those.

The prompt was evidently edited in place rather than versioned, since no
file of it survived. It cannot be said when that happened: the git
history available in this repository does not reach back to the
extraction work, so it carries no evidence about the sequence.

## What changed between the predecessor and version 1

Normalising away formatting, exactly one line differs. Version 1 added a
fourth bullet to `uncertainty_attention`:

> Do not code 1 merely because technical, operational, legal, credit,
> settlement, or financial-system risks are discussed. Such risks count
> as uncertainty attention only when the text substantively emphasizes
> uncertainty, unpredictability, unusually wide possible outcomes, or
> difficulty assessing the outlook.

This is the same clause that was misfiled into section 5 of
`docs/coding_rules.md`, where it contradicted that section's inclusion
list. The early codebook draft, preserved as
`docs/coding_rules_early_draft.md`, carries no such clause in either
section, so **the misfiling happened at the revision that produced
version 1**: the clause entered the prompt's section 6 correctly and the
codebook's section 5 incorrectly, and the two copies were never
compared.

That dating rests on the documents themselves rather than on version
control, which is the only evidence available.

## The prompt-sensitivity comparison

`openai_test5_gpt5mini_predecessor.csv` and its version-1 counterpart
cover the same five speeches with the same model and the same reasoning
effort, so the prompt is the only thing that differs. Twenty-eight of
thirty cells agree, and the two that differ are now interpretable.

`uncertainty_attention` on `ecb_0066` moves from 1 to 0. That speech is
titled "Payments and the Eurosystem" and contains 24 mentions of
payment, 10 of settlement and 3 of clearing against a single mention of
uncertainty. It is exactly the case the new clause was written to
exclude, and it is the only speech among the five that instantiates it.
One targeted wording change produced one code change, on the intended
variable, on the intended kind of speech.

`growth_attention` on `ecb_0175` moves from 1 to 0, and **no prompt
change can explain it** --- nothing about `growth_attention` differs
between the two versions. Two explanations remain and thirty cells
cannot separate them: run-to-run nondeterminism, or spillover, since a
language model reads the whole prompt and a clause added under one
variable can shift behaviour under another. Either way it is the
project's only estimate of how much the extraction moves when the
instructions do not, and it is one cell in thirty.

From version 1 onward each version is a separate file and none is
edited in place.
