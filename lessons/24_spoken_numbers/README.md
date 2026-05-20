# Lesson 24 · Spoken-number → digit normalization

The follow-up to lessons 21, 22, and 23. When a Turkish customer says
*"hamdi yirmi beş on iki yetmiş dokuz"* (= hamdi / 25 / 12 / 79),
we need those number-words turned into digits **before** the LLM sees
them — otherwise we re-introduce the decomposed-fields hallucination
from lesson 21 by asking the model to do the conversion in-prompt.

This lesson surveys the PyPI landscape, builds a rule-based Turkish
parser (since none exists), adds fuzzy partial-matching for typos and
speech-to-text glitches, wraps everything as a tool the LLM can call,
and answers three honest questions about the design.

## What you'll learn

- What's on PyPI for spoken-number parsing — `text2num` (7 European
  languages, **no Turkish**), `num2words` (opposite direction), `word2number`
  (English only), Zemberek (Turkish morphology but no number parser),
  NVIDIA NeMo ITN (heavy, neural, English/Russian-leaning).
- Why **tokenizers are the wrong tool** for this problem.
- How to build a ~100-line rule-based Turkish number parser that
  handles 0–999,999 and the diacritic variants.
- How to handle **partial matches and typos** with `rapidfuzz` and a
  three-tier confidence escalation (accept / confirm / reject).
- How to wrap parsers as **`@tool`s the LLM can call**, and an honest
  answer to *"does adding a tool cause hallucination?"*.

## Why it matters

Speech-to-text systems (Whisper, AssemblyAI, Deepgram, Azure Speech)
sometimes emit numbers as digits, sometimes as words, and the
choice varies by language. Turkish STT tends to emit words. If you're
building voice IVR, customer-support hotlines, or any tool that
ingests transcribed audio, you'll hit this problem.

## The library landscape — honest survey

| Library / Approach | Direction | Languages | Turkish? | Maintained? | Verdict |
|---|---|---|---|---|---|
| **`text2num`** ([PyPI](https://pypi.org/project/text2num/)) | words → int | EN, FR, ES, DE, PT, IT, NL | ❌ | Yes (v3.0 with Rust core) | **Use it for these 7 languages.** Wrap as a tool. |
| **`num2words`** ([PyPI](https://pypi.org/project/num2words/)) | int → words | 40+ langs incl. TR | (reverse direction only) | Yes | Wrong direction. Useful for the OUTPUT side (lesson 23). |
| **`word2number`** ([PyPI](https://pypi.org/project/word2number/)) | words → int | EN only | ❌ | Stalled | Skip. text2num is the more active project. |
| **`text2digits`** ([PyPI](https://pypi.org/project/text2digits/)) | words → int (inline) | EN-focused | ❌ | Older | Use if you need *in-line* replacement of words with digits in a sentence. |
| **Zemberek** / `zemberek-python` ([GitHub](https://github.com/ahmetaa/zemberek-nlp)) | Turkish morphology | TR | (no number parser) | Yes | Excellent for stemming / spell-check, but **does not** ship a number-word parser. |
| **`trnltk`** / `turkishnlp` | Turkish NLP | TR | (no number parser) | Limited | Same — morphology, not numbers. |
| **NVIDIA NeMo-text-processing** (ITN) | words → digits (full text) | EN, RU, DE, ES (limited TR) | weak | Yes | Heavyweight (WFST grammars + neural). Overkill for a single tool, but state-of-the-art for production ASR post-processing. |
| **Whisper "digit mode"** (model prompt) | configurable at ASR layer | many | partial | Yes | Often the simplest fix — push the problem upstream to the STT layer. |
| **Rule-based parser (this lesson)** | words → int | TR | ✅ | This repo | What we built. ~100 lines because Turkish numbers are grammatically regular. |

**The verdict for Turkish:** there is no off-the-shelf PyPI library
that parses Turkish number-words into integers. We roll our own — it's
much smaller than you'd expect because Turkish number formation is
strictly compositional (`tens + unit`, multiplied by `yüz` for
hundreds and `bin` for thousands).

## Can we use tokenizers for this?

**Honest answer: tokenizers are the wrong tool.**

It's an understandable instinct — Hugging Face has tokenizers for
Turkish (BERTurk, mBERT, XLM-R), and Turkish number words are in their
vocab. But tokenizers split text into IDs; they don't *understand*
what those IDs mean numerically.

What you'd still need:

```
tokenizer('yirmi beş')  →  [token_id_for_yirmi, token_id_for_beş]
                            ↓
                  manual lookup table: token_id → integer value
                            ↓
                  manual logic: tens × 1 + units = result
```

The lookup table and the compositional logic are the actual work, and
neither comes from the tokenizer. **The tokenizer just splits.** If
you already have to write the table and the math, you might as well
skip the tokenizer and write the table-driven parser directly — which
is what this lesson does.

Where tokenizers *do* help adjacent problems:

- **Pre-segmenting noisy STT output** into word boundaries when whitespace is unreliable
- **Spell-correcting unknown tokens** by finding the nearest vocab entry (rapidfuzz does this without a tokenizer dependency)
- **As input to a fine-tuned token-classifier** (BERT-NER) that predicts whether each token is part of a number — but that's full ML, not "use a tokenizer"

For the specific job *"convert this number phrase to an integer"*,
deterministic Python beats anything ML-shaped here.

## Walk through `example.py`

Six demos, all runnable without an API key:

| Demo | What it shows |
|---|---|
| 1 · `--numbers` | Core Turkish parser — 12 test phrases from `"sıfır"` to `"üç bin dört yüz elli altı"` |
| 2 · `--dob` | DOB segmenter — splits `"yirmi beş on iki yetmiş dokuz"` into `(25, 12, 79)`. Handles month NAMES too: `"on beş mart bin dokuz yüz seksen yedi"` → `(15, 3, 1987)` |
| 3 · `--others` | Note pointing at `text2num` for EU languages |
| 4 · `--full` | End-to-end: spoken phrase → ISO 8601 string (`"1979-12-25"`) |
| 5 · `--fuzzy` | Fuzzy partial-matching for typos / STT glitches with `rapidfuzz` |
| 6 · `--multilingual` | The `parse_spoken_number` `@tool` — `text2num` for EU langs, our parser for TR, clean error for unsupported locales |

Run any individually or all at once:

```bash
uv run python -m lessons.24_spoken_numbers.example                # all
uv run python -m lessons.24_spoken_numbers.example --fuzzy        # demo 5
uv run python -m lessons.24_spoken_numbers.example --multilingual # demo 6
```

---

## Q1 · What to do when `partial_ratio > 80` (the partial-match question)

This is the meat of the lesson. When the rule parser fails outright on
a token (typo, STT glitch, missed letter), we fall back to fuzzy
matching against the vocabulary. **The score determines the
*next action*, not a single binary "accept/reject".**

### Three-tier escalation policy

```
              ┌─────────────────────────────────────────────┐
              │  rapidfuzz score of best vocab match        │
              └─────────────────────┬───────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
        ≥ 95                    80 – 94                    < 80
            │                       │                       │
            ▼                       ▼                       ▼
       AUTO-ACCEPT             CONFIRM                   REJECT
   silently snap to vocab    snap AND surface to    ask user to repeat
   continue parsing          user for confirmation  (or hand to LLM)
```

Concretely:

```python
ACCEPT_THRESHOLD  = 95   # auto-snap, log it but don't bother the user
CONFIRM_THRESHOLD = 80   # snap AND surface — "did you mean X?"
# below 80          → don't snap; reject; ask user to repeat
```

### Why these tiers, not a single threshold

- **`≥ 95`**: only minor typos or diacritic variants (`"yirmı"` → `"yirmi"`). Auto-correction is safe and silent.
- **`80–94`**: ambiguous — the match is plausible but not certain (`"yedı"` → `"yedi"` scores 86 on `partial_ratio`). The cost of a wrong silent snap is high (wrong birth date in the database, lesson 21), so we **always surface to the user**: *"Did you say 'doksan yedi' (97)?"*. Cheap conversational round-trip; protects the database.
- **`< 80`**: too uncertain to guess. Either ask the user to repeat *"Sorry, could you repeat the year?"* or escalate to an LLM (next section).

### A subtle fact about short tokens

Turkish number units are short (`"bir"`, `"iki"`, `"üç"`, 2-3 chars).
A one-character substitution in a 3-char word scores ~67 on
`fuzz.ratio` — below both thresholds. **Fuzzy matching is more
reliable on longer tokens** (`"doksan"`, `"yetmiş"`, `"yüz"`). For the
shortest tokens (1-2 chars), prefer a hard typo allow-list:

```python
# Add common STT confusions explicitly
TR_TYPO_FIXES = {"bın": "bin", "ıki": "iki", "bır": "bir", "uc": "üç"}
```

The rule parser in `example.py` already accepts both diacritic and
no-diacritic forms (`"kırk"` / `"kirk"`, `"beş"` / `"bes"`) for the
common single-character difference.

### Should we send the whole conversation to the LLM on partial match?

**Usually no.** Three reasons:

1. **The agent already has the context.** In an agentic flow (lesson 10's `create_agent`), the conversation lives in `MessagesState`. The agent's *next* decision step already sees everything; you don't need to *re-supply* it. Just return the partial-parse result from your tool — the agent picks it up.

2. **Token cost.** Sending the whole conversation back through the LLM for every uncertain token escalates cost super-linearly with conversation length. Send the **minimum payload** that lets the LLM decide.

3. **Misleading context.** Prior turns might bias the LLM. If the customer was just told their balance is $1,987, the LLM might "see" 1987 in an ambiguous year. Pass the spoken phrase **in isolation** plus the parser's best guess.

### The right escalation payload

```python
# When the rule parser is uncertain, hand the LLM the smallest useful payload:
escalation_payload = {
    "spoken_text":   "yedı",                          # the raw user input
    "rule_best":     "yedi",                          # parser's best guess (if any)
    "confidence":    86,                               # rapidfuzz score 0-100
    "alternatives":  [("yedi", 86), ("edi", 81)],     # other candidates
}
```

The agent's system prompt then says something like:

> *If the user's spoken input is ambiguous, ask them to confirm or
> repeat. Use the rule parser's best guess only if confidence ≥ 95.
> Otherwise ask: "Did you say X?".*

The LLM doesn't need the whole conversation to make that decision —
it has the *minimum* it needs (the ambiguous phrase + the parser's
suggestion) and the agent state to draft a clarifying question.

---

## Q2 · Can we wrap `text2num` as a tool?

**Yes** — and it's the canonical pattern. The lesson ships a
`parse_spoken_number` `@tool` that:

- Dispatches to `text2num` for `en` / `fr` / `es` / `de` / `pt` / `it` / `nl`
- Dispatches to our rule parser for `tr`
- Returns a structured result with `ok`, `value`, `engine`, and (for failures) `supported_locales`

```python
from langchain_core.tools import tool
from text_to_num import text2num

@tool
def parse_spoken_number(text: str, locale: str = "en") -> dict:
    """Parse a spoken-form number phrase into an integer.

    The agent should call this tool RATHER THAN doing the conversion
    in-prompt. Every digit the LLM generates itself is a chance to
    drift (lesson 21's decomposed-fields hallucination). Tool output
    is deterministic.
    """
    locale = locale.lower()
    if locale == "tr":
        # Our Turkish rule parser
        value, corrections = turkish_words_to_int_fuzzy(text)
        return ({"ok": True, "value": value, "engine": "turkish_rule_parser"}
                if value is not None else
                {"ok": False, "error": f"could not parse {text!r}"})

    if locale in {"en", "fr", "es", "de", "pt", "it", "nl"}:
        try:
            return {"ok": True, "value": text2num(text, locale), "engine": "text2num"}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": f"unsupported locale {locale!r}",
            "supported_locales": ["en", "fr", "es", "de", "pt", "it", "nl", "tr"]}
```

Verified output (`--multilingual`):

```
✓ (en) 'one thousand nine hundred eighty seven'  →  {'ok': True, 'value': 1987, 'engine': 'text2num'}
✓ (fr) 'mille neuf cent quatre-vingt-sept'       →  {'ok': True, 'value': 1987, 'engine': 'text2num'}
✓ (es) 'mil novecientos ochenta y siete'         →  {'ok': True, 'value': 1987, 'engine': 'text2num'}
✓ (de) 'neunzehnhundertsiebenundachtzig'         →  {'ok': True, 'value': 1987, 'engine': 'text2num'}
✓ (tr) 'bin dokuz yüz seksen yedi'               →  {'ok': True, 'value': 1987, 'engine': 'turkish_rule_parser'}
✗ (jp) 'twenty five'                             →  {'ok': False, 'error': "unsupported locale 'jp'"}
```

---

## Q3 · Does wrapping it as a tool cause hallucination?

The honest answer has three parts.

### (a) The tool itself does NOT hallucinate

A tool is a deterministic Python function. `text2num("mille", "fr")`
returns `1000`. Always. It has no LLM in its body. It cannot
hallucinate by construction.

### (b) The LLM's **tool-call arguments** can be wrong

This is where the risk lives. The agent picks the `text` and `locale`
arguments to pass. Two things can drift:

1. **`text` drift** — the LLM passes a paraphrase instead of the raw user span. E.g., user says *"yirmi beş on iki yetmiş dokuz"*, LLM passes `text="yirmi on iki yetmiş dokuz"` (silently dropped `"beş"`). The tool then parses an *almost-right* number that looks plausible. This is the **decomposed-fields hallucination** from lesson 21 in a new disguise.

   **Fix:** make the description forceful — *"Pass the customer's spoken phrase **exactly as said**. Do NOT paraphrase, summarise, or normalise. The tool does the normalisation."*

2. **`locale` drift** — the LLM passes `locale="tr"` for an English-spoken number, or `locale="jp"` for Japanese (unsupported). The tool returns an error or wrong value.

   **Fix:** the tool's `supported_locales` field in the failure response gives the LLM a corrective hint to retry with a valid locale.

### (c) Using a tool **REDUCES** hallucination compared to the alternative

The alternative — having the LLM do the number-word → digit conversion
in-prompt — has *higher* hallucination risk because:

- The conversion is a deterministic computation; LLMs are unreliable at deterministic computations (Test of Time, arXiv 2406.09170)
- Numeric outputs have low character-level redundancy; a single mistaken digit produces a syntactically valid wrong answer that's hard to detect downstream
- Multi-digit numbers (1987 vs 1897 vs 1797) score similarly on the model's next-token distribution

With the tool: the LLM only has to copy the user's phrase verbatim
(its strongest skill — span identification) and pass the right
locale. The arithmetic happens outside.

### Verifiable rule of thumb

| Pattern | Hallucination surface | Verdict |
|---|---|---|
| LLM does conversion in-prompt: *"convert 'on iki' to integer in your head"* | High — every digit is a draw from the next-token distribution | **Avoid.** |
| LLM fills multiple decomposed int fields (lesson 21's case) | High × N — N fields = N independent draws | **Avoid.** |
| LLM calls a tool with `text` = raw span, `locale` = known | Low — only `text` and `locale` are LLM-generated, both validatable | **The right pattern.** This lesson. |
| LLM calls a tool, tool returns a number, LLM uses it | Low — the number is the tool's output, not the LLM's | Same as above. |

### One more guardrail

Pair the tool with `field_validator`-style sanity checks (lesson 21's
layer 4) downstream:

```python
class ParsedDOB(BaseModel):
    day: int = Field(ge=1, le=31)
    month: int = Field(ge=1, le=12)
    year: int = Field(ge=1900, le=date.today().year)

# If text2num returns 25 / 13 / 1987, the schema rejects month=13.
# The agent sees the error and asks the user to repeat the month.
```

The tool returns plausible numbers; the schema enforces them being
*sensible* numbers. Defence in depth — the principle from lesson 22.

---

## Anti-patterns

| Smell | Why it's bad | Fix |
|---|---|---|
| Asking the LLM `"convert 'yirmi beş' to an integer"` in a tool description | LLM does it in-prompt, drift possible | Use this lesson's tool wrapper |
| Snapping every fuzzy match below 95 without user confirmation | Silent corruption | Three-tier policy: accept ≥ 95, confirm 80-94, reject < 80 |
| Sending the whole conversation back to the LLM on each ambiguous token | Token cost explodes; misleading context | Pass minimum payload (raw phrase + best guess + confidence) |
| Using `text2num` for Turkish | Not supported; raises ValueError | Use this lesson's Turkish parser |
| Trusting STT-emitted digits silently | Whisper can also hallucinate (lesson 21 deep-dive cites arXiv 2506.20168) | Validate downstream with schema + business rules |
| Fuzzy-matching very short tokens (1-2 chars) | Score is unreliable; one-char error = ~50% score | Use a hard typo allow-list for the small known set |

## Try it yourself

- **Add a French DOB demo.** Same pattern as the Turkish one but using `text2num` directly. Test against `"vingt-cinq décembre mille neuf cent soixante-dix-neuf"`.
- **Hook the tool into the `customer_support_bot` capstone.** When the bot asks for DOB, the agent calls `parse_spoken_number` instead of letting the model fill `birth_day` / `birth_month` / `birth_year` decomposed integers.
- **Build a Whisper-side comparison.** Run Whisper with and without `"--digits"` mode on a recording of *"yirmi beş on iki yetmiş dokuz"* and compare the output. (Whisper's behaviour for Turkish varies by version.)
- **Add `alpha2digit` for in-line replacement.** When the user says a long sentence *"I was born on the twenty-fifth of December nineteen eighty-seven"*, `alpha2digit(text, 'en')` returns `"I was born on the 25 of December 1987"`. That output then goes through lesson 21's pipeline cleanly.

## References

### Vendor / library docs

- [`text2num` on PyPI](https://pypi.org/project/text2num/) + [GitHub: allo-media/text2num](https://github.com/allo-media/text2num) — the canonical European-languages word-to-int parser
- [`num2words` on PyPI](https://pypi.org/project/num2words/) — opposite direction; useful for lesson 23
- [`rapidfuzz` documentation](https://rapidfuzz.github.io/RapidFuzz/) + [`fuzz.partial_ratio`](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html) — the fuzzy-matching engine
- [Zemberek-NLP (Java)](https://github.com/ahmetaa/zemberek-nlp) + [zemberek-python](https://pypi.org/project/zemberek-python/) — Turkish morphology / stemming, but no number parser
- [NVIDIA NeMo-text-processing — text normalization](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html) — WFST-based ITN, heavier and more complete for production ASR

### Engineering blogs

- [Kasper Junge · *Rapidfuzz Explained* (Medium)](https://medium.com/@kasperjuunge/rapidfuzz-explained-c26e93b6012d) — practical intro to `ratio` / `partial_ratio` / `token_sort_ratio`
- [CodeCut · *RapidFuzz: Find Similar Strings Despite Typos*](https://codecut.ai/rapidfuzz-rapid-string-matching-in-python/) — typo correction patterns
- [DataCamp · *Fuzzy String Matching in Python*](https://www.datacamp.com/tutorial/fuzzy-string-python) — threshold tuning guide

### Pairs with

- **[Lesson 21 · Date parsing](../21_date_parsing/README.md)** — the *decomposed-fields hallucination* anti-pattern this lesson explicitly avoids
- **[Lesson 22 · Architecture](../22_architecture/README.md)** — the validation + guardrail layers this lesson's tool fits into
- **[Lesson 23 · Date computation & localized output](../23_date_localization/README.md)** — the OPPOSITE direction (int → words via `num2words` + `babel`)
- **[`docs/research/llm-date-solutions-deep-dive.md`](../../docs/research/llm-date-solutions-deep-dive.md)** — the "let a tool do the parsing, not the LLM" pattern at the architecture level

## Next →

Capstones in [`projects/`](../../projects/). The
`customer_support_bot` is the natural place to wire this lesson's
`parse_spoken_number` tool in for any voice / chat input that needs
DOB.
