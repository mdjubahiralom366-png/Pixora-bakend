# Pixora AI — Docker Build Fix Report

## A. ROOT CAUSE

The Dockerfile unconditionally runs:

```
COPY models/pixora/model/ ./models/pixora/model/
COPY models/pixora/tokenizer/ ./models/pixora/tokenizer/
```

`models/pixora/tokenizer/` is correctly gitignored (`models/**/tokenizer/`
in `.gitignore`) and was never committed — that part is intentional, not
a mistake. The real bug is deeper: **in the project's own default,
recommended training path, that directory is never produced at all.**

Tracing the actual code:

- `.env.example` sets `TRAINING_MODE=finetune` by default, labelled
  *"Mode A, recommended first"*.
- `training/train.py`'s `run_finetune()` loads the tokenizer from the
  base model (`Qwen/Qwen2.5-0.5B`) and, at the end, does:
  ```python
  final_dir = os.path.join(cfg["model_dir"], "model")
  trainer.save_model(final_dir)
  tokenizer.save_pretrained(final_dir)
  ```
  Both the model **and** the tokenizer are saved into
  `models/pixora/model/` — one directory, not two.
- `training/tokenizer.py` (the script that would populate
  `models/pixora/tokenizer/`) is a standalone CLI, only ever invoked
  manually, and only documented/needed for `TRAINING_MODE=scratch`
  (Mode B — explicitly called "a research track, not a v1 plan" in the
  code's own docstring and in the README).
- `server/model_server.py` confirms this split: in `finetune` mode it
  loads both `AutoTokenizer` and `AutoModelForCausalLM` from the same
  `MODEL_DIR` (i.e. `models/pixora/model/`); only in `scratch` mode does
  it separately load a tokenizer from `TOKENIZER_DIR`
  (`models/pixora/tokenizer/`).

So the Dockerfile's second `COPY` line assumes a directory that, for the
project's own default deployment path, is designed to never exist.
That's why the build fails with "not found" on a fresh checkout — it's
not a missing-step accident, it's a stale assumption left over from
(or never updated for) the two-directory Mode B layout.

The README itself has the same slip: section 6/8 correctly explain that
Mode A doesn't need a separate tokenizer directory, but section 14
("Deploying") still said both directories must exist — the doc
contradicted itself. Fixed below.

## B. ARCHITECTURE DECISION

**Fine-tuning (Mode A)**, per the project's own config and docs. Evidence:
`.env.example` defaults `TRAINING_MODE=finetune`; `train.py`'s docstring
calls it "the practical path"; the README calls Mode B "not a v1 plan."

Deployment is therefore built around **one self-contained
`models/pixora/model/` directory** holding weights + tokenizer + config
together, exactly as `run_finetune()` already produces it. Mode B is not
removed — the code path, `training/tokenizer.py`, and
`TOKENIZER_DIR`/`models/pixora/tokenizer/` all still work exactly as
before for anyone who sets `TRAINING_MODE=scratch` — but it's no longer
what the default Dockerfile silently assumes.

## C. FILES CHANGED

| File | Change |
|---|---|
| `Dockerfile` | Removed the unconditional `COPY models/pixora/tokenizer/`; single `COPY models/pixora/model/`; added a build-time existence check with a clear error message; Mode B path documented as an opt-in commented line. |
| `.dockerignore` | **New file.** Keeps `.git`, secrets, `dataset/`, checkpoints, and dev-only folders out of the build context. |
| `README.md` | Rewrote section 14 ("Deploying the server") to match reality; added section 15 explaining the gitignore strategy for model/dataset artifacts. |
| `requirements.txt` | Added `huggingface_hub>=0.24.0` as an explicit direct dependency (was already present transitively via `transformers`/`datasets`; now the deploy workflow calls `huggingface-cli` directly, so it's pinned explicitly). |

**Files inspected and found already correct — no change made:**
`training/tokenizer.py`, `training/train.py`, `training/model.py`,
`training/formatter.py`, `server/model_server.py`, `server/main.py`,
`.gitignore`. Reasoning for each is in section F/H below and inline in
the code trace above — none of them contain the bug; the bug was
entirely in the Dockerfile + the README's deploy instructions.

## D. COMPLETE FIXED FILES

All four changed files are attached alongside this report
(`Dockerfile`, `.dockerignore`, `README.md`, `requirements.txt`) — full
contents, not snippets, ready to drop into the repo as-is.

## E. REQUIRED MODEL/TOKENIZER DIRECTORY STRUCTURE

```
models/
└── pixora/
    ├── model/                  # REQUIRED before `docker build`
    │   ├── config.json         # <- Dockerfile's sanity check looks for this
    │   ├── model.safetensors   # (or pytorch_model.bin — depends on torch/transformers version)
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── special_tokens_map.json
    │   └── vocab.json / merges.txt (if applicable to the base tokenizer)
    │
    ├── tokenizer/               # OPTIONAL — Mode B (scratch) only, not used by default Dockerfile
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── vocab.json
    │   └── merges.txt
    │
    └── checkpoints/             # training-time only — gitignored AND dockerignored,
                                  # never shipped in the serving image
```

## F. GIT/GITHUB HANDLING

`.gitignore` is correct as-is and was **not** changed — `models/**/model/`,
`models/**/tokenizer/`, `models/**/checkpoints/`, and `*.bin`/`*.safetensors`/`*.pt`
being excluded is intentional (a fine-tuned Qwen2.5-0.5B is hundreds of
MB+; `dataset/*.jsonl` can hold real Firebase user data). Stripping
those from `.gitignore` to "fix" the Docker build would have been the
wrong move — you'd have committed large binaries and possibly user data
to git history permanently.

Instead, the trained model travels between machines via the
**Hugging Face Hub** (free, and the project already speaks
`transformers`/HF format natively — `AutoModelForCausalLM.from_pretrained`
already expects exactly this directory layout):

```bash
# one-time, after training completes
huggingface-cli login
huggingface-cli upload <your-hf-username>/pixora-model ./models/pixora/model .

# on whichever machine runs `docker build`
huggingface-cli download <your-hf-username>/pixora-model --local-dir ./models/pixora/model
```

`huggingface-cli` ships with `huggingface_hub`, now pinned directly in
`requirements.txt`. This needs no new account, no billing, and no extra
infrastructure — it fits a zero-investment, solo-maintained setup better
than standing up Git LFS (GitHub's free LFS bandwidth quota is tight for
repeated CI builds of a multi-hundred-MB model) or an S3/GCS/R2 bucket
(free tiers exist, e.g. Cloudflare R2, but need a cloud account and
credential wiring you don't otherwise have here). If you're already
comfortable with Git LFS or already have a bucket for something else,
either works fine too — swap the two commands above for
`git lfs pull`/`aws s3 cp` equivalents; nothing else in this fix depends
on which one you pick.

## G. LOCAL VERIFICATION

```bash
# 1. Confirm repository structure
find . -maxdepth 3 -not -path './.git*' | sort

# 2. Confirm no secrets or large artifacts are actually tracked by git
git ls-files | grep -E '(models/|dataset/|\.env$|secrets/)'   # expect: no output
git status                                                     # confirm .gitignore is working

# 3. Confirm the model directory is populated BEFORE building
ls -la models/pixora/model/            # expect config.json + weight file(s) + tokenizer files

# 4. Build
docker build -t pixora-server .
# If models/pixora/model/config.json is missing, this now fails immediately
# with a clear message instead of the old cryptic checksum error.

# 5. Run
docker run --rm -p 8000:8000 --env-file .env pixora-server

# 6. Health check
curl http://localhost:8000/health
# expect: {"status":"ok","model":"pixora"}

# 7. Chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, who are you?"}'

# 8. Sanity-check cross-directory imports inside the built image
docker run --rm pixora-server python -c \
  "import sys; sys.path.append('/app/training'); from formatter import DEFAULT_SYSTEM_PROMPT; print('imports OK')"
```

## H. FINAL CHECK

- ✅ No Docker `COPY` in the fixed `Dockerfile` references
  `models/pixora/tokenizer/` unconditionally — that line is now
  commented out, opt-in only for Mode B deployments, and only used once
  the directory genuinely exists on the build machine.
- ✅ `MODEL_DIR` matches the actual saved model location everywhere it's
  read: `.env.example` (`MODEL_DIR=./models/pixora`), the fixed
  `Dockerfile` (`ENV MODEL_DIR=/app/models/pixora`),
  `server/model_server.py` (`os.path.join(MODEL_DIR, "model")`), and
  `training/train.py`'s save path (`os.path.join(cfg["model_dir"], "model")`)
  all agree: `MODEL_DIR` is the *base* `models/pixora` path, and each
  consumer appends `"model"` itself. This was already consistent before
  this fix — confirmed by inspection, not changed.
- ✅ `PYTHONPATH=/app/training` left unchanged. It turned out to be
  redundant-but-harmless: every file that actually needs a cross-directory
  import (`server/model_server.py`, `server/memory.py`,
  `evaluation/benchmark.py`, `evaluation/evaluate_model.py`,
  `tests/test_pipeline.py`) already manages its own
  `sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))`
  at runtime, independent of any env var. `PYTHONPATH` doesn't conflict
  with this and adds a small extra safety net, so it was left as-is
  rather than removed for the sake of removing it.
- ✅ No Firebase credentials, service-account JSON, or `.env` values are
  copied into the image — the Dockerfile never referenced them, and the
  new `.dockerignore` now excludes them from the build context too as a
  defense-in-depth measure against future Dockerfile changes.
- ✅ No unrelated application code was changed. `training/tokenizer.py`,
  `training/train.py`, `training/model.py`, `server/model_server.py`,
  and `server/main.py` are untouched — inspected and confirmed already
  correct for this issue.
- ✅ Pixora remains fully self-hosted — nothing routes through OpenAI,
  Gemini, Claude, or any external inference API; this fix only touches
  the build/deploy plumbing.
