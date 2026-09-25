# Pixora AI — self-hosted language-model pipeline

Firebase → Python data pipeline → dataset → tokenizer → Pixora model →
training/fine-tuning → FastAPI server → Android app.

No OpenAI, Gemini, Claude, or Grok API is used anywhere in this stack.
Pixora only ever runs its own locally-saved weights.

```
pixora-ai/
├── training/        firebase_loader.py, cleaner.py, formatter.py,
│                     tokenizer.py, dataset_builder.py, model.py,
│                     train.py, evaluate.py
├── server/           main.py, model_server.py, prompts.py, memory.py
├── evaluation/        benchmark.py, evaluate_model.py, test_questions.json
├── models/            saved model + tokenizer + checkpoints (gitignored)
├── dataset/           generated train/validation data (gitignored)
├── tests/             pipeline smoke tests
├── requirements.txt, .env.example, .gitignore, Dockerfile
```

## 1. Firebase setup

Your web config (`apiKey`, `authDomain`, etc.) is what the **Android app**
uses to talk to Firebase directly — it's meant to be public-ish and
restricted via Firebase Security Rules, not a secret.

Training, by contrast, needs **Admin SDK** access (full read of your
training collection), which is a different, privileged credential:

1. Firebase Console → Project Settings → Service Accounts → **Generate new private key**.
2. Save the downloaded JSON as `secrets/firebase-admin.json` (already gitignored).
3. Copy `.env.example` to `.env` and set `FIREBASE_CREDENTIALS_PATH` to that path.

**Never commit `secrets/firebase-admin.json` or `.env` to GitHub.** That
key can read/write your entire database — treat it like a password.

## 2. How training data should be structured in Firebase

Store either shape in your `training_examples` collection (Firestore) or
RTDB path:

```json
{"instruction": "Explain photosynthesis simply.", "input": "", "output": "Photosynthesis is..."}
```
```json
{"messages": [{"role": "user", "content": "What is gravity?"},
              {"role": "assistant", "content": "Gravity is..."}]}
```

If you track user consent, add a field like `"consented": true` and pass
`--require-consent-field consented` to `dataset_builder.py` — records
without it are dropped, not just skipped silently.

## 3. How Python downloads the data

`training/firebase_loader.py` uses `firebase-admin` to page through your
collection/path and return plain dicts. It does not clean or judge the
data — it only retrieves it. Run it standalone to sanity-check a pull:

```bash
python training/firebase_loader.py
```

## 4. How the dataset is cleaned

`training/cleaner.py`:
- drops empty/malformed records
- deduplicates by content fingerprint
- normalizes unicode/whitespace
- filters records that are too short (<8 chars) or absurdly long (>20k chars)
- redacts API keys, AWS keys, private-key blocks, passwords, bearer tokens
- redacts emails and phone numbers
- never trains on anything lacking your consent flag, if you set one

## 5. Building a dataset version end-to-end

```bash
python training/dataset_builder.py --version dataset-v1
```

This runs loader → cleaner → formatter, writes
`dataset/dataset-v1/{raw_pull,cleaned,train,validation}.jsonl` plus a
`manifest.json` with counts and the clean report, and copies
`train.jsonl` / `validation.jsonl` to `dataset/` root for training to pick up.

## 6. Tokenization

`training/tokenizer.py` trains a real byte-level BPE tokenizer (not
character-splitting) over your formatted corpus:

```bash
python training/tokenizer.py --train ./dataset/train.jsonl --out ./models/pixora/tokenizer --vocab-size 32000
```

This is only needed for **Mode B** (from-scratch). In **Mode A**
(fine-tuning), you reuse the base model's own tokenizer — don't retrain
one, or the embeddings won't line up.

## 7. Training

Two modes, controlled by `TRAINING_MODE` in `.env`:

**Mode A — fine-tuning (recommended first)**
Starts from an open-weight base model (default `Qwen/Qwen2.5-0.5B`,
swap for anything on Hugging Face you're licensed to use) and continues
training it on your Firebase-derived data via `transformers.Trainer`.
This is the realistic way to get a Firebase-sized dataset to actually
change model behavior.

**Mode B — from-scratch (research track)**
Trains `training/model.py`'s `PixoraForCausalLM` — a real GPT-style
Transformer with configurable `vocab_size`, `hidden_size`, `num_layers`,
`num_attention_heads`, `max_sequence_length` — from randomly initialized
weights on packed token blocks. Be aware: a from-scratch model needs
orders of magnitude more data and compute than fine-tuning to produce
coherent text at all. A dataset the size of a typical app's Firebase
collection will not get a from-scratch model to ChatGPT-level anything —
plan on this as a longer-term research effort, not v1.

```bash
# Mode A
python training/dataset_builder.py --version dataset-v1
python training/train.py

# Mode B (also needs a tokenizer and packed token shards first)
python training/tokenizer.py --train ./dataset/train.jsonl --out ./models/pixora/tokenizer
python -c "from dataset_builder import pack_for_scratch_training as p; \
p('./models/pixora/tokenizer', './dataset/train.jsonl', './dataset/train_packed.npy', 1024); \
p('./models/pixora/tokenizer', './dataset/validation.jsonl', './dataset/validation_packed.npy', 1024)"
python training/train.py   # with TRAINING_MODE=scratch in .env
```

Both modes support batch size, gradient accumulation, learning rate,
epochs, checkpointing every `SAVE_EVERY_STEPS`, resuming
(`--resume <path>`), GPU auto-detection with CPU fallback, and mixed
precision (bf16/fp16) when a GPU supports it.

## 8. Where the trained model lands

```
models/pixora/
├── model/          the deployable model (+ tokenizer for Mode A)
├── tokenizer/       standalone tokenizer (Mode B, or a shared copy)
├── checkpoints/      periodic training checkpoints
```

## 9. Inference

`server/model_server.py` loads `models/pixora/model` once at startup and
exposes `generate(message, history)`. It never calls out to any external
AI API — inference is 100% local to this process.

## 10. Running the FastAPI server

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your values
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /api/chat` — `{"message": "...", "chat_id": "abc123"}` → `{"success": true, "response": "...", "model": "pixora"}`
- `POST /api/chat/stream` — same body, newline-delimited JSON chunks (see note in `main.py` about wiring true token-level streaming for Mode A via `TextIteratorStreamer`)

`server/memory.py` stores per-`chat_id` history in Firestore so
conversations have context across turns; `server/prompts.py` does basic
input/output safety filtering (expand this — it's a starting point, not
a complete safety system).

## 11. Connecting the Android app

Point your existing Pixora Android HTTP client at
`POST http://<your-server>:8000/api/chat` with the same JSON shape shown
above. No client-side changes to your Firebase config are needed —
Android keeps talking to Firebase for auth/storage/chat history as
before; only the *AI response* comes from this server now, not any
third-party API.

## 12. Continuing training with new data (the improvement loop)

```
New authorized data → Firebase → dataset_builder.py --version dataset-v2
                    → train.py --resume <last checkpoint>
                    → evaluate_model.py --version pixora-v0.2
                    → compare against pixora-v0.1's report → deploy if better
```

Version everything: `dataset-v1`, `dataset-v2`, ... and `pixora-v0.1`,
`pixora-v0.2`, ... so you can always trace which data produced which
model and roll back.

## 13. Evaluating a model version

```bash
python training/evaluate.py --mode finetune --model-dir ./models/pixora/model --val ./dataset/validation.jsonl
# combined quantitative + qualitative report:
python evaluation/evaluate_model.py --version pixora-v0.2 --mode finetune
```

Reports validation loss, perplexity, repetition rate, and pass/fail on
`evaluation/test_questions.json` (instruction-following, factuality,
reasoning, safety refusals, multilingual, repetition). A lower training
loss does not automatically mean a better assistant — always check the
qualitative report too.

## 14. Deploying the server

The `Dockerfile` copies a pre-built model in rather than training or
downloading anything itself, so `models/pixora/model/` must exist **on
the build machine** before you run `docker build`. For Mode A
(fine-tuning, the default), that one directory already contains both
the model weights and the tokenizer — there is no separate
`models/pixora/tokenizer/` directory to provide in this mode.

**Get the model onto your build machine.** Pick one:

- **You just trained it locally** — nothing to do, `models/pixora/model/`
  is already there from step 7.
- **You trained elsewhere (e.g. a free Colab/Kaggle GPU) and are
  deploying from a different machine** — push the trained folder to a
  free Hugging Face Hub model repo once, then pull it wherever you
  build:
  ```bash
  # one-time, after training
  huggingface-cli login
  huggingface-cli upload <your-hf-username>/pixora-model ./models/pixora/model .

  # on the build machine, before `docker build`
  huggingface-cli download <your-hf-username>/pixora-model --local-dir ./models/pixora/model
  ```
  This keeps large weights out of git entirely (see section 15) at no
  cost — `huggingface-cli` ships with `huggingface_hub`, already
  installed via `requirements.txt`. Git LFS or an S3/GCS/R2 bucket work
  too if you'd rather keep everything in one place; the Hub is just the
  path of least setup here since the project already speaks
  `transformers`/HF format natively.

**Then build and run:**

```bash
docker build -t pixora-server .
docker run -p 8000:8000 --env-file .env pixora-server
```

The build fails fast with a clear message if `models/pixora/model/` is
missing or incomplete, instead of failing later inside the container.

If you deploy a **Mode B (scratch)** model instead, it needs a separate
tokenizer directory — either save your tokenizer into
`models/pixora/model/` too (so it stays a single self-contained
directory, same as Mode A), or uncomment the second `COPY` line in the
`Dockerfile` once `models/pixora/tokenizer/` exists on your build
machine.

## 15. Why model/dataset files are gitignored, and how they move around instead

`.gitignore` excludes `dataset/*.jsonl`, `models/**/model/`,
`models/**/tokenizer/`, `models/**/checkpoints/`, and raw weight files
(`*.bin`, `*.safetensors`, `*.pt`) on purpose, not by accident:

- **Dataset files** can contain real user content pulled from Firebase —
  they don't belong in a public (or even private) git history.
- **Model/tokenizer/checkpoint files** are large binary artifacts (a
  fine-tuned Qwen2.5-0.5B is several hundred MB+) that bloat a git repo
  every time they change, which git isn't designed for.

None of this is removed to work around the Docker build — the intended
architecture is that trained artifacts travel between machines via the
Hugging Face Hub (or Git LFS / object storage, see section 14), not via
`git add`. Secrets (`.env`, `secrets/`, `firebase-admin*.json`) stay
gitignored for the obvious reason and are never baked into the Docker
image either (see `.dockerignore`).

## Concepts, kept separate on purpose

1. **Training** — updating model weights on a full pass of a dataset.
2. **Fine-tuning** — continuing training an existing model on new data (Mode A).
3. **Retrieval from Firebase** — reading stored data/chat history; never generates text.
4. **Model inference** — generating new text from learned weights, given a prompt.

Pixora's chat responses always come from (4), informed by data that
went through (1)/(2) at training time — never by looking up and
returning a stored Firebase answer verbatim.
