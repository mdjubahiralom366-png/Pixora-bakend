FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ ./training/
COPY server/ ./server/

# ---------------------------------------------------------------------------
# Trained model artifacts.
#
# models/pixora/model/ must already exist ON THE BUILD MACHINE before you
# run `docker build` — this Dockerfile never trains or downloads a model
# itself. See README section "Deploying the server" for how to produce or
# fetch it.
#
# Mode A (fine-tuning, the default/recommended TRAINING_MODE) saves BOTH
# the model weights AND the tokenizer into this one directory
# (train.py: trainer.save_model(final_dir); tokenizer.save_pretrained(final_dir)),
# so a single self-contained COPY is correct here — there is no separate
# models/pixora/tokenizer/ directory in this mode, and this Dockerfile no
# longer assumes one exists.
#
# Mode B (from-scratch, research track) additionally needs a standalone
# tokenizer directory. If you deploy a Mode B model, either:
#   (a) also save your tokenizer into models/pixora/model/ after training
#       so it ships in the same self-contained directory as Mode A, or
#   (b) uncomment the line below once models/pixora/tokenizer/ actually
#       exists on your build machine.
# ---------------------------------------------------------------------------
COPY models/pixora/model/ ./models/pixora/model/
# COPY models/pixora/tokenizer/ ./models/pixora/tokenizer/

# Fail the build now, with a clear message, if the model directory is
# missing or incomplete — instead of a confusing "checksum of ref: not
# found" error or a runtime crash after the container is already deployed.
RUN test -f ./models/pixora/model/config.json || \
    (echo "ERROR: ./models/pixora/model/config.json not found." && \
     echo "Train or download the model into models/pixora/model/ before running 'docker build'." && \
     echo "See README section 'Deploying the server'." && \
     exit 1)

ENV MODEL_DIR=/app/models/pixora
ENV PYTHONPATH=/app/training

WORKDIR /app/server
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
