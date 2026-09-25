"""
evaluate_model.py

Produces one combined evaluation report for a given Pixora model version:
  - quantitative: validation loss, perplexity  (training/evaluate.py)
  - qualitative:  held-out test questions, repetition, refusal checks (benchmark.py)

Save the output alongside the model version so you can compare
pixora-v0.1 vs pixora-v0.2 objectively before deploying.

    python evaluate_model.py --version pixora-v0.2 --mode finetune
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
import evaluate as quant_eval  # noqa: E402
from benchmark import run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="e.g. pixora-v0.2")
    parser.add_argument("--mode", choices=["finetune", "scratch"], default="finetune")
    parser.add_argument("--model-dir", default=os.path.join(os.environ.get("MODEL_DIR", "./models/pixora"), "model"))
    parser.add_argument("--val", default=os.path.join(os.environ.get("DATASET_DIR", "./dataset"), "validation.jsonl"))
    parser.add_argument("--skip-server-benchmark", action="store_true",
                         help="Skip the live /api/chat qualitative pass (server not running).")
    args = parser.parse_args()

    report = {
        "version": args.version,
        "mode": args.mode,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    if args.mode == "finetune":
        report["quantitative"] = quant_eval.evaluate_finetuned(args.model_dir, args.val)
    else:
        config_path = os.path.join(args.model_dir, "config.json")
        val_packed = os.path.join(os.environ.get("DATASET_DIR", "./dataset"), "validation_packed.npy")
        report["quantitative"] = quant_eval.evaluate_scratch(args.model_dir, val_packed, config_path)

    if not args.skip_server_benchmark:
        try:
            report["qualitative"] = run_benchmark(
                os.path.join(os.path.dirname(__file__), "test_questions.json")
            )
        except Exception as exc:  # noqa: BLE001
            report["qualitative_error"] = f"Could not reach live server: {exc}"

    out_path = os.path.join(os.path.dirname(__file__), f"report-{args.version}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Wrote evaluation report to {out_path}")
    print(json.dumps(report.get("quantitative", {}), indent=2))


if __name__ == "__main__":
    main()
