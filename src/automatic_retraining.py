
from pathlib import Path
import json
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MONITORING = ROOT / "monitoring"
TRIGGER_FILE = MONITORING / "retraining_trigger.json"
VERIFIED_FILE = MONITORING / "verified_retraining_data.csv"
STATE_FILE = MONITORING / "retraining_state.json"
LOCK_FILE = MONITORING / "retraining.lock"

MIN_SAMPLES = 100
LABELS = {"negative", "neutral", "positive"}

with open(TRIGGER_FILE, encoding="utf-8") as file:
    trigger = json.load(file)

data = pd.read_csv(VERIFIED_FILE)

required = {"text", "true_sentiment", "label_source"}
if not required.issubset(data.columns):
    raise ValueError(f"Missing columns: {required - set(data.columns)}")

verified = data[
    data["label_source"].astype(str).str.strip().str.lower()
    == "human_verified"
].copy()

verified["text"] = verified["text"].fillna("").astype(str).str.strip()
verified["true_sentiment"] = (
    verified["true_sentiment"].fillna("").astype(str).str.strip().str.lower()
)

verified = verified[
    verified["text"].ne("")
    & verified["true_sentiment"].isin(LABELS)
].drop_duplicates(subset=["text"])

production_ready = (
    trigger.get("dataset_drift", False)
    and trigger.get("production_samples", 0) >= MIN_SAMPLES
)

labels_ready = (
    len(verified) >= MIN_SAMPLES
    and set(verified["true_sentiment"].unique()) == LABELS
    and verified["true_sentiment"].value_counts().min() >= 2
)

decision = "WAITING_FOR_MORE_DATA"

if not trigger.get("dataset_drift", False):
    decision = "NO_DRIFT"

elif not production_ready:
    decision = "WAITING_FOR_MORE_DATA"

elif not labels_ready:
    decision = "WAITING_FOR_VERIFIED_LABELS"

else:
    previous_state = {}
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as file:
            previous_state = json.load(file)

    if previous_state.get("status") == "CANDIDATE_CREATED_REQUIRES_EVALUATION":
        decision = "WAITING_FOR_CANDIDATE_EVALUATION"

    else:
        try:
            with open(LOCK_FILE, "x", encoding="utf-8") as lock:
                lock.write("Retraining in progress")

            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "retrain.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            print(result.stdout)

            if result.returncode != 0:
                print(result.stderr)
                decision = "RETRAINING_FAILED"
            else:
                decision = "CANDIDATE_CREATED_REQUIRES_EVALUATION"

            with open(STATE_FILE, "w", encoding="utf-8") as file:
                json.dump({
                    "status": decision,
                    "candidate_evaluation_required": True,
                    "production_promotion": False
                }, file, indent=2)

        except FileExistsError:
            decision = "RETRAINING_ALREADY_RUNNING"

        finally:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()

trigger.update({
    "decision": decision,
    "verified_examples": len(verified),
    "automatic_retraining_enabled": True,
    "automatic_production_promotion": False,
})

with open(TRIGGER_FILE, "w", encoding="utf-8") as file:
    json.dump(trigger, file, indent=2)

print("\nAutomatic retraining controller")
print("Drift detected:", trigger.get("dataset_drift"))
print("Production samples:", trigger.get("production_samples"))
print("Human-verified examples:", len(verified))
print("Decision:", decision)
print("Production promotion: DISABLED")
