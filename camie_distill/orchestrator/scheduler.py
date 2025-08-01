from __future__ import annotations
import json, os, shutil, subprocess, time, yaml
from pathlib import Path

CFG       = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
GPU0, GPU1 = CFG["gpus"]["trainer"], CFG["gpus"]["label_gen"]

INIT_THR: float = CFG["initial_confidence"]
MIN_THR:  float = CFG["min_confidence"]

# ───────────────── helper wrappers ────────────────────────────────
def _env(gpu: int) -> dict[str, str]:
    # Inherit the current environment and set the visible CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env

def build_dataset(repo: str, thr: float, out_dir: Path):
    cmd = [
        "camie-build-dataset",
        "--input-dir",  CFG["raw_data_root"],
        "--output-dir", str(out_dir),
        "--model-repo", repo,
        "--confidence-threshold", str(thr),
        "--hf-token",   CFG["hf_token"],
        "--device",     "cuda",
        # NOTE: Add any other required args for build_dataset here
    ]
    # In the updated dataset_builder, metadata.json is now created in the output_dir
    # We must download it from the teacher repo first if it is a HF repo string
    if not Path(repo).exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo, filename="model/metadata.json", local_dir=out_dir,
                        local_dir_use_symlinks=False, token=CFG["hf_token"])
    else: # If teacher is a local path, copy its metadata
        shutil.copy2(Path(repo) / "model/metadata.json", out_dir / "metadata.json")

    return subprocess.run(cmd, env=_env(GPU1), check=True)

def train_student(csv_dir: Path, tag_meta: Path, out_dir: Path):
    cmd = [
        "camie-train-student",
        "--csv-path", str(csv_dir / "train_dataset.csv"),
        "--img-root", str(csv_dir / "images"),
        "--tag-meta", str(tag_meta),
        "--output-dir", str(out_dir),
        # NOTE: Add any other required args for train_student here
    ]
    return subprocess.run(cmd, env=_env(GPU0), check=True)

# ───────────────── orchestrator main loop ─────────────────────────
def main() -> None:
    Path(CFG["teachers_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CFG["students_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CFG["datasets_dir"]).mkdir(parents=True, exist_ok=True)

    # bootstrap teacher
    teachers = sorted(Path(CFG["teachers_dir"]).glob("teacher_v*"))
    if teachers:
        teacher = teachers[-1]
        version = int(teacher.name.split("_v")[-1])
    else:
        teacher = Path("Camais03/camie-tagger")   # HF repo string
        version = 0

    curr_thr        = INIT_THR
    win_streak      = 0

    while True:
        print(f"--- Starting cycle for v{version} | Teacher: {teacher} | Threshold: {curr_thr:.2f} ---")
        ds_dir = Path(CFG["datasets_dir"]) / f"dataset_v{version}"
        build_dataset(str(teacher), curr_thr, ds_dir)

        stu_dir = Path(CFG["students_dir"]) / f"student_v{version}"
        # The metadata is now in the dataset directory
        train_student(ds_dir, ds_dir / "metadata.json", stu_dir)

        # ── promotion check ───────────────────────────────────────
        try:
            # For HF repo, metrics file needs to be downloaded or assumed not present
            if not Path(teacher).exists():
                 # For the initial bootstrap, we assume no prior metrics and auto-win
                 print("ℹ️  Initial teacher is HF repo, cannot read metrics. Assuming first run is a win.")
                 t_met = {"micro_f1": 0.0, "macro_f1": 0.0}
            else:
                t_met = json.loads((Path(teacher) / "metrics.jsonl").read_text().splitlines()[-1])
            s_met = json.loads((stu_dir / "metrics.jsonl").read_text().splitlines()[-1])
        except (FileNotFoundError, IndexError):
            print("⚠️  metrics.jsonl missing or empty – skipping evaluation.")
            time.sleep(60)
            continue

        better = all(
            (s_met[k] - t_met.get(k, 0)) >= CFG["promotion_gap"][k] for k in ("micro_f1", "macro_f1")
        )

        if better:
            win_streak += 1
            print(f"🟢 Student win {win_streak}/{CFG['promotion_patience']} (μF1: {s_met['micro_f1']:.4f} > {t_met['micro_f1']:.4f}, MF1: {s_met['macro_f1']:.4f} > {t_met['macro_f1']:.4f})")
        else:
            win_streak = 0
            print(f"🔴 Student fell short (μF1: {s_met['micro_f1']:.4f} vs {t_met['micro_f1']:.4f}, MF1: {s_met['macro_f1']:.4f} vs {t_met['macro_f1']:.4f})")

        if win_streak >= CFG["promotion_patience"]:
            # promote & lower threshold
            version  += 1
            new_teacher = Path(CFG["teachers_dir"]) / f"teacher_v{version}"
            shutil.copytree(stu_dir, new_teacher, dirs_exist_ok=True)
            teacher   = new_teacher
            curr_thr  = max(curr_thr - 0.10, MIN_THR)
            win_streak = 0
            print(f"🎉 Promoted to v{version}. Next threshold: {curr_thr:.2f}")

        # small cool‑down to avoid busy‑spin
        time.sleep(300)      # 5 min

if __name__ == "__main__":
    main()