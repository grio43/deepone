#!/usr/bin/env python3
"""
Kick off or resume the whole pipeline.
"""
import yaml, subprocess, shutil, json, time, itertools
from pathlib import Path

CFG = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
GPU0, GPU1 = CFG["gpus"]["trainer"], CFG["gpus"]["label_gen"]

INIT_THR = float(CFG.get("initial_confidence", 0.65))
MIN_THR  = float(CFG.get("min_confidence",   0.35))

def make_ds(teacher_repo, thr, out_dir):
    cmd = [
        "camie-build-dataset",
        "--input-dir",  str(CFG["raw_data_root"]),              # streaming handled inside
        "--output-dir", str(out_dir),
        "--model-repo", teacher_repo,
        "--confidence-threshold", str(thr),
        "--hf-token",  CFG["hf_token"],
        "--device",    "cuda",
    ]
    env = {"CUDA_VISIBLE_DEVICES": str(GPU1)}
    return subprocess.Popen(cmd, env=env)

def train_student(csv_dir, tag_meta, out_dir):
   cmd = [
        "camie-train-student",
        "--csv-path", str(csv_dir / "train_dataset.csv"),
        "--img-root", str(csv_dir / "images"),
        "--tag-meta", str(tag_meta),
        "--output-dir", str(out_dir),
    ]
    env = {"CUDA_VISIBLE_DEVICES": str(GPU0)}
    return subprocess.Popen(cmd, env=env)

def main():
    teachers = sorted(Path(CFG["teachers_dir"]).glob("teacher_v*"))
    teacher = teachers[-1] if teachers else Path("Camais03/camie-tagger")
    version = len(teachers)

    curr_thr = INIT_THR
    if wins:
        new_teacher = Path(CFG["teachers_dir"]) / f"teacher_v{version+1}"
        shutil.copytree(stu_dir, new_teacher)
        teacher, version = new_teacher, version+1
        ds_proc.wait()

        stu_dir = Path(CFG["students_dir"]) / f"student_v{version}"
        stu_proc = train_student(ds_dir, teacher/"model/metadata.json", stu_dir)
        stu_proc.wait()

        # ── promotion logic ─────────────────────────────────────────
        try:
            met_teacher = json.loads((teacher/"metrics.json").read_text())
            met_student = json.loads((stu_dir/"metrics.json").read_text())
        except FileNotFoundError:
            print("⚠️  metrics.json missing; skipping promotion check.")
            continue
        wins = sum(
            (met_student[k] - met_teacher[k]) >= CFG["promotion_gap"][k]
            for k in ("micro_f1", "macro_f1")
        ) == 2

        if wins:
            new_teacher = Path(CFG["teachers_dir"]) / f"teacher_v{version+1}"
            shutil.copytree(stu_dir, new_teacher)
            teacher, version = new_teacher, version+1
            curr_thr = max(curr_thr - 0.10, MIN_THR)
            # optional: upload to HF Hub
        else:
            # else: tighten sampling or lower LR, etc.
            pass

if __name__ == "__main__":
    main()
