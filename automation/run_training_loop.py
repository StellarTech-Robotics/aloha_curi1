import argparse
import subprocess
import time
from pathlib import Path
from typing import Tuple


def load_stats(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0

    runs = 0
    steps = 0
    try:
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "runs":
                runs = int(value)
            elif key == "steps":
                steps = int(value)
    except ValueError:
        return 0, 0

    return runs, steps


def save_stats(path: Path, runs: int, steps: int) -> None:
    path.write_text(f"runs={runs}\nsteps={steps}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="重复执行 imitate_episodes 训练命令并累计成功次数。"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="重复执行的次数；缺省为无限循环直到手动停止或失败。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        help="每次成功运行后再次启动前的等待秒数。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_runs = args.runs
    pause_seconds = max(args.sleep, 0.0)

    repo_root = Path(__file__).resolve().parents[1]
    stats_file = repo_root / "training_stats.txt"

    base_command = [
        "python3",
        "imitate_episodes.py",
        "--task_name",
        "sim_transfer_cube_scripted",
        "--ckpt_dir",
        "/mnt/d/mycodes/act++/curi1/checkpoints/transfer_cube",
        "--policy_class",
        "ACT",
        "--kl_weight",
        "10",
        "--chunk_size",
        "20",
        "--hidden_dim",
        "512",
        "--batch_size",
        "1",
        "--dim_feedforward",
        "3200",
        "--lr",
        "1e-5",
        "--seed",
        "0",
        "--resume_ckpt_path",
        "/mnt/d/mycodes/act++/curi1/checkpoints/transfer_cube/policy_last.ckpt",
        "--num_steps",
        "500",
    ]

    total_runs, total_steps = load_stats(stats_file)
    print(f"[trainer] starting loop (completed runs so far: {total_runs}, total steps: {total_steps})")
    if target_runs is not None:
        print(f"[trainer] target runs this session: {target_runs}")

    try:
        session_completed = 0
        while True:
            print("[trainer] launching training run...")
            process = subprocess.run(base_command, cwd=repo_root, check=False)

            if process.returncode == 0:
                try:
                    num_steps = int(base_command[-1])
                except ValueError:
                    num_steps = 0
                total_runs += 1
                total_steps += num_steps
                save_stats(stats_file, total_runs, total_steps)
                session_completed += 1
                print(
                    f"[trainer] run finished successfully "
                    f"(session runs: {session_completed}, total runs: {total_runs}, total steps: {total_steps})"
                )
            else:
                print(f"[trainer] run failed with return code {process.returncode}, stopping loop.")
                break

            if target_runs is not None and session_completed >= target_runs:
                print("[trainer] session target reached, exiting loop.")
                break

            if pause_seconds > 0:
                time.sleep(pause_seconds)
    except KeyboardInterrupt:
        print("\n[trainer] interrupted by user, exiting.")


if __name__ == "__main__":
    main()
