"""Nov 2: data pushes must survive a remote that moved while the working tree is dirty."""
import subprocess
from pathlib import Path

import yaml

from edge_engine import k1, m1_live

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd, cwd):
    return subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True).stdout


def setup(tmp_path):
    """A data clone with a tracked log, behind a remote another job pushed to."""
    remote, data, other = tmp_path / "r.git", tmp_path / "data", tmp_path / "other"
    sh(f"git init -q --bare -b data {remote}", tmp_path)
    for d in (data, other):
        sh(f"git clone -q {remote} {d}", tmp_path)
        sh("git config user.name t; git config user.email t@t; git checkout -q -B data", d)
    (data / "log.jsonl").write_text("a\n")
    sh("git add -A; git commit -qm base; git push -q origin data", data)
    sh("git pull -q origin data", other)
    (other / "scoreboard.json").write_text("{}")
    sh("git add -A; git commit -qm scoreboard; git push -q origin data", other)
    (data / "log.jsonl").write_text("a\nb\n")        # unstaged, as during a monitor run
    return remote, data


def on_remote(remote, path):
    return path in sh("git ls-tree -r --name-only data", remote)


def test_forecast_checkpoint_pushes_with_dirty_tree(tmp_path):
    remote, data = setup(tmp_path)
    m1_live.write_and_push(str(data), "2026-11-02", [{"player": "x"}])
    assert on_remote(remote, "m1/forecasts/2026-11-02.jsonl")
    assert (data / "log.jsonl").read_text() == "a\nb\n"


def test_k1_checkpoint_pushes_with_dirty_tree(tmp_path):
    remote, data = setup(tmp_path)
    k1.write_and_push(str(data), "2026-11-02", [{"ticker": "x"}])
    assert on_remote(remote, "k1/quotes/2026-11-02.jsonl")


def commit_step():
    wf = yaml.safe_load((ROOT / ".github/workflows/nba-monitor.yml").read_text())
    step = next(s for s in wf["jobs"]["monitor"]["steps"] if s.get("name") == "Commit data")
    assert "pkill -f edge_engine.monitor" in step["run"]
    return step["run"].replace("pkill -f edge_engine.monitor || true", "").replace("sleep 2", "").replace("sleep $((i * 5))", "")


def test_commit_step_pushes_after_rejection(tmp_path):
    remote, data = setup(tmp_path)
    (tmp_path / "step.sh").write_text(commit_step().replace("cd data", f"cd {data}"))
    sh("bash -e step.sh", tmp_path)
    assert on_remote(remote, "log.jsonl")
    assert "b" in sh("git show data:log.jsonl", remote)


def test_commit_step_fails_visibly_when_push_never_lands(tmp_path):
    remote, data = setup(tmp_path)
    sh("git remote set-url origin /nonexistent", data)
    (tmp_path / "step.sh").write_text(commit_step().replace("cd data", f"cd {data}"))
    assert subprocess.run("bash -e step.sh", shell=True, cwd=tmp_path, capture_output=True).returncode == 1
