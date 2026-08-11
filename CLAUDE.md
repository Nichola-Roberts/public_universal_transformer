# Working in this repo

This checkout lives on an **ephemeral RunPod GPU box**. When the pod is
terminated the whole disk is destroyed — anything not pushed to GitHub is gone.
Work accordingly.

## Everything worth keeping goes in the repo

- **Conversations**: `~/.claude/projects` is a symlink to `.claude-sessions/` in this
  repo. It's gitignored by request, so session transcripts are NOT backed up —
  they're lost when the pod is terminated, same as anything else outside the repo.
- **Logs**: write run/training/eval logs to `logs/`. Don't leave anything you
  care about in `/tmp` or elsewhere under `/root`.
- **This file**: keep CLAUDE.md in the repo and update it when you learn
  something about the project worth remembering next session.

## Commit and push often

- Commit and push after every meaningful chunk of work — assume the box can
  vanish without warning. Don't batch a whole session into one final push.
- `.claude-sessions/` and `logs/` are deliberately tracked; include them.
- `origin` is GitHub over SSH and is already authenticated.

## Environment

- The pod image ships CUDA-matched **torch preinstalled in the system Python**.
  Install into it — `uv pip install --system <pkg>` — and record new deps in
  `requirements.txt`.
- **Do not create a venv** (`uv venv`, `uv sync`, `python -m venv`). A fresh env
  doesn't inherit torch, so it re-downloads ~2.5GB of CUDA wheels on a box that
  is billed by the hour and thrown away afterwards, and can land on a build that
  doesn't match the driver.
- The GPU is billed by the hour, so don't leave it idle mid-task.
