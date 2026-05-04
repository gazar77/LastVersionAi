---
title: Angio Ai Service
emoji: 👀
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
license: apache-2.0
---
..
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Model weights (`models/`)

This repo lists two checkpoints: `models/stenosis/best_stenosis_model_final.pth` and `models/vessel/best_vessel_model_run2.pth`.

If those files are only **~100 bytes**, they are Git **LFS pointer stubs**, not the real weights, and GitHub LFS may still be missing the large objects (clones will not get working models). Put the **actual** PyTorch `.pth` files in those paths (same names), then run `git add models/`, commit, and push. The repo is configured so paths under `models/**/*.pth` are stored as **normal Git files** (not LFS), as long as each file is under GitHub’s per-file size limit.
