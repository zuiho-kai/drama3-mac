# Third-Party Notices

- **mlx-speech** (https://github.com/appautomaton/mlx-speech), MIT License,
  Copyright (c) 2026 AppAutomaton swarm of agents. `scripts/setup.sh` clones it at a pinned
  commit; the Drama 3 MLX port, the FP8→MLX converter and the generate script all come from it.
  `harness/convert4.py` wraps its converter.
- **Fish Audio Drama 3** weights (https://huggingface.co/fishaudio/drama-3) are released under the
  Fish Audio Community License. They are not included here; download them from Fish Audio.
- **mlx-whisper** (MIT) is used by `harness/quality.py` for transcription checks.
