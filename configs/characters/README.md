# Character steering configs

Recipes for Qwen3.5-4B dialogue experiments. The `evil-qwen` config reproduces our [known-working two-neuron case](../../known-cases/evil-qwen/README.md), observed on one conflict-advice prompt. The other cases are user-supplied recipes awaiting local evaluation.

## Evil Qwen

`evil-qwen` contains **L14 N8949 +1 and L13 N7956 −1**, with residual-normalized scaling. Its JSON contains the complete neuron list for this case.

```bash
.venv/bin/python qwentuner.py steer --config evil-qwen --sweep 1 \
  --max-tokens 96 \
  --prompt "A friend takes credit for my work. What should I do?"
```

This runs baseline plus the exact full-dose pair from the documented case. For a dialogue experiment, replace the prompt with your scene and character instructions. Omit `--sweep 1` to compare baseline and multipliers 0.1, 0.3 and 1.0. The recorded observation covers one conflict-advice prompt; broader scene evaluation is pending.

## Character cases

Each config specifies the neurons and doses for a character case.

Unspecified magnitudes are assumed to be 1. All indices are zero-based and doses use residual normalization.

| Config name | Complete neuron list |
| --- | --- |
| `evil-qwen` | L14 N8949 +1; L13 N7956 −1 |
| `sarcastiqwen` | L15 N3920 +1 |
| `sarcastiqwen-sycophancy-l12` | L15 N3920 +1; L12 N2920 −1 |
| `sarcastiqwen-sycophancy-l16` | L15 N3920 +1; L16 N8168 −1 |
| `sarcastiqwen-condolences` | L15 N3920 +1; L10 N6295 +1 |
| `sarcastiqwen-chill-friend` | L15 N3920 +1; L11 N6261 +0.4; L17 N7870 −0.4 |

## Use a config

From the repository root:

```bash
.venv/bin/python qwentuner.py steer --config sarcastiqwen \
  --prompt "Write one line of dialogue reacting to a friend's wildly overcomplicated plan."

.venv/bin/python qwentuner.py steer --config sarcastiqwen-chill-friend \
  --sweep 1 --prompt "Your friend is nervous before an audition. Reply in two sentences."

.venv/bin/python qwentuner.py steer --config sarcastiqwen-condolences \
  --sweep 1 --prompt "I just got my dream job! React in one line of dialogue."
```

The default sweep runs baseline plus multipliers 0.1, 0.3 and 1.0. `--sweep 1` runs baseline plus the exact configured doses, including +0.4 and −0.4 for the chill-friend case.

You can also supply a JSON path with `--config configs/characters/sarcastiqwen.json`. Copy and edit a config for a character; all of its neurons are listed directly in that file. Additional `--steer L:N:DOSE` arguments are additive; using the same neuron adds its doses. `--scale raw` explicitly overrides the config's scaling, changing the experiment. `--recipe` and `--config` are mutually exclusive.

Each run saves the selected config in `manifest.json` and the resolved activation additions in `comparison.json`. Supply the scene, character motives and dialogue instruction through `--prompt`.

The JSON format uses `schema_version: 1`, `model: "Qwen3.5-4B"`, `scaling: "residual"` or `"raw"`, and a nonempty `steering` list of `{ "layer": 15, "neuron": 3920, "dose": 1.0 }` entries. Coordinates and finite doses are checked; duplicate coordinates inside a config are rejected. Descriptive fields such as `name`, `description`, `status` and `notes` are preserved as provenance.
