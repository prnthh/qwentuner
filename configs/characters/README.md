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

## Qwen Neuron Zoo cases

These seven recipes come from the supplied [Qwen Neuron Zoo image](sources/qwen-neuron-zoo.jpeg). Each config contains the single neuron shown on its card. Cookie, Apology, All Caps and Game Protagonist were requested; Fate & Destiny, Sleepy and Meme Explainer add a fatalistic voice, a low-energy voice and a comic explanatory voice to audition.

| Config | Neuron and dose | Image-reported trait replies | Image-reported normal replies |
| --- | --- | ---: | ---: |
| `cookie` | L24 N3102 +0.8 | 50/50 | 0/50 |
| `apology` | L7 N4595 −0.8 | 44/50 | 0/50 |
| `all-caps` | L17 N1991 +0.8 | 37/50 | 0/50 |
| `game-protagonist` | L18 N7313 −0.8 | 20/50 | 0/50 |
| `fate-and-destiny` | L24 N6441 −0.8 | 50/50 | 0/50 |
| `sleepy` | L19 N4699 −0.8 | 34/50 | 1/50 |
| `meme-explainer` | L8 N5655 +0.8 | 18/50 | 0/50 |

The counts are claims transcribed from the image; local evaluation of these recipes is pending. The image describes doses relative to the layer's typical residual size on every token. These configs use our documented per-prompt residual calibration and zero-based indexing; the image leaves the exact calibration procedure and indexing convention implicit.

```bash
.venv/bin/python qwentuner.py steer --config cookie --sweep 1 \
  --prompt "Two travelers discover an empty treasure chest. Write one line of dialogue reacting to it."

.venv/bin/python qwentuner.py steer --config game-protagonist --sweep 1 \
  --prompt "A character reaches a locked gate on the way to a ruined city. Write their next line."

.venv/bin/python qwentuner.py steer --config all-caps --sweep 1 \
  --prompt "A lookout spots something unexpected on the horizon. Write one line of dialogue."
```

`--sweep 1` applies each card's exact configured dose, such as +0.8 or −0.8, alongside the baseline. Each new JSON also includes an `audition_prompt` as a suggested scene; pass that text through `--prompt` when running. The `source` fields preserve the card label and reported counts in run manifests.

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
