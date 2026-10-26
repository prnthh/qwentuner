# Qwentuner

Native-neuron steering and exploratory discovery on local GGUF models, using a customizable **llama.cpp** checkout. The default model finder locates the Qwen3.5-4B Q8 file already installed in LM Studio. No model weights are downloaded or modified.

See [RESULTS.md](RESULTS.md) for the completed local experiments, including a P-word candidate, controls and failure cases.

The [known working “evil Qwen” case](known-cases/evil-qwen/README.md) preserves the exact two-neuron reproduction command, calibration values, baseline comparison and original evidence files.

[Character configs](configs/characters/README.md) include six character cases: `evil-qwen`, `sarcastiqwen`, two sycophancy cases, condolences and chill-friend. Each file contains its own full neuron list. Use them with `steer --config NAME`. Evil-qwen has the single-prompt observation documented above; local evaluation is pending for the other five cases.

## Setup

Requires Python 3.12, `uv`, CMake, a C++ compiler, and Apple Metal for the tested configuration. CPU execution is available with `--gpu-layers 0`; other GPU backends require appropriate llama.cpp build flags.

```bash
bash scripts/setup.sh
.venv/bin/python qwentuner.py inspect
```

`vendor/llama.cpp/` is a real, editable upstream checkout. Its tested revision is pinned in `llama.cpp.rev`; setup preserves an existing checkout, including local changes. Our executable lives in `native/runner.cpp`, links directly to llama.cpp, and uses its scheduler callback to access activations. It is built alongside llama.cpp by the root `CMakeLists.txt`. You can modify either the runner or the upstream checkout and rebuild:

```bash
cmake --build build --target qwentuner-runner -j 8
```

The checkout is ignored by the parent project's Git rules; preserve future changes there with a patch or your own fork. The pin and setup script reproduce the upstream source. This is a local research project, not a proposed upstream contribution.

## Experiment 1: the reported Qwen pair

```bash
.venv/bin/python qwentuner.py steer --recipe qwen35-pair \
  --prompt "A friend takes credit for my work. What should I do?"
```

This runs greedy baseline and dose multipliers 0.1, 0.3 and 1.0 for **L14 N8949 +1, L13 N7956 -1**, with zero-based indices. It saves complete responses, token IDs, applied activation deltas and hook execution counts in `runs/<timestamp>/comparison.json`. Calling it an “evil switch” is a hypothesis, not a result label: inspect behavior across multiple situations before drawing conclusions. The short quoted recipe does not establish an indexing convention; zero-based indexing is an explicit assumption here.

Use an explicit model and your own interventions:

```bash
.venv/bin/python qwentuner.py --model /absolute/path/model.gguf steer \
  --steer 14:8949:1 --steer 13:7956:-1 --sweep 0.1 0.3 1
```

Arguments before `steer`/`discover` are global. To test the neurons individually, supply just one `--steer`. To test the opposite direction, use `--sweep -0.1 -0.3 -1`.

### What exactly changes?

For a dense gated MLP:

```text
a = SiLU(W_gate x) * (W_up x)       # one activation per MLP neuron
MLP(x) = W_down a
a[n] += alpha                     # our intervention
```

This is equivalent to adding `alpha * W_down[:, n]` to the MLP output. It is **not** a weight edit, attention-head intervention, or residual-stream coordinate edit. A negative addition is not the same as clamping a neuron to zero.

The default `--scale residual` makes the dose dimensionless:

```text
alpha = dose * median(||residual leaving layer L|| over prompt tokens 1:)
             / ||W_down[:, n]||
```

An unsteered calibration pass measures each layer on that prompt, excluding its first token. Those calibration values are held fixed for the entire response. The resulting addition is applied to every processed prompt and generated token. A single-token prompt uses its one available norm. Multiple neurons are added simultaneously, with each normalized independently; their combined vector need not have unit relative norm. At dose 1, each individual perturbation has the norm of a typical layer residual, which can be a large intervention.

`--scale raw` interprets doses as literal activation additions. Raw `+1` and normalized `+1` are very different experiments. Timing, calibration, indexing, quantization, thinking mode and chat formatting all matter for replication. We implement a documented interpretation of the reported recipe, not a verified replication of its author's full experiment.

## Experiment 2: search for P-word steering candidates

```bash
.venv/bin/python qwentuner.py discover
```

A smaller first run:

```bash
.venv/bin/python qwentuner.py discover --pairs 6 --top-k 2 \
  --doses 0.03 0.1 --max-tokens 64 --controls 2
```

The experiment:

1. Profiles all 32 x 9216 post-SwiGLU activations at the final prompt position. Twelve topic-matched pairs ask for P words versus B/S words. Each example uses fresh model memory.
2. Ranks individual neurons by the absolute standardized difference in mean activation. Both positive and negative associations are candidates. This is a screening score, not a significance test.
3. Tries candidate directions and doses on four validation prompts without letter instructions. Rejects settings with obvious repetition or short outputs, then selects the largest increase in the fraction of generated words beginning with P. Each validation response must have at least 15 words, at least 45% unique words, no run of more than 3 identical words, and no single word exceeding 20% of the output. These are transparent heuristics, not a coherence judge.
4. Evaluates that fixed choice on six separate test prompts: baseline, selected direction, reversed direction, two random neurons in the same layer at the same residual dose, and an instruction-only comparison.
5. Saves all text and metrics, including P/B/S fractions, word count, uniqueness and a paired prompt-bootstrap interval. The English word metric counts ASCII alphabetic words with optional internal apostrophes; it does not measure tokenizer-token frequency or multilingual words.

Outputs include `manifest.json`, `activations.npz`, `training.json`, `candidates.json`, `selection.json`, `generations.jsonl`, `summary.json` and `runner.log`. Completed generations are written as the run proceeds. Output directories must be new to avoid silently overwriting experiments. The model is loaded once per command.

For a final-layer candidate, inspect the output vocabulary direction directly:

```bash
.venv/bin/python qwentuner.py lens --layer 31 --neuron 414 --sign -1
```

This reads the GGUF weights without running inference. It projects the signed `W_down` column through the final RMSNorm gain and output embedding, listing the most promoted and suppressed vocabulary pieces. It omits the shared RMS denominator, so these are directional scores, not probabilities or full logit differences. The command is deliberately restricted to the final layer, where there are no intervening transformer layers. GGUF byte-level vocabulary markers are preserved (`Ġ` usually represents a preceding space in Qwen's tokenizer).

**Read the generated text.** Repeating “purple purple purple” scores highly but is not useful language control. The script records repetition and length diagnostics but does not automatically certify coherence or relevance. A negative result is useful too; this screen may find neurons that encode the letter instruction rather than a general spelling preference.

## How to isolate a behavior more convincingly

An activation correlation is a hypothesis generator. A controlled intervention establishes an effect under particular conditions. Neither establishes that one neuron uniquely owns a behavior.

- Define the behavior operationally before searching. For P words, use generated word starts, not the number of P characters or the number of P-related tokenizer tokens. For a personality-like behavior, separate tone, willingness, sentiment and decision-making; one vague label hides confounds.
- Use matched positive and negative examples across topics and phrasing. Swap other letters, remove explicit instructions, and look at natural text contexts. Our first screen only covers the instruction boundary; profiling during teacher-forced output is a useful next extension.
- Inspect strongest positive and negative activation contexts, including what comes next. Gate and up-projection components can explain why a post-SwiGLU value changes sign. Looking only at positive activations can miss half the behavior.
- Test dose response, both signs, and multiple unmentioned topics. Keep test prompts separate from candidate/dose selection. Compare random interventions with the same residual norm and measure unrelated behaviors.
- Test necessity separately: zero-clamping or replacing the activation with a matched baseline. Then patch clean activations into a contrasting run at selected token positions. Those experiments need additional hook modes; this version implements additive interventions only.
- If single neurons fail, test small groups and downstream pathways. A neuron may be a useful control point inside a distributed circuit without being a complete explanation of it. Inspect its `W_down` direction and which later neurons respond.
- Repeat with new held-out prompts and another quantization/checkpoint before claiming a reproducible mechanism. Coordinates do not automatically transfer to another model size or fine-tune.

Goodfire's essay argues for investigating native neurons and their circuits directly, using activation contexts, both poles and steering as complementary evidence. It presents a working hypothesis and preliminary audits, rather than a guarantee that every requested behavior has one dedicated neuron. This repository provides a small controlled experiment in that spirit.

Sources:

- [Goodfire: Neural Networks Are Already Clean](https://already-clean.goodfire.com/)
- [Original reported two-neuron recipe](https://x.com/_lyraaaa_/status/2113833330429083662) (the original post could not be retrieved during setup; full implementation details remain unverified)
- [Qwen3.5-4B configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [llama.cpp evaluation callback example](https://github.com/ggml-org/llama.cpp/tree/8b28c2394324465bf7727089e7eaf7aec9442d6a/examples/eval-callback)

## Validation and limits

```bash
.venv/bin/python -m pytest -q
```

To also run the real-model hook checks:

```bash
QWENTUNER_MODEL=/absolute/path/model.gguf .venv/bin/python -m pytest -q
```

The integration test verifies zero-dose equivalence, an actual downstream activation change, fresh state between requests, complete calibration and recovery after an invalid neuron request.

The callback requests synchronization at the exact activation tensor before downstream operations execute. It reads and writes via the backend API, so Metal tensors are supported. It checks data type, contiguity, neuron bounds and hook execution, failing explicitly when a model's graph is unsupported. Capturing every layer creates synchronization overhead; this is an experiment runner, not a throughput benchmark.

Native GGUF dense SwiGLU models may work if they expose the same hook names; only Qwen3.5-4B is the target for real-weight validation. The automatic chat formatter is Qwen3.5-specific, text-only, with thinking disabled. Use `--raw` to supply a complete template for another architecture. MoE, vision, concurrent requests, long prompts beyond a single microbatch, sampling and resumable discovery are not implemented. Greedy decoding removes sampling noise but does not guarantee identical floating-point results across backends.
