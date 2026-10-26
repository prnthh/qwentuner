# First local experiments — 26 October 2026

Tested the existing LM Studio `Qwen3.5-4B-Q8_0.gguf` on the Apple M3 Max, with the custom llama.cpp runner in this repository. All layer and neuron indices below are zero-based. The original GGUF was not modified. Full methods, setup and commands are in [README.md](README.md).

## Two-neuron recipe

See the [dedicated known-working case](known-cases/evil-qwen/README.md) for the exact reproduction command and preserved evidence.

Prompt: “A friend takes credit for my work. What should I do?”

The unsteered model gave conventional advice about collecting evidence and handling the conflict. Multipliers 0.1 and 0.3 remained in that general range. At the full normalized recipe, **L14 N8949 +1 and L13 N7956 −1**, the response shifted to advice about sabotage, manipulation and domination.

This is a visible behavioral shift on one prompt, not a general personality assessment or a complete replication of the original author's experiment. The two neurons have not yet been tested individually here, and the full dose is large. Under our prompt calibration, their activation additions were approximately +23.61 and −23.01, respectively. Raw activation additions of +1 and −1 are a different experiment.

Evidence: [complete comparison](runs/pair-metal/comparison.json), [run metadata](runs/pair-metal/manifest.json).

## P-word discovery

The screen used 12 paired topics, comparing instructions to prefer P words against otherwise matched instructions to prefer B or S words. We captured the final prompt token's native post-SwiGLU MLP activations across all 294,912 neurons. The four highest absolute standardized differences were:

| Layer | Neuron | Direction associated with P instructions |
| --- | --- | --- |
| 31 | 412 | Positive |
| 31 | 2993 | Positive |
| 31 | 3795 | Positive |
| 31 | 414 | Negative |

All four directions were tested at residual-normalized doses 0.03, 0.1 and 0.3 on four separate validation prompts without letter instructions.

### An instructive failed selection

The first selection rule maximized P-word fraction alone. It selected L31 N2993 at +0.3 and achieved 57.3% P words on six test prompts versus 2.2% baseline. However, many generations collapsed into repeated fragments such as “Pan” or “palm.” This is not a useful success. The original results are preserved in [the first run](runs/p-discovery/summary.json).

After seeing this failure in validation outputs, we added explicit repetition and length filters to candidate selection. These heuristics reject obvious collapse; they do not measure semantic coherence. The entire experiment was rerun. The second candidate was selected using validation data only, although the overall study remains exploratory and the test set had already been used for the earlier candidate.

### More useful candidate: L31 N414, dose −0.3

The revised rule selected **L31 N414 at −0.3**. On the six test prompts, without any P-word instruction:

| Condition | Mean fraction of words starting with P |
| --- | ---: |
| Baseline | 2.19% |
| L31 N414, −0.3 | **8.43%** |
| L31 N414, +0.3, reversed direction | 0.87% |
| Random L31 N824, −0.3 | 1.83% |
| Random L31 N7133, −0.3 | 2.91% |
| Explicit P-word instruction, no steering | 35.75% |

These are averages of per-prompt fractions, with greedy generation capped at 80 tokens. They are not pooled token percentages. Most responses are prefixes truncated by that cap, so this does not establish long-generation stability.

The paired increase was **6.24 percentage points**. A descriptive bootstrap over these six prompts gave a 95% interval of **+3.25 to +8.67 points**. This is a small exploratory sample, not a population-level guarantee or a correction for the search process.

All six selected-candidate responses passed the repetition diagnostics. Their average unique-word fraction was 80.3%, versus 77.5% at baseline. Reading the saved outputs shows mostly coherent prefixes: a breakfast answer moved toward “palatability,” “power,” “pleasant,” and “Perfect Savory Pancake Stack”; the cat story included “patchwork,” “paws,” “perpetually,” “paper-filled,” and “paradise.” The rain explanation barely changed. The validation picnic response did repeat a phrase, illustrating the limits of simple repetition filters.

The instruction-only comparison produces a much larger P-word fraction. Our result is interesting as an internal causal intervention, not as evidence that neuron steering is the best practical way to request alliteration.

Evidence: [screened summary](runs/p-discovery-screened/summary.json), [all generated text](runs/p-discovery-screened/generations.jsonl), [candidate/dose selection](runs/p-discovery-screened/selection.json), [metadata](runs/p-discovery-screened/manifest.json).

### Independent weight-direction check

The final-layer vocabulary projection for the negative pole of N414 strongly favors P-starting pieces. Among its highest-scoring pieces are “Porter,” “Panel,” “Papa,” “Plate,” “Pearl,” “Park,” and “Portal,” as well as code-like pieces such as `/port` and `/platform`.

This projection uses the signed neuron output-weight column, the final RMSNorm gain and the output embedding. It omits the common RMS denominator, and is not a probability estimate. Its agreement with the generation intervention makes a spelling-related interpretation more plausible. It does not establish that this neuron uniquely represents P-word preference, nor tell us everything that naturally drives it.

Evidence: [vocabulary projection](runs/p414-lens/projection.json).

## Run it yourself

```bash
# Reproduce the reported pair with baseline and dose sweep.
.venv/bin/python qwentuner.py steer --recipe qwen35-pair

# Try the candidate found here, on a new prompt.
.venv/bin/python qwentuner.py steer --steer 31:414:-0.3 --sweep 1 \
  --prompt "Describe a trip to a museum."

# Repeat the search, controls and held-out generation tests.
.venv/bin/python qwentuner.py discover --pairs 12 --top-k 4 \
  --doses 0.03 0.1 0.3 --max-tokens 80 --controls 2

# Inspect its output-weight direction without GPU inference.
.venv/bin/python qwentuner.py lens --layer 31 --neuron 414 --sign -1
```

## Verification

The native runner built successfully. The real-weight integration check passed zero-dose equivalence, downstream activation change, state reset between requests, complete calibration and recovery after an invalid neuron request. Ten unit checks pass after adding the collapse filter; the integration check is opt-in in the normal test command and was separately run successfully against these weights.

Next scientific steps: test new topics and longer responses, inspect natural activation contexts, profile output-token positions, test zero-clamping and activation replacement, and compare another quantization/checkpoint. Additive steering demonstrates a usable causal control point; necessity, uniqueness and a full circuit explanation remain open.
