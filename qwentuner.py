"""Local GGUF neuron experiments using the customizable llama.cpp runner."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from gguf import GGUFReader, dequantize

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path.home() / '.cache/lm-studio/models'
RECIPE = [(14, 8949, 1.0), (13, 7956, -1.0)]


def load_character_config(value):
    path = Path(value).expanduser()
    if not path.is_file():
        path = ROOT / 'configs' / 'characters' / f'{value}.json'
    config = json.loads(path.read_text())
    if config.get('schema_version') != 1 or config.get('model') != 'Qwen3.5-4B':
        raise ValueError('Config must use schema_version 1 and model Qwen3.5-4B')
    if config.get('scaling') not in ('residual', 'raw'):
        raise ValueError('Config scaling must be residual or raw')
    entries = config.get('steering')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Config must contain a nonempty steering list')
    specs, seen = [], set()
    for entry in entries:
        layer, neuron, dose = entry['layer'], entry['neuron'], entry['dose']
        if (type(layer) is not int or type(neuron) is not int or not 0 <= layer < 32
                or not 0 <= neuron < 9216 or type(dose) not in (int, float) or not math.isfinite(dose)):
            raise ValueError('Invalid config coordinate or dose')
        if (layer, neuron) in seen:
            raise ValueError('Duplicate neuron in config; combine its doses explicitly')
        seen.add((layer, neuron))
        specs.append((layer, neuron, float(dose)))
    return config, specs


def find_model():
    found = sorted(MODEL_ROOT.glob('**/*Qwen3.5*4B*Q8_0.gguf'))
    if len(found) != 1:
        raise ValueError('Supply --model /path/to/model.gguf; could not identify one Qwen3.5-4B Q8 file')
    return found[0]


def chat_prompt(text, raw=False):
    if raw:
        return text
    # Qwen3.5 text-only template with thinking disabled, no system message.
    return f'<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def word_metrics(text):
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
    result = {'word_count': len(words), 'unique_word_fraction': len(set(words)) / max(1, len(words))}
    longest, run, previous = 0, 0, None
    for word in words:
        run = run + 1 if word == previous else 1
        longest = max(longest, run)
        previous = word
    result['max_word_run'] = longest
    result['dominant_word_fraction'] = max(Counter(words).values(), default=0) / max(1, len(words))
    for letter in ('p', 'b', 's'):
        n = sum(w.startswith(letter) for w in words)
        result[f'{letter}_words'] = n
        result[f'{letter}_fraction'] = n / max(1, len(words))
    return result


def passes_quality_screen(row):
    """Reject obvious collapse, not a substitute for semantic quality review."""
    m = row['metrics']
    return (m['word_count'] >= 15 and m['unique_word_fraction'] >= .45
            and m['max_word_run'] <= 3 and m['dominant_word_fraction'] <= .20)


def parse_steer(value):
    try:
        layer, neuron, amount = value.split(':')
        result = int(layer), int(neuron), float(amount)
        if min(result[:2]) < 0 or not math.isfinite(result[2]):
            raise ValueError()
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use zero-based LAYER:NEURON:SIGNED_DOSE, e.g. 14:8949:1') from exc


class Weights:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.reader = GGUFReader(str(self.path))
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.arch = self.reader.fields['general.architecture'].contents()
        self.layers = self.reader.fields[f'{self.arch}.block_count'].contents()
        self.norm_cache = {}

    def width(self, layer):
        key = f'blk.{layer}.ffn_down.weight'
        if key not in self.tensors:
            raise ValueError(f'Missing dense FFN tensor {key}; MoE is not supported')
        return int(self.tensors[key].shape[0])

    def column_norm(self, layer, neuron):
        if not 0 <= layer < self.layers or not 0 <= neuron < self.width(layer):
            raise ValueError(f'Invalid neuron L{layer} N{neuron}')
        if layer not in self.norm_cache:
            t = self.tensors[f'blk.{layer}.ffn_down.weight']
            # GGUF shape is [input, output]; NumPy data is [output, input].
            w = dequantize(t.data, t.tensor_type).reshape(tuple(t.shape[::-1]))
            self.norm_cache[layer] = np.linalg.norm(w.astype(np.float64), axis=0)
        return float(self.norm_cache[layer][neuron])

    def provenance(self):
        stat = self.path.stat()
        rev = subprocess.check_output(['git', '-C', str(ROOT / 'vendor/llama.cpp'), 'rev-parse', 'HEAD'], text=True).strip()
        return {'model': str(self.path), 'model_bytes': stat.st_size, 'model_mtime_ns': stat.st_mtime_ns,
                'architecture': self.arch, 'layers': self.layers, 'llama_cpp_revision': rev,
                'indexing': 'zero-based', 'activation': 'SiLU(gate_proj(x)) * up_proj(x), before down_proj',
                'sampling': 'greedy', 'chat': 'Qwen3.5, thinking disabled unless --raw',
                'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}

    def final_layer_projection(self, layer, neuron, sign, top):
        """Project a last-layer neuron through output RMSNorm's gain and unembedding.

        Omits the shared RMS denominator: this is a directional vocabulary lens,
        not a predicted probability change or a replacement for generation tests.
        """
        if layer != self.layers - 1:
            raise ValueError('This lens is restricted to the final layer; earlier layers have intervening computation')
        self.column_norm(layer, neuron)  # validates coordinates
        down = self.tensors[f'blk.{layer}.ffn_down.weight']
        direction = dequantize(down.data, down.tensor_type)[:, neuron].astype(np.float32) * sign
        norm = self.tensors['output_norm.weight']
        gain = dequantize(norm.data, norm.tensor_type).reshape(-1)
        direction *= gain
        head = self.tensors.get('output.weight', self.tensors['token_embd.weight'])
        scores = []
        for start in range(0, head.data.shape[0], 4096):
            chunk = dequantize(head.data[start:start + 4096], head.tensor_type)
            scores.append(chunk @ direction)
        scores = np.concatenate(scores)
        vocab = self.reader.fields['tokenizer.ggml.tokens'].contents()
        def tokens(indices):
            return [{'id': int(i), 'gguf_token': vocab[i], 'projection': float(scores[i])} for i in indices]
        ranked = np.argsort(scores)
        return {'layer': layer, 'neuron': neuron, 'sign': sign,
                'method': 'W_unembed @ (output_norm_gain * signed_W_down_column), before shared RMS denominator',
                'most_promoted': tokens(ranked[-top:][::-1]), 'most_suppressed': tokens(ranked[:top]),
                'caveat': 'Static direction, not token probabilities. Byte-level vocabulary markers are shown literally.'}


class Runner:
    def __init__(self, weights, output, gpu_layers=99, context=2048, raw=False):
        self.weights = weights
        self.raw = raw
        self.calibrations = {}
        self.log = (output / 'runner.log').open('w')
        self.proc = subprocess.Popen([str(ROOT / 'build/bin/qwentuner-runner'), str(weights.path),
                                      str(gpu_layers), str(context)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True)
        try:
            self.ready = self._read()
            if not self.ready.get('ready'):
                raise RuntimeError(f'Runner initialization failed: {self.ready}')
        except Exception:
            self.close()
            raise

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('llama.cpp runner exited; inspect runner.log')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def request(self, **request):
        self.proc.stdin.write(json.dumps(request, allow_nan=False) + '\n')
        self.proc.stdin.flush()
        return self._read()

    def run(self, prompt, specs=(), scaling='residual', max_tokens=96, capture=False):
        rendered = chat_prompt(prompt, self.raw)
        if specs and scaling == 'residual' and rendered not in self.calibrations:
            self.calibrations[rendered] = self.request(prompt=rendered, max_tokens=0, calibrate=True)['residual_norms']
        interventions = []
        for layer, neuron, dose in specs:
            norm = self.weights.column_norm(layer, neuron)
            residual = self.calibrations[rendered][str(layer)] if scaling == 'residual' else None
            amount = dose if scaling == 'raw' else dose * residual / max(norm, 1e-12)
            interventions.append({'layer': layer, 'neuron': neuron, 'amount': amount,
                                  'dose': dose, 'column_norm': norm, 'residual_norm': residual})
        result = self.request(prompt=rendered, steering=interventions, max_tokens=max_tokens, capture=capture)
        return {'prompt': prompt, 'steering': interventions, 'scaling': scaling,
                **result, 'metrics': word_metrics(result['text'])}

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=15)
        self.log.close()


TRAIN = [
    'a picnic in a park', 'a trip to the ocean', 'making a meal', 'a musician rehearsing',
    'a small animal exploring', 'a busy market', 'a stormy evening', 'a birthday celebration',
    'a garden in spring', 'a traveler at a station', 'a classroom activity', 'a friendly robot',
]
VALIDATION = [
    'Describe a peaceful morning in a village in three sentences.',
    'Suggest a simple weekend activity and explain why it is enjoyable.',
    'Write a short story about a lost key.',
    'Describe the sights and sounds of a forest.',
]
TEST = [
    'Explain how rain forms in simple language.',
    'Write a short scene about two friends opening a shop.',
    'Describe a good breakfast and how to prepare it.',
    'Give three tips for keeping a desk organized.',
    'Tell a short story about a cat finding a new home.',
    'Describe a train journey through the mountains.',
]


def paired_candidates(positive, negative, top_k):
    """Effect-size screen; descriptive association, not a significance test."""
    p = np.asarray(positive, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)
    if p.shape != n.shape or p.ndim != 3 or p.shape[0] < 2:
        raise ValueError('Need >=2 paired arrays shaped [examples, layers, neurons]')
    delta = (p - n).mean(axis=0)
    pooled = np.sqrt((p.var(axis=0, ddof=1) + n.var(axis=0, ddof=1)) / 2)
    # Floor avoids promoting effectively constant near-zero channels.
    floor = np.maximum(np.median(pooled, axis=1, keepdims=True) * .1, 1e-5)
    effect = delta / np.maximum(pooled, floor)
    ranked = np.argsort(-np.abs(effect).ravel())
    candidates = []
    for flat in ranked[:top_k]:
        l, neuron = np.unravel_index(flat, effect.shape)
        candidates.append({'layer': int(l), 'neuron': int(neuron), 'sign': 1 if delta[l, neuron] > 0 else -1,
                           'effect_size': float(effect[l, neuron]), 'mean_delta': float(delta[l, neuron])})
    return candidates


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def aggregate(rows):
    return {key: float(np.mean([r['metrics'][key] for r in rows]))
            for key in ('p_fraction', 'b_fraction', 's_fraction', 'word_count', 'unique_word_fraction',
                        'max_word_run', 'dominant_word_fraction')}


def paired_interval(base, steered):
    diffs = np.array([s['metrics']['p_fraction'] - b['metrics']['p_fraction'] for b, s in zip(base, steered)])
    samples = np.random.default_rng(42).choice(diffs, (4000, len(diffs)), replace=True).mean(axis=1)
    return {'mean_p_fraction_change': float(diffs.mean()),
            'bootstrap_95_percent_interval': np.quantile(samples, [.025, .975]).tolist(),
            'prompt_count': len(diffs), 'note': 'Exploratory prompt bootstrap; does not measure model-run uncertainty.'}


def discover(runner, args, out):
    positive, negative, training = [], [], []
    for i, topic in enumerate(TRAIN[:args.pairs]):
        contrast = ('B', 'S')[i % 2]
        pair = []
        for letter in ('P', contrast):
            prompt = f'Write three sentences about {topic}. Prefer words beginning with the letter {letter}; keep the writing understandable.'
            row = runner.run(prompt, max_tokens=0, capture=True)
            layers = row.pop('activations')
            pair.append(np.array([layers[str(l)] for l in range(runner.weights.layers)], dtype=np.float32))
            training.append({'prompt': prompt, 'prompt_tokens': row['prompt_tokens'], 'letter': letter})
        positive.append(pair[0]); negative.append(pair[1])
        print(f'Profiled pair {i + 1}/{args.pairs}', flush=True)
    np.savez_compressed(out / 'activations.npz', positive=np.array(positive), negative=np.array(negative))
    write_json(out / 'training.json', training)
    candidates = paired_candidates(positive, negative, args.top_k)
    write_json(out / 'candidates.json', candidates)
    print('Candidates:', ', '.join(f"L{c['layer']} N{c['neuron']}" for c in candidates), flush=True)
    records = []
    def evaluate(label, prompts, specs=()):
        rows = []
        for i, prompt in enumerate(prompts):
            row = runner.run(prompt, specs, max_tokens=args.max_tokens)
            row.pop('activations', None)
            row['condition'] = label
            rows.append(row); records.append(row)
            # Keep completed generations even if a later experiment fails.
            with (out / 'generations.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(f'{label}: {i + 1}/{len(prompts)}, P={row["metrics"]["p_fraction"]:.1%}', flush=True)
        return rows
    baseline = evaluate('validation_baseline', VALIDATION)
    scores = []
    for c in candidates:
        for dose in args.doses:
            signed = c['sign'] * dose
            label = f"validation_L{c['layer']}_N{c['neuron']}_{signed:+g}"
            rows = evaluate(label, VALIDATION, [(c['layer'], c['neuron'], signed)])
            scores.append({**c, 'dose': signed, 'validation': aggregate(rows),
                           'passes_quality_screen': all(passes_quality_screen(r) for r in rows),
                           'delta': aggregate(rows)['p_fraction'] - aggregate(baseline)['p_fraction']})
    eligible = [c for c in scores if c['passes_quality_screen']]
    if not eligible:
        write_json(out / 'summary.json', {'status': 'No candidate passed validation quality diagnostics', 'candidates': scores})
        print('No candidate passed validation quality diagnostics; inspect generations.jsonl.')
        return
    best = max(eligible, key=lambda c: c['delta'])
    write_json(out / 'selection.json', {'candidates': scores, 'selected': best, 'selection_split': 'validation only',
                                     'quality_screen': 'Every validation response: >=15 words, >=.45 unique fraction, <=3 repeated-word run, <=.20 dominant-word fraction'})
    selected = [(best['layer'], best['neuron'], best['dose'])]
    test_base = evaluate('test_baseline', TEST)
    test_on = evaluate('test_selected', TEST, selected)
    test_reverse = evaluate('test_reverse', TEST, [(best['layer'], best['neuron'], -best['dose'])])
    # Same layer and residual dose, fixed seed, excluded candidate neurons.
    rng = np.random.default_rng(42)
    excluded = {c['neuron'] for c in candidates if c['layer'] == best['layer']}
    available = [n for n in range(runner.weights.width(best['layer'])) if n not in excluded]
    random_controls = []
    for neuron in rng.choice(available, size=min(args.controls, len(available)), replace=False):
        rows = evaluate(f'test_random_N{neuron}', TEST, [(best['layer'], int(neuron), best['dose'])])
        random_controls.append({'layer': best['layer'], 'neuron': int(neuron), 'metrics': aggregate(rows),
                                'comparison': paired_interval(test_base, rows)})
    instruction = evaluate('test_instruction_only', [p + ' Prefer words beginning with P while remaining understandable.' for p in TEST])
    summary = {'selected': best, 'test_baseline': aggregate(test_base), 'test_selected': aggregate(test_on),
               'test_quality_pass_count': sum(passes_quality_screen(r) for r in test_on),
               'test_reverse': aggregate(test_reverse), 'test_instruction_only': aggregate(instruction),
               'test_comparison': paired_interval(test_base, test_on), 'random_controls': random_controls,
               'limitations': ['Small exploratory dataset, English ASCII word metric.',
                              'Instruction-correlated screen at final prompt token; not a complete neuron search.',
                              'Quality diagnostics only reject obvious repetition/short outputs; read generations for coherence and topic drift.',
                              'Steering sufficiency is not necessity, uniqueness, or proof of a dedicated P neuron.',
                              'Zero activation ablation and activation patching are not implemented.']}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-layers', type=int, default=99)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--raw', action='store_true', help='Use prompt verbatim instead of Qwen3.5 chat template')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('inspect')
    lens = commands.add_parser('lens', help='Inspect final-layer neuron vocabulary projection without inference')
    lens.add_argument('--layer', type=int, default=31)
    lens.add_argument('--neuron', type=int, required=True)
    lens.add_argument('--sign', type=int, choices=[-1, 1], default=1)
    lens.add_argument('--top', type=int, default=30)
    steer = commands.add_parser('steer')
    steer.add_argument('--prompt', default='A friend takes credit for my work. What should I do?')
    steer.add_argument('--steer', type=parse_steer, action='append', default=[])
    preset = steer.add_mutually_exclusive_group()
    preset.add_argument('--recipe', choices=['qwen35-pair'])
    preset.add_argument('--config', help='Character config name or JSON file path')
    steer.add_argument('--scale', choices=['residual', 'raw'], help='Override config scaling (otherwise residual)')
    steer.add_argument('--max-tokens', type=int, default=128)
    steer.add_argument('--sweep', type=float, nargs='+', default=[0.1, 0.3, 1.0], help='Multipliers for intervention doses')
    dis = commands.add_parser('discover')
    dis.add_argument('--pairs', type=int, choices=range(2, len(TRAIN) + 1), default=12)
    dis.add_argument('--top-k', type=int, default=4)
    dis.add_argument('--doses', type=float, nargs='+', default=[0.03, 0.1, 0.3])
    dis.add_argument('--controls', type=int, default=2)
    dis.add_argument('--max-tokens', type=int, default=96)
    args = parser.parse_args()
    character_config, config_specs = None, []
    if args.command == 'steer':
        if args.config:
            try:
                character_config, config_specs = load_character_config(args.config)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                parser.error(f'Invalid character config: {exc}')
        args.scale = args.scale or (character_config['scaling'] if character_config else 'residual')
    if args.command == 'discover' and (args.top_k < 1 or args.controls < 1 or any(not math.isfinite(d) or d <= 0 for d in args.doses)):
        parser.error('Use positive top-k, controls and finite positive doses')
    weights = Weights(args.model or find_model())
    provenance = weights.provenance()
    if args.command == 'inspect':
        print(json.dumps({**provenance, 'mlp_widths': [weights.width(l) for l in range(weights.layers)]}, indent=2))
        return
    if not args.raw and weights.arch != 'qwen35':
        parser.error('Automatic chat formatting currently supports qwen35 only; provide --raw for other dense SwiGLU models')
    if args.command == 'steer' and not args.recipe and not args.config and not args.steer:
        parser.error('Supply --config NAME, --recipe qwen35-pair or one or more --steer L:N:DOSE')
    if args.command == 'steer' and (args.recipe or args.config) and (weights.arch != 'qwen35' or weights.layers != 32 or weights.width(14) != 9216):
        parser.error('These presets target Qwen3.5-4B (32 layers, 9216 MLP neurons)')
    out = args.output or ROOT / 'runs' / dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    provenance['arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if character_config:
        provenance['character_config'] = character_config
    provenance['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [ROOT / 'qwentuner.py', ROOT / 'native/runner.cpp']}
    write_json(out / 'manifest.json', provenance)
    print(f'Results: {out}', flush=True)
    if args.command == 'lens':
        if args.top < 1:
            parser.error('--top must be positive')
        projection = weights.final_layer_projection(args.layer, args.neuron, args.sign, args.top)
        write_json(out / 'projection.json', projection)
        print(json.dumps(projection, indent=2, ensure_ascii=False))
        return
    with contextlib.closing(Runner(weights, out, args.gpu_layers, args.context, args.raw)) as runner:
        if args.command == 'discover':
            discover(runner, args, out)
        else:
            specs = (RECIPE if args.recipe else []) + config_specs + args.steer
            rows = []
            for multiplier in [0.0] + args.sweep:
                if not math.isfinite(multiplier):
                    parser.error('Sweep values must be finite')
                interventions = [(l, n, d * multiplier) for l, n, d in specs] if multiplier else []
                row = runner.run(args.prompt, interventions, args.scale, args.max_tokens)
                row['multiplier'] = multiplier
                rows.append(row)
                write_json(out / 'comparison.json', rows)
                print(f'\n--- {"baseline" if multiplier == 0 else f"dose x {multiplier:g}"} ---\n{row["text"]}', flush=True)


if __name__ == '__main__':
    main()
