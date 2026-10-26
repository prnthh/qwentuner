"""Opt-in correctness checks against actual GGUF weights (requires GPU access)."""
import os
from pathlib import Path

import numpy as np
import pytest

from qwentuner import Runner, Weights, chat_prompt


@pytest.mark.skipif(not os.getenv('QWENTUNER_MODEL'), reason='Set QWENTUNER_MODEL for real-weight tests')
def test_hook_effect_and_cache_isolation(tmp_path):
    weights = Weights(Path(os.environ['QWENTUNER_MODEL']))
    runner = Runner(weights, tmp_path)
    try:
        prompt = 'Describe a small animal exploring a garden in two sentences.'
        first = runner.run(prompt, max_tokens=16, capture=True)
        zero = runner.run(prompt, [(13, 7956, 0)], scaling='raw', max_tokens=16)
        assert first['token_ids'] == zero['token_ids']
        steered = runner.run(prompt, [(13, 7956, 1)], scaling='raw', max_tokens=0, capture=True)
        assert steered['hook_hits']['13'] == steered['prompt_tokens']
        # Capture is before injection: layer 13 is unchanged, layer 14 changes.
        np.testing.assert_allclose(first['activations']['13'], steered['activations']['13'], atol=1e-5)
        assert not np.allclose(first['activations']['14'], steered['activations']['14'], atol=1e-5)
        again = runner.run(prompt, max_tokens=16)
        assert first['token_ids'] == again['token_ids']
        calibrated = runner.request(prompt=chat_prompt(prompt), max_tokens=0, calibrate=True)
        assert len(calibrated['residual_norms']) == 32
        assert all(np.isfinite(v) and v > 0 for v in calibrated['residual_norms'].values())
        with pytest.raises(RuntimeError, match='out of range'):
            runner.request(prompt=chat_prompt(prompt), max_tokens=0,
                           steering=[{'layer': 13, 'neuron': 100000, 'amount': 1}])
        # A rejected request must not poison the next request.
        assert runner.run(prompt, max_tokens=16)['token_ids'] == first['token_ids']
    finally:
        runner.close()
