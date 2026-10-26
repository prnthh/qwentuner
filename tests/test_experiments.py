import argparse
import numpy as np
import pytest

from qwentuner import chat_prompt, paired_candidates, parse_steer, passes_quality_screen, word_metrics


def test_word_boundaries_and_repetition():
    m = word_metrics('Purple, purple pears! Apple pie; happy people.')
    assert m['word_count'] == 7
    assert m['p_words'] == 5
    assert m['p_fraction'] == 5 / 7
    assert m['unique_word_fraction'] == 6 / 7
    assert word_metrics('123...')['p_fraction'] == 0


@pytest.mark.parametrize('bad', ['-1:2:1', '0:-1:1', '0:2:nan', '0:2:inf', '0:2', 'hello'])
def test_invalid_interventions_rejected(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_steer(bad)


def test_contrast_screen_recovers_both_signed_effects():
    rng = np.random.default_rng(42)
    negative = rng.normal(size=(20, 2, 8))
    positive = negative.copy()
    positive[:, 0, 3] += 5
    positive[:, 1, 6] -= 8
    found = paired_candidates(positive, negative, 2)
    assert {(x['layer'], x['neuron'], x['sign']) for x in found} == {(0, 3, 1), (1, 6, -1)}


def test_template_has_closed_thinking_and_raw_passthrough():
    assert chat_prompt('hello').endswith('<think>\n\n</think>\n\n')
    assert chat_prompt('literal', raw=True) == 'literal'


def test_letter_metric_cannot_qualify_repetition_as_success():
    collapsed = {'metrics': word_metrics('Pan ' * 50)}
    assert collapsed['metrics']['p_fraction'] == 1
    assert not passes_quality_screen(collapsed)
    assert not passes_quality_screen({'metrics': word_metrics('P.P.P.P.P.P.P.P.')})
    readable = {'metrics': word_metrics('Pale pink petals drift past the pond as a patient painter prepares a portrait of the quiet garden.')}
    assert passes_quality_screen(readable)
