import json

import pytest

from qwentuner import RECIPE, load_character_config


def test_evil_config_matches_known_working_recipe_exactly():
    config, specs = load_character_config('evil-qwen')
    assert specs == RECIPE
    assert config['scaling'] == 'residual'
    assert (15, 3920, 1.0) not in specs


def test_independent_cases_preserve_complete_neuron_lists():
    _, sarcastic = load_character_config('sarcastiqwen')
    _, l12 = load_character_config('sarcastiqwen-sycophancy-l12')
    _, l16 = load_character_config('sarcastiqwen-sycophancy-l16')
    _, condolences = load_character_config('sarcastiqwen-condolences')
    config, chill = load_character_config('sarcastiqwen-chill-friend')
    assert sarcastic == [(15, 3920, 1.0)]
    assert l12 == [(15, 3920, 1.0), (12, 2920, -1.0)]
    assert l16 == [(15, 3920, 1.0), (16, 8168, -1.0)]
    assert condolences == [(15, 3920, 1.0), (10, 6295, 1.0)]
    assert chill == [(15, 3920, 1.0), (11, 6261, .4), (17, 7870, -.4)]
    assert config['scaling'] == 'residual'


@pytest.mark.parametrize('entries', [
    [],
    [{'layer': 32, 'neuron': 1, 'dose': 1}],
    [{'layer': 1, 'neuron': 9216, 'dose': 1}],
    [{'layer': 1, 'neuron': 1, 'dose': float('nan')}],
    [{'layer': 1, 'neuron': 1, 'dose': 1}] * 2,
])
def test_bad_config_fails_before_model_load(tmp_path, entries):
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps({'schema_version': 1, 'model': 'Qwen3.5-4B',
                                'scaling': 'residual', 'steering': entries}))
    with pytest.raises(ValueError):
        load_character_config(str(path))
