import pytest
from elevenx.emotions import Emotion, EmotionPreset

def test_emotion_values_are_explicit():
    assert Emotion("neutral") is Emotion.NEUTRAL
    assert Emotion("empathetic") is Emotion.EMPATHETIC

def test_valid_preset():
    preset = EmotionPreset.from_mapping(
        {"temperature": 0.75, "top_k": 50, "top_p": 0.85, "speed": 1.0})
    assert preset.top_k == 50

@pytest.mark.parametrize("value", [
    {"temperature": 0, "top_k": 50, "top_p": 0.8, "speed": 1.0},
    {"temperature": 0.7, "top_k": 0, "top_p": 0.8, "speed": 1.0},
    {"temperature": 0.7, "top_k": 50, "top_p": 0, "speed": 1.0},
    {"temperature": 0.7, "top_k": 50, "top_p": 0.8, "speed": 4.0},
])
def test_invalid_preset_is_rejected(value):
    with pytest.raises(ValueError):
        EmotionPreset.from_mapping(value)
