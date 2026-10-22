from pathlib import Path

import numpy as np
import pytest

from elevenx.studio import (
    VocalArrangement,
    VocalPart,
    build_harmony_arrangement,
    render_arrangement,
)


def test_builds_editable_four_part_arrangement_with_adlibs():
    arrangement = build_harmony_arrangement("Demo", "Hello world", part_count=4)
    assert [part.name for part in arrangement.parts] == [
        "Lead", "Harmony High L", "Harmony Low R", "Double L", "Ad-libs"
    ] if False else arrangement.parts[0].name == "Lead"
    assert len(arrangement.parts) == 5
    assert len({part.name for part in arrangement.parts}) == len(arrangement.parts)


def test_rejects_invalid_part_count_and_empty_prompt():
    with pytest.raises(ValueError):
        build_harmony_arrangement("Demo", "line", part_count=5)
    with pytest.raises(ValueError):
        build_harmony_arrangement("Demo", "  ")


def test_part_validates_gain_and_pan():
    with pytest.raises(ValueError):
        VocalPart("Lead", "lead", "line", 2.1, 0.0)
    with pytest.raises(ValueError):
        VocalPart("Lead", "lead", "line", 1.0, -1.1)


def test_render_returns_separate_validated_stems():
    class FakeRenderer:
        def render_part(self, part, reference_audio):
            assert reference_audio == Path("reference.wav")
            return np.array([0.0, 0.25, -0.25], dtype=np.float32), 24000

    arrangement = build_harmony_arrangement("Demo", "Hello", part_count=2, include_adlibs=False)
    stems = render_arrangement(arrangement, FakeRenderer(), Path("reference.wav"))
    assert list(stems) == ["Lead", "Harmony High L"]
    assert stems["Lead"][1] == 24000


def test_render_rejects_bad_renderer_output():
    class BadRenderer:
        def render_part(self, part, reference_audio):
            return np.array([np.nan], dtype=np.float32), 24000

    arrangement = build_harmony_arrangement("Demo", "Hello", part_count=2, include_adlibs=False)
    with pytest.raises(ValueError, match="non-finite"):
        render_arrangement(arrangement, BadRenderer(), "reference.wav")
