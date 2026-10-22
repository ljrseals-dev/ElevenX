# AOC-style Vocal Studio adapter

This milestone adds a renderer-neutral arrangement contract to ElevenX. It creates editable lead, harmony, double, and ad-lib part specifications and can request one independent mono stem per part from an injected renderer.

## Important capability boundary

The adapter is not itself a singing model. A part's prompt is a production direction, not a MIDI melody or guaranteed interval. The current XTTS-v2 speech backend does not guarantee pitch-accurate singing or distinct musical harmony parts; do not present a plan or repeated speech takes as finished harmonies. A future renderer must implement `VocalRenderer.render_part` and pass musical-quality tests before it is advertised as singing support.

## Example

```python
from elevenx.studio import build_harmony_arrangement, render_arrangement

plan = build_harmony_arrangement(
    "Chorus",
    "Your lyric goes here",
    part_count=4,
    include_adlibs=True,
)
# renderer must implement render_part(part, reference_audio) -> (mono_float_audio, sample_rate)
# stems = render_arrangement(plan, renderer, "authorized_reference.wav")
for part in plan.parts:
    print(part.name, part.role, part.gain, part.pan)
```

Each part carries a name, role, rendering direction, gain, pan, and optional start offset. Rendering returns a dictionary of separately editable stems; this adapter intentionally does not silently mix them or apply unimplemented DSP.

## Relationship to Source/AOC-Magic-v3-main.zip

The original archive remains unchanged in `Source/`. This PR does not claim to have imported the archive's internal Python implementation: the archive has not yet been extracted and audited for its actual modules, license compatibility, and dependencies. Once that review is complete, the matching AOC implementation can replace or back this adapter without coupling arrangement planning to XTTS. HarmoniaCodec remains out of the active path until trained weights and quality gates exist.
