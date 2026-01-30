# RMS-Power and Max-Amplitude Mixing: Why-When


This custom PyTorch-based audio mixing tool, **rms_power** and **max_amplitude** represent two distinct approaches to **pre-balancing individual tracks** (gain staging/review) and **summing/mixing** them into a final stereo (or multi-channel) output.

These aren't standard DAW terms but logical custom modes inspired by classic audio metering and summation physics. Below is a clear comparison of their key features, strengths, typical applications, and why one might be "better" for certain use cases (based on real audio engineering principles like coherent vs. incoherent summation, perceived loudness, crest factor, and headroom).

### Core Features Comparison

| Aspect                        | rms_power mode                                                                 | max_amplitude mode                                                              | Winner / Notes |
|-------------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------|---------------|
| **Balancing metric (per-track)** | Measures **RMS** (root mean square) after current gain → average power/loudness | Measures **peak** (max abs sample value) after current gain → instantaneous max amplitude | RMS usually wins for musical balance |
| **Perceived loudness match**  | Excellent — correlates well with how loud something sounds over time           | Poor — transient-heavy tracks sound much quieter than sustained ones at same peak | RMS |
| **Crest factor handling**     | Ignores it; balances average energy → good for dynamics preservation           | Ignores it completely; can over-boost pads/synths and kill percussion transients | RMS (preserves natural dynamics) |
| **Summation method (after fixes)** | Coherent voltage sum (`torch.sum`) by default; type-dependent adjustments (Standard/Absonic/Sample-accurate) | Coherent sum for Standard/Absonic + peak norm; strict `torch.max` for Sample-accurate | Depends on track types |
| **Typical headroom after sum** | Good if targets set low (-23…-18 dB RMS); leaves room for bus processing      | Tighter; relies heavily on final peak safety trim (can still feel spiky)        | RMS (more predictable) |
| **Clipping risk (pre-fix)**   | High with naive power-average sum; fixed with coherent sum + safety           | Very high with sample-wise max; mitigated with type-based coherent fallback     | Both need safety |
| **Best for multi-channel/spatial** | Coherent sum works; Absonic type can add weighting for higher-order Ambisonics | Coherent sum fallback helps; strict max can break spatial phase                | RMS (more natural) |
| **Experimental/glitch use**   | Less ideal unless Sample-accurate type forces special handling                 | Excellent when Sample-accurate → preserves exact aligned transients/peaks       | max_amplitude |

### Why rms_power is usually better (and for which applications)

**rms_power** aligns with **modern gain-staging and perceived-loudness workflows** (short-term LUFS or classic -18 dB RMS targets).  
It is generally **superior** in these scenarios:

- **Standard music production / remixing / stem balancing** (pop, electronic, rock, hip-hop, vocals + instruments)  
  → Balances **average loudness** → tracks sit naturally together without massive boosts to sustained elements or killing transients.  
  → Leaves realistic headroom for EQ, compression, saturation, bus glue.  
  → Matches how humans perceive volume (sustained pads/vocals aren't drowned by snappy drums).

- **Automated pre-mix leveling** before serious processing  
  → RMS is closer to LUFS/short-term loudness → fewer surprises when summing 10–30 tracks.

- **Any workflow where dynamics and musical balance matter**  
  → Avoids the classic peak-normalization trap (percussion becomes whisper-quiet while pads blast).

**When rms_power shines most**: You have a diverse multitrack session (drums + bass + guitars + vocals + synths) and want the mix to feel "pro" and cohesive without manual fader tweaks.

### Why max_amplitude can be better (niche but powerful cases)

**max_amplitude** prioritizes **peak preservation** over average loudness — it's closer to "peak normalization" but with safeguards.

It outperforms rms_power in:

- **Sample-accurate / glitch / granular / experimental audio**  
  → When you **need** to keep instantaneous maximum values aligned (e.g. phase-coherent percussion layers, clicky effects, anti-phase tricks, or aligned transient stacks).  
  → Strict `torch.max` per sample prevents destructive interference on critical peaks.

- **Very transient-heavy libraries or one-shot processing** (e.g. drum replacers, foley, extreme compression already applied)  
  → Ensures no peak is lost or softened unintentionally.

- **Debug / visualization / safety pass** ("what's the absolute worst-case peak contribution?")  
  → Useful diagnostically before switching to coherent sum.

- **Scenarios where everything is already heavily limited/compressed** (loudness-war stems, pre-mastered elements)  
  → Peak ≈ RMS behavior → difference shrinks, and max mode avoids over-summing averages.

**When max_amplitude shines most**: Experimental electronic/IDM/glitch workflows, sound design with tight phase alignment, or when you explicitly want "the loudest thing wins at every sample" for artistic effect.

### Quick Recommendation Table (2026 perspective)

| Your primary goal / content type                  | Recommended mode (with track_type selector) | Why better |
|---------------------------------------------------|---------------------------------------------|------------|
| Normal songs, remixes, pop/EDM/rock stems         | rms_power (mostly Standard type)            | Natural perceived balance, headroom, musicality |
| Spatial/Ambisonics/immersive audio                | rms_power (Absonic type)                    | Coherent sum preserves phase/space better |
| Glitch, granular, phase-critical experiments      | max_amplitude (Sample-accurate type)        | Preserves exact peaks/transients/alignment |
| Mixed session with very different crest factors   | rms_power + manual type overrides           | RMS handles dynamics better overall |
| Quick "don't clip anything" safety render         | max_amplitude (with safety trim)            | Peak-focused by design |

**Bottom line**:  
In almost all **musical/real-world mixing** scenarios, **rms_power** (with coherent summation and proper targets) is the better, more professional choice — it's why most DAWs and loudness tools focus on average/RMS/LUFS rather than pure peak.  
**max_amplitude** is a specialized tool — powerful for experimental/peak-exact needs, but usually less musical unless you're intentionally going for that character.

Use your unified `track_type` selector to fine-tune per track — that's the smart hybrid approach your code now supports! If you share more about your typical audio (e.g. "mostly electronic stems with heavy percussion"), I can suggest optimal default targets/types.