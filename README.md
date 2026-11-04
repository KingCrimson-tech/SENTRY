# SENTRY — live voice-clone detector

A CPU-only, streaming **audio deepfake detector** built as a final-year project
around:

> Banerjee, Chatterjee, Bhattacharya et al.
> **SENTRY: A Streaming Edge Network Trust Engine for Real-Time Audio Deepfake Detection**
> arXiv:[2610.24468](https://arxiv.org/abs/2610.24468) (Oct 2026)

It scores 1-second windows four times per second and emits an action, not just a
score:

| Evidence | Action | Meaning |
|---|---|---|
| window score < 0.40, in distribution | **LOG** | no synthetic evidence — keep monitoring |
| window score 0.40 – 0.60 | **CHALLENGE** | suspicious — challenge the caller |
| window score > 0.60 | **DROP** | AI-generated — do not trust |
| far from the training data | **UNKNOWN** | outside validated conditions — unresolved, not authentic |

`LOG` deliberately does **not** mean "this is a human". It means "nothing in the
validated feature set matched known synthesis". The console says so on screen.

This repo reproduces the paper's **streaming framing, feature families and
four-level LOG/CHALLENGE/DROP/UNKNOWN policy** in Python/LightGBM. It does **not** reproduce the
paper's bit-exact integer front end, its Mahalanobis track, or its FPGA
deployment, and it does **not** claim the paper's 98.75% / 1.25% numbers.

---

## Measured results

All numbers are from this repository; none are the paper's.

| Experiment | n | Accuracy | AUC | EER |
|---|---|---|---|---|
| Grouped held-out (final model) | 9,908 | **99.7%** | **0.9999** | **0.24%** |
| FoR-only → FoR held-out | 9,876 | 99.71% | 0.9999 | 0.30% |
| FoR-only → unseen 2026 TTS | 300 | **60.0%** | — | — |
| Clean model → speaker → mic | 400 | **42.0%** | 0.566 | 44.5% |
| Channel specialist, leave-one-clip-out | 400 | 34.2% | — | — |

Leave-one-voice-out on the five edge-tts voices: **96.7–100%**. Clip
aggregation: **0/80** false positives on held-out real FoR, **80/80** on held-out
FoR fakes, **150/150** generated TTS clips.

The last three rows are the honest part. Accuracy collapses on modern TTS it has
never seen and through a real microphone channel. That is the same
generalisation failure the literature reports, and it is why the live mic is
marked experimental.

### What drives the decisions

Gain-based feature importance from the trained model:

| Family | Share |
|---|---|
| MFCC + deltas | 59.4% |
| Spectral shape | 28.4% |
| Wavelet packet | 8.9% |
| Waveform micro-stats | 2.8% |
| Walsh–Hadamard | 0.3% |
| Pitch | 0.2% |

Almost 88% comes from the ordinary spectral front end, not the multiplier-free
transforms. That is worth stating plainly in the viva: the paper's cheap
transforms are cheap, but they are not where the accuracy lives.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py          # http://127.0.0.1:7860
```

The repo ships a trained model (`models/sentry_lgbm.txt`), its metrics
(`models/meta.json`, `models/eval.json`), the novelty guard
(`models/ood.npz`), and three short demo clips in `samples/`. The web console
runs with **no front-end build step** and streams results over Server-Sent
Events.

### Before presenting

```bash
make check        # or: .venv/bin/python scripts/demo_check.py
```

This scores the three shipped clips plus any held-out clips present, verifies
that known material is not flagged out of scope, checks that the figures shown
in the console match the stored evaluation, and exits non-zero on any failure.
Run it in the actual room, on the actual laptop, five minutes before the demo.

---

## The demo

1. **File analysis** — load the *spoofed speech* sample → **DROP**.
2. **File analysis** — load the *modern TTS* sample → **DROP**.
3. **File analysis** — load the *real speech* sample → **LOG** (no synthetic
   evidence).
4. **Attack lab** — generate a scam line with a current voice → **DROP**.
5. **Method & limits** — show the 60% / 42% rows and the feature-importance
   table. Present the failure before anyone asks about it.
6. **Live mic** — last, and labelled experimental.

If an unfamiliar clip comes back **UNKNOWN**, that is the system working as
designed. The sentence to use: *"It is telling us the clip is outside the
conditions it was validated on, which is exactly the behaviour we added after
measuring the 60% generalisation gap."*

Do **not** use random social-media reels as pass/fail evidence. They carry
music, compression and multiple speakers, so they are out of distribution.

---

## How a clip is scored

A fake clip only needs *some* windows to expose the artefact, so the file and
attack-lab tabs do not average every window:

- **clip score** = mean of the top 30% of windows
- **AI-window ratio** = fraction of windows ≥ 0.60
- **DROP** if clip score ≥ 0.50 and AI-window ratio ≥ 15%
- **CHALLENGE** if clip score ≥ 0.40 or AI-window ratio ≥ 5%
- **gray zone**: clip score ≥ 0.20 also becomes CHALLENGE, so LOG is reserved
  for clearly low evidence
- **sparse guard**: top-10% ≥ 0.45 but clip score < 0.50 → escalate LOG to
  CHALLENGE ("unresolved")
- **novelty guard**: if at least half the windows are far from the training
  manifold, return **UNKNOWN** instead of LOG; a confident DROP is left alone

The sparse guard fixes voices such as `en-GB-SoniaNeural`, whose windows are
bimodal: about 35% score 1.0 while the plain mean is only 0.44.

---

## Reproducing the model

```bash
make data                 # Fake-or-Real (norm), real + fake shards (~2.8 GB)
make tts                  # 150–200 edge-tts clips, 5 voices (needs internet)
make train                # grouped split + LightGBM + novelty guard
make eval                 # models/eval.json + figures
.venv/bin/python -m src.ood --build   # rebuild models/ood.npz on its own
```

⚠️ **FoR is sorted by class**: shards 0–9 are real, 10–19 are fake. Downloading
only the first shards gives a 100%-real dataset and a useless "always real"
model.

On this laptop the detector costs about **23–26 ms per 1 s window** on one CPU
core. The live mic path is server-side (`arecord`), so no browser permission
prompt is needed. If the level is low, pick another device in the dropdown or
set `SENTRY_ALSA_DEV=plughw:2,0`; `arecord -l` lists the cards.

The optional channel experiment (`make chan`) plays clips through the laptop
speakers and re-records them with the internal mic. It is a negative result and
does **not** change the deployed clean model.

---

## Layout

```
sentry/
├── app.py                  # FastAPI console: file, mic, TTS attack lab
├── web/                    # index.html, styles.css, app.js
├── src/
│   ├── features.py         # 178-d handcrafted feature extractor
│   ├── data.py             # FoR download + feature cache
│   ├── train.py            # grouped split + LightGBM + EER
│   ├── detector.py         # windows, EMA, hysteresis, clip aggregation
│   ├── ood.py              # novelty guard (UNKNOWN vs LOG)
│   ├── evaluate.py         # held-out, leave-one-voice-out, channel study
│   ├── augment.py          # noise / low-pass / reverb / gain
│   ├── channel_data.py     # speaker → room → mic recordings
│   └── make_samples.py     # edge-tts generation + cache
├── scripts/demo_check.py   # pre-flight check before presenting
├── samples/                # three deterministic demo clips
├── models/                 # model, metrics, novelty reference
├── data/                   # datasets and recordings (not committed)
└── reports/                # per-file JSON analyses
```

## Limitations to state in the viva

1. FoR is 2019-era; modern zero-shot clones (ElevenLabs-class) are harder, and
   accuracy drops to 60% on unseen TTS.
2. 1 s windows discard long-range prosody.
3. Codec, noise and reverb shift scores; the microphone channel drops accuracy
   to 42%, so live detection is experimental.
4. English-only training; Indic languages are out of distribution.
5. The novelty guard is a safety net, not a guarantee: unfamiliar audio returns
   **UNKNOWN**, not proof.
6. A single score is **not** forensic proof. Always combine it with a challenge
   phrase, caller verification, and human judgement.

## Ethics

Only analyse audio you have consent to use. Do not deploy this to accuse
individuals. The TTS attack lab exists to test defences, not to create scams.
