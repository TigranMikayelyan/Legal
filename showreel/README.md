# SkyPivot Showreel 2026

48-second, 1920x1080, 30 fps motion-design showreel for SkyPivot Drone Services.

- `SkyPivot_Showreel_2026.mp4` — the finished video (H.264 + AAC).
- `showreel.html` — the animation itself. Open it in a browser for a live loop preview. Every scene is driven by a single `seek(t)` function, so any frame can be rendered deterministically.
- `assets/` — the photographs and keyed logo used in the piece.
- `tools/prep_assets.py` — converts the raw uploads into the named assets and keys the logo background out.
- `tools/render.py` — renders `showreel.html` to PNG frames with headless Chromium (Playwright).
- `tools/audio.py` — synthesizes the music bed and sound effects with numpy, timed to the scene events.

## Rebuilding

```bash
python3 tools/render.py showreel.html frames            # ~1440 PNG frames
python3 tools/audio.py soundtrack.wav
ffmpeg -framerate 30 -i frames/f_%05d.png -i soundtrack.wav \
  -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p -c:a aac -b:a 192k \
  -movflags +faststart -shortest SkyPivot_Showreel_2026.mp4
```

## Changing a photo

Replace the file in `assets/` with one of the same name, or edit the `background-image` URL in `showreel.html`, then re-render.

## Scene map

| Time | Scene |
|---|---|
| 0–1s | Intro line |
| 1–5.7s | Opening orbit, altitude counter, "Construction moves fast. Stay ahead of it." |
| 5.5–10.3s | 01 Construction progress monitoring, five top-down captures |
| 10.1–13.5s | Timeline strip: site work → complete |
| 13.3–18.3s | Progress report card, stakeholder chips |
| 18.1–22.4s | Cost-to-fix chart, "Save time. Save money." |
| 22.2–26.3s | 02 Roof inspection with tagged points |
| 26.1–29.3s | Facade inspection with scan and frame counter |
| 29.1–32.3s | 03 Construction marketing mockups |
| 32.1–34.5s | 04 Premium aerials: residential / commercial / industrial |
| 34.3–37.9s | Golf course, "Make the property the story." |
| 37.7–41.1s | Fast delivery / Consistent results / Client satisfaction |
| 40.9–43.9s | Certifications |
| 43.7–48s | Logo, "Let's pivot your project." |
