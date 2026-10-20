"""Synthesize the showreel soundtrack (music bed + sound effects) with numpy only.
usage: python3 audio.py <out.wav>
All event times are in seconds and mirror the timeline in showreel.html.
"""
import sys, wave, numpy as np
SR = 48000; DUR = 48.0; N = int(SR * DUR)
rng = np.random.default_rng(7)
L = np.zeros(N); Rr = np.zeros(N)
tt = np.arange(N) / SR

def add(buf_l, buf_r, sig, t0, pan=0.0, gain=1.0):
    """overlap-add mono sig at t0 with constant-power pan (-1..1)."""
    i = int(t0 * SR); n = min(len(sig), N - i)
    if n <= 0 or i < 0: return
    gl = np.cos((pan + 1) * np.pi / 4) * gain; gr = np.sin((pan + 1) * np.pi / 4) * gain
    buf_l[i:i + n] += sig[:n] * gl; buf_r[i:i + n] += sig[:n] * gr

def fft_filter(x, lo=None, hi=None, slope=2.0):
    """static band filter via FFT with soft edges"""
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); g = np.ones_like(f)
    if hi: g *= 1 / (1 + (f / hi) ** (2 * slope))
    if lo: g *= 1 - 1 / (1 + (f / max(lo, 1)) ** (2 * slope))
    return np.fft.irfft(X * g, n=len(x))

def onepole_sweep(x, f0, f1, curve=1.0):
    """time-varying one-pole lowpass whose cutoff sweeps f0 -> f1 (log) — python loop, use on short signals"""
    n = len(x); y = np.empty(n); p = (np.arange(n) / n) ** curve
    fc = f0 * (f1 / f0) ** p; a = np.exp(-2 * np.pi * fc / SR); s = 0.0
    for i in range(n):
        s = a[i] * s + (1 - a[i]) * x[i]; y[i] = s
    return y

def env_exp(n, decay):  # exponential decay envelope
    return np.exp(-np.arange(n) / SR / decay)

def sine(f, dur, phase=0):
    t = np.arange(int(dur * SR)) / SR; return np.sin(2 * np.pi * f * t + phase)

def noise(dur): return rng.standard_normal(int(dur * SR))

def fade(sig, a=0.005, d=0.02):
    n = len(sig); na = int(a * SR); nd = int(d * SR)
    if na: sig[:na] *= np.linspace(0, 1, na)
    if nd: sig[-nd:] *= np.linspace(1, 0, nd)
    return sig

# ---------------- instruments ----------------
def kick(amp=1.0, dur=0.45, f0=150, f1=42):
    n = int(dur * SR); t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.045); ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.16); click = fft_filter(noise(dur), lo=1500) * np.exp(-t / 0.004) * 0.6
    return np.tanh((body + click) * 1.8) * amp

def hat(amp=0.12, dur=0.09):
    n = int(dur * SR); return fft_filter(noise(dur), lo=7000) * env_exp(n, 0.018) * amp

def boom(amp=0.9, dur=1.6, f=48):
    n = int(dur * SR); t = np.arange(n) / SR
    sub = np.sin(2 * np.pi * (f + 30 * np.exp(-t / 0.08)) * t) * np.exp(-t / 0.5)
    air = fft_filter(noise(dur), lo=200, hi=3000) * np.exp(-t / 0.12) * 0.5
    return np.tanh((sub + air) * 1.5) * amp

def whoosh(amp=0.35, dur=0.5, f0=6000, f1=400, pan_from=-0.6):
    n = int(dur * SR); t = np.arange(n) / SR
    x = onepole_sweep(fft_filter(noise(dur), lo=250), f0, f1, 1.3)
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    return x * e * amp

def blip(amp=0.22, f0=1500, f1=950, dur=0.09):
    n = int(dur * SR); t = np.arange(n) / SR; f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.028) * amp

def tick(amp=0.1, dur=0.012):
    n = int(dur * SR); return fft_filter(noise(dur), lo=2500, hi=9000) * env_exp(n, 0.0025) * amp

def shutter(amp=0.3):
    a = fft_filter(noise(0.05), lo=1800, hi=8000) * env_exp(int(0.05 * SR), 0.006)
    b = fft_filter(noise(0.08), lo=800, hi=5000) * env_exp(int(0.08 * SR), 0.012) * 0.7
    out = np.zeros(int(0.12 * SR)); out[:len(a)] += a; out[int(0.035 * SR):int(0.035 * SR) + len(b)] += b
    return out * amp

def thud(amp=0.2):
    n = int(0.25 * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.07) + fft_filter(noise(0.25), hi=900) * np.exp(-t / 0.03) * 0.8) * amp

def riser(dur, amp=0.4):
    n = int(dur * SR); t = np.arange(n) / SR
    x = onepole_sweep(noise(dur), 250, 9000, 1.0) * (t / dur) ** 2.2
    tone = np.sin(2 * np.pi * np.cumsum(90 * 2 ** (t / dur * 3)) / SR) * (t / dur) ** 3 * 0.35
    return (x + tone) * amp * np.where(t > dur - 0.01, 0, 1)

def pluck(f, dur=0.35, amp=0.08):
    n = int(dur * SR); t = np.arange(n) / SR
    tri = 2 / np.pi * np.arcsin(np.sin(2 * np.pi * f * t)); s2 = np.sin(2 * np.pi * f * 2 * t) * 0.35
    return (tri + s2) * np.exp(-t / 0.11) * amp

def bass(f, dur=0.6, amp=0.4):
    n = int(dur * SR); t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / 0.1)
    return np.tanh(x * 1.6) * np.exp(-t / 0.28) * amp

def pad_chord(freqs, dur, amp=0.1, cutoff=800, seed=0):
    n = int(dur * SR); t = np.arange(n) / SR; out = np.zeros(n); r = np.random.default_rng(seed)
    for f in freqs:
        for det in (-0.004, 0.0, 0.004):
            ph = r.uniform(0, 1); out += (2 * ((f * (1 + det) * t + ph) % 1) - 1) / 3
    out = fft_filter(out / len(freqs), hi=cutoff, slope=1.2)
    e = np.ones(n); a = int(0.35 * SR); d = int(0.35 * SR)
    e[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2; e[-d:] = np.cos(np.linspace(0, np.pi / 2, d)) ** 2
    return out * e * amp

# ---------------- music ----------------
BPM = 100; BEAT = 60 / BPM
M = {'A2': 110.0, 'C3': 130.81, 'E3': 164.81, 'A3': 220.0, 'F2': 87.31, 'F3': 174.61, 'G2': 98.0, 'B2': 123.47, 'D3': 146.83, 'G3': 196.0, 'C4': 261.63, 'E4': 329.63, 'A4': 440.0, 'F4': 349.23, 'G4': 392.0, 'D4': 293.66, 'B3': 246.94}
PROG = [(['A2', 'C3', 'E3', 'A3'], 55.0, ['A3', 'C4', 'E4', 'A4']), (['F2', 'A2', 'C3', 'F3'], 43.65, ['F3', 'A3', 'C4', 'F4']),
        (['C3', 'E3', 'G3', 'C4'], 65.41, ['C4', 'E4', 'G4', 'C4']), (['G2', 'B2', 'D3', 'G3'], 49.0, ['G3', 'B3', 'D4', 'G4'])]
CH = 4 * BEAT  # one chord per bar (2.4s)
music_l = np.zeros(N); music_r = np.zeros(N)
MUSIC_START = 0.6; MUSIC_END = 46.2
# pad + bass + arp per bar
bar = 0; t = MUSIC_START
while t < MUSIC_END:
    notes, root, arpn = PROG[bar % 4]
    # dynamics by section
    cutoff = 900 if t < 5.5 else 1500 if t < 22 else 2000 if t < 37.6 else 1400
    amp = 0.16 if t < 5.5 else 0.22 if t < 37.6 else 0.18
    pc = pad_chord([M[n] for n in notes], CH + 0.4, amp=amp, cutoff=cutoff, seed=bar)
    add(music_l, music_r, pc, t, pan=-0.25); add(music_l, music_r, pc, t + 0.012, pan=0.25)  # haas width
    for b in range(4):
        tb = t + b * BEAT
        if 5.4 <= tb < 46:
            add(music_l, music_r, bass(root, amp=0.3 if tb >= 22 else 0.22), tb)
        if 13.9 <= tb < 37.6:
            for s in range(4):
                ts = tb + s * BEAT / 4; f = M[arpn[(b * 4 + s) % 4]] * (2 if s == 3 else 1)
                add(music_l, music_r, pluck(f, amp=0.09 if s else 0.12), ts, pan=0.5 * np.sin(ts * 2.1))
    bar += 1; t += CH
# sub drone under the opening scene
dr = np.sin(2 * np.pi * 55 * np.arange(int(5.2 * SR)) / SR) * np.sin(np.linspace(0, np.pi, int(5.2 * SR))) ** 0.6 * 0.18
add(music_l, music_r, dr, 0.6)
# drums
t = 5.4; i = 0
while t < 43.75:
    beat_in_bar = i % 4
    if t < 22:
        if beat_in_bar in (0, 2): add(music_l, music_r, kick(0.55), t)
    elif t < 37.6:
        add(music_l, music_r, kick(0.62 if beat_in_bar == 0 else 0.5), t)
    elif t < 43.7:
        if beat_in_bar in (0, 2): add(music_l, music_r, kick(0.45), t)
    if 10.1 <= t < 37.6:
        add(music_l, music_r, hat(0.11), t, pan=0.3); add(music_l, music_r, hat(0.16), t + BEAT / 2, pan=0.35)
        if t >= 22: add(music_l, music_r, hat(0.08), t + BEAT / 4, pan=0.25); add(music_l, music_r, hat(0.08), t + 3 * BEAT / 4, pan=0.25)
    i += 1; t += BEAT
# risers + impacts
for t0, d in [(19.1, 1.3), (36.3, 1.4), (42.5, 1.3)]:
    add(music_l, music_r, riser(d, 0.38), t0)
for t0, a in [(1.0, 0.55), (20.4, 0.6), (22.2, 0.55), (37.7, 0.6), (43.8, 0.8)]:
    add(music_l, music_r, boom(a), t0)
# final chord swell at logo
pc = pad_chord([M[n] for n in ['A2', 'E3', 'A3', 'C4', 'E4']], 4.5, amp=0.3, cutoff=2200, seed=99)
add(music_l, music_r, pc, 43.8, pan=-0.3); add(music_l, music_r, pc, 43.815, pan=0.3)
shimmer = fft_filter(noise(4.0), lo=5000, hi=12000) * np.linspace(1, 0, int(4.0 * SR)) ** 2 * 0.05
add(music_l, music_r, shimmer, 43.8)

# ---------------- sound effects ----------------
sfx_l = np.zeros(N); sfx_r = np.zeros(N)
for t0 in [5.5, 10.1, 13.3, 18.1, 22.2, 26.1, 29.1, 32.1, 34.3, 37.7, 40.9, 43.7]:
    add(sfx_l, sfx_r, whoosh(0.32), t0 - 0.12, pan=-0.4); add(sfx_l, sfx_r, whoosh(0.32, f0=5000, f1=600), t0 + 0.02, pan=0.4)
for t0 in [13.95, 23.0]:
    add(sfx_l, sfx_r, whoosh(0.22, dur=0.7, f0=3000, f1=300), t0, pan=0.3)
add(sfx_l, sfx_r, whoosh(0.28, dur=0.9, f0=2000, f1=6000), 0.1)  # intro line draw
# altitude counter ticks
tc = 1.35
while tc < 3.3:
    add(sfx_l, sfx_r, tick(0.09), tc, pan=0.55); tc += 0.05 + 0.08 * ((tc - 1.35) / 2) ** 2
add(sfx_l, sfx_r, blip(0.2, 1200, 1200, 0.16), 3.35, pan=0.55)
# capture changes (shutter)
for k in range(5): add(sfx_l, sfx_r, shutter(0.28), 5.7 + k * 0.92, pan=0.35)
# timeline dot travel + labels
for n in range(5): add(sfx_l, sfx_r, blip(0.14, 1000 + n * 120, 700 + n * 120), 10.7 + n * 0.4, pan=-0.7 + n * 0.35)
# chips
for n in range(7): add(sfx_l, sfx_r, blip(0.14, 1300, 900), 15.45 + n * 0.27, pan=-0.5)
# chart draw hum + dot
hum = np.sin(2 * np.pi * np.cumsum(300 * 2 ** (np.linspace(0, 1.6, int(1.4 * SR)))) / SR) * np.sin(np.linspace(0, np.pi, int(1.4 * SR))) * 0.05
add(sfx_l, sfx_r, hum, 18.5, pan=0.4); add(sfx_l, sfx_r, blip(0.2, 1600, 1100), 18.9, pan=0.4)
# roof tags
for n in range(4): add(sfx_l, sfx_r, blip(0.2, 1500, 1000), 24.3 + n * 0.42, pan=0.5)
# facade scan + counter ticks + tags
scan = fft_filter(noise(2.1), lo=1500, hi=5000) * 0.03 * np.sin(np.linspace(0, np.pi, int(2.1 * SR))) ** 0.5
add(sfx_l, sfx_r, scan, 26.4, pan=0.5)
for n in range(30): add(sfx_l, sfx_r, tick(0.06), 26.5 + n * (2.4 / 30), pan=-0.5)
for n, t0 in enumerate([27.0, 27.55, 28.15]): add(sfx_l, sfx_r, blip(0.2, 1500, 1000), t0, pan=0.5)
# mockups
for n, t0 in enumerate([29.6, 29.8, 30.0]): add(sfx_l, sfx_r, whoosh(0.16, dur=0.35, f0=4000, f1=800), t0, pan=[0, -0.6, 0.6][n])
for n, t0 in enumerate([30.5, 30.65, 30.8]): add(sfx_l, sfx_r, blip(0.1, 1100, 800), t0, pan=[-0.6, 0, 0.6][n])
# property panels + labels
for n in range(3): add(sfx_l, sfx_r, whoosh(0.2, dur=0.45, f0=5000, f1=700), 32.15 + n * 0.12, pan=-0.6 + n * 0.6)
for n in range(3): add(sfx_l, sfx_r, blip(0.12, 1200, 850), 33.0 + n * 0.15, pan=-0.6 + n * 0.6)
# badges
for n in range(5): add(sfx_l, sfx_r, blip(0.16, 1400 + n * 60, 1000 + n * 60), 41.8 + n * 0.17, pan=-0.6 + n * 0.3)
# logo ring + line
add(sfx_l, sfx_r, whoosh(0.3, dur=1.2, f0=800, f1=7000), 43.85)
add(sfx_l, sfx_r, blip(0.12, 900, 1300, 0.2), 44.7)
# headline thuds
for t0 in [2.0, 2.15, 2.95, 6.05, 6.95, 7.85, 11.0, 11.2, 14.6, 15.2, 18.45, 19.05, 20.55, 21.15, 23.45, 23.85, 26.5, 26.9, 29.5, 30.1, 32.65, 32.85, 34.85, 35.05, 38.0, 38.65, 39.3, 41.15, 44.95]:
    add(sfx_l, sfx_r, thud(0.16), t0)

# ---------------- mix ----------------
mix_l = music_l * 0.85 + sfx_l; mix_r = music_r * 0.85 + sfx_r
# gentle fade in/out
fi = int(0.4 * SR); mix_l[:fi] *= np.linspace(0, 1, fi); mix_r[:fi] *= np.linspace(0, 1, fi)
fo_s = int(47.0 * SR); fo = np.ones(N); fo[fo_s:] = np.linspace(1, 0, N - fo_s) ** 1.5; mix_l *= fo; mix_r *= fo
# soft clip + normalise
for arr in (mix_l, mix_r): np.tanh(arr * 1.15, out=arr)
peak = max(np.abs(mix_l).max(), np.abs(mix_r).max()); g = 10 ** (-1 / 20) / peak
out = np.stack([mix_l * g, mix_r * g], axis=1)
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype('<i2').tobytes())
rms = 20 * np.log10(np.sqrt((out ** 2).mean()) + 1e-9)
print(f'wrote {sys.argv[1]}  peak -1 dBFS  rms {rms:.1f} dBFS')
