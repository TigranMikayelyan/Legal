"""Showreel v4 score: cinematic hybrid (braams, taiko, clock hat, string pads, piano motif) + SFX. numpy only.
usage: python3 audio.py <out.wav>
"""
import sys, wave, numpy as np
SR = 48000; DUR = 48.0; N = int(SR * DUR)
rng = np.random.default_rng(23)

def add(bl, br, sig, t0, pan=0.0, gain=1.0):
    i = int(t0 * SR); n = min(len(sig), N - i)
    if n <= 0 or i < 0: return
    gl = np.cos((pan + 1) * np.pi / 4) * gain; gr = np.sin((pan + 1) * np.pi / 4) * gain
    bl[i:i + n] += sig[:n] * gl; br[i:i + n] += sig[:n] * gr

def fft_filter(x, lo=None, hi=None, slope=2.0):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); g = np.ones_like(f)
    if hi: g *= 1 / (1 + (f / hi) ** (2 * slope))
    if lo: g *= 1 - 1 / (1 + (f / max(lo, 1)) ** (2 * slope))
    return np.fft.irfft(X * g, n=len(x))

def onepole_sweep(x, f0, f1, curve=1.0):
    n = len(x); y = np.empty(n); p = (np.arange(n) / n) ** curve
    fc = f0 * (f1 / f0) ** p; a = np.exp(-2 * np.pi * fc / SR); s = 0.0
    for i in range(n):
        s = a[i] * s + (1 - a[i]) * x[i]; y[i] = s
    return y

def env_exp(n, decay): return np.exp(-np.arange(n) / SR / decay)
def noise(dur): return rng.standard_normal(int(dur * SR))
def tline(dur): return np.arange(int(dur * SR)) / SR
def adsr(n, a, d, s, r):
    e = np.ones(n); A = int(a * SR); D = int(d * SR); Rr = int(r * SR)
    e[:A] = np.linspace(0, 1, A); e[A:A + D] = np.linspace(1, s, D); e[A + D:] = s; e[-Rr:] *= np.linspace(1, 0, Rr); return e

# ---------- instruments ----------
def taiko(amp=1.0, dur=0.9, f0=120, f1=48):
    t = tline(dur); f = f1 + (f0 - f1) * np.exp(-t / 0.06); ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.28); skin = fft_filter(noise(dur), lo=200, hi=1800) * np.exp(-t / 0.05) * 0.9
    room = fft_filter(noise(dur), lo=150, hi=900) * np.exp(-t / 0.35) * 0.25
    return np.tanh((body + skin + room) * 1.7) * amp

def kick(amp=1.0, dur=0.45, f0=160, f1=44):
    t = tline(dur); f = f1 + (f0 - f1) * np.exp(-t / 0.04); ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.15); click = fft_filter(noise(dur), lo=1500) * np.exp(-t / 0.004) * 0.7
    return np.tanh((body + click) * 1.9) * amp

def clap(amp=0.3):
    out = np.zeros(int(0.35 * SR))
    for k, d in enumerate([0, 0.011, 0.022, 0.034]):
        n = fft_filter(noise(0.3), lo=900, hi=6500) * np.exp(-tline(0.3) / (0.012 if k < 3 else 0.09))
        i = int(d * SR); out[i:i + len(n)] += n * (0.6 if k < 3 else 1.0)
    return out * amp

def clockhat(amp=0.1):
    t = tline(0.06); return (fft_filter(noise(0.06), lo=6000, hi=14000) * np.exp(-t / 0.012) + np.sin(2 * np.pi * 5200 * t) * np.exp(-t / 0.008) * 0.5) * amp

def hat(amp=0.12, dur=0.09, open_=False):
    n = int(dur * SR); return fft_filter(noise(dur), lo=7000) * env_exp(n, 0.09 if open_ else 0.018) * amp

def braam(f=55.0, dur=2.4, amp=0.5):
    t = tline(dur); out = np.zeros(len(t))
    for det in (-0.012, -0.004, 0.004, 0.012):
        out += 2 * ((f * (1 + det) * t) % 1) - 1
    out += np.sin(2 * np.pi * f / 2 * t) * 2.0
    out = onepole_sweep(out / 6, 180, 1400, 0.7)
    e = np.sin(np.linspace(0, np.pi, len(t))) ** 0.5; e[:int(0.05 * SR)] *= np.linspace(0, 1, int(0.05 * SR))
    return np.tanh(out * 2.2) * e * amp

def boom(amp=0.9, dur=1.8, f=48):
    t = tline(dur); sub = np.sin(2 * np.pi * (f + 30 * np.exp(-t / 0.08)) * t) * np.exp(-t / 0.55)
    air = fft_filter(noise(dur), lo=200, hi=3000) * np.exp(-t / 0.12) * 0.5
    return np.tanh((sub + air) * 1.5) * amp

def subdrop(amp=0.8, dur=1.2, f0=110, f1=30):
    t = tline(dur); f = f1 + (f0 - f1) * np.exp(-t / 0.25); ph = 2 * np.pi * np.cumsum(f) / SR
    return np.tanh(np.sin(ph) * 2) * np.exp(-t / 0.5) * amp

def slam(amp=1.0):
    out = np.zeros(int(2.4 * SR)); b = boom(1.0, 1.8); out[:len(b)] += b
    nb = fft_filter(noise(0.25), lo=300, hi=9000) * env_exp(int(0.25 * SR), 0.05); out[:len(nb)] += nb * 0.9
    sd = subdrop(0.9); out[:len(sd)] += sd; br = braam(55, 2.4, 0.5); out += br * 0.6
    return np.tanh(out * 1.2) * amp

def whoosh(amp=0.35, dur=0.5, f0=6000, f1=400):
    t = tline(dur); x = onepole_sweep(fft_filter(noise(dur), lo=250), f0, f1, 1.3)
    return x * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5 * amp

def flyby(amp=0.3, dur=0.9):
    """doppler-ish pass: pitch falls through the middle"""
    t = tline(dur); f = 900 * 2 ** (-(t / dur - 0.5) * 1.4); tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3
    air = onepole_sweep(fft_filter(noise(dur), lo=300), 5000, 900, 1.0)
    return (tone + air) * np.sin(np.pi * t / dur) ** 1.2 * amp

def rev_cymbal(dur=1.0, amp=0.3):
    t = tline(dur); return fft_filter(noise(dur), lo=3000, hi=14000) * (t / dur) ** 2.5 * amp * np.where(t > dur - 0.02, 0, 1)

def blip(amp=0.22, f0=1500, f1=950, dur=0.09):
    t = tline(dur); f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.028) * amp

def tick(amp=0.1, dur=0.012):
    n = int(dur * SR); return fft_filter(noise(dur), lo=2500, hi=9000) * env_exp(n, 0.0025) * amp

def keytick(amp=0.06):
    t = tline(0.03); return (fft_filter(noise(0.03), lo=1800, hi=7000) * np.exp(-t / 0.004) + np.sin(2 * np.pi * 2400 * t) * np.exp(-t / 0.006) * 0.4) * amp

def shutter(amp=0.3):
    a = fft_filter(noise(0.05), lo=1800, hi=8000) * env_exp(int(0.05 * SR), 0.006)
    b = fft_filter(noise(0.08), lo=800, hi=5000) * env_exp(int(0.08 * SR), 0.012) * 0.7
    out = np.zeros(int(0.12 * SR)); out[:len(a)] += a; out[int(0.035 * SR):int(0.035 * SR) + len(b)] += b
    return out * amp

def thud(amp=0.2):
    t = tline(0.25); return (np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.07) + fft_filter(noise(0.25), hi=900) * np.exp(-t / 0.03) * 0.8) * amp

def riser(dur, amp=0.4):
    t = tline(dur); x = onepole_sweep(noise(dur), 250, 9000, 1.0) * (t / dur) ** 2.2
    tone = np.sin(2 * np.pi * np.cumsum(90 * 2 ** (t / dur * 3)) / SR) * (t / dur) ** 3 * 0.35
    return (x + tone) * amp * np.where(t > dur - 0.01, 0, 1)

def piano(f, dur=1.2, amp=0.1):
    t = tline(dur); x = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.3) + 0.2 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t / 0.15)
    x += fft_filter(noise(dur), lo=1500, hi=5000) * np.exp(-t / 0.006) * 0.3
    return x * np.exp(-t / 0.45) * amp

def strings(freqs, dur, amp=0.1, cutoff=1500, seed=0):
    n = int(dur * SR); t = np.arange(n) / SR; out = np.zeros(n); r = np.random.default_rng(seed)
    for f in freqs:
        for det in (-0.006, -0.002, 0.002, 0.006):
            ph = r.uniform(0, 1); vib = 1 + 0.003 * np.sin(2 * np.pi * (4.5 + r.uniform(0, 1)) * t + ph * 6)
            out += (2 * ((f * (1 + det) * vib * t + ph) % 1) - 1) / 4
    out = fft_filter(out / len(freqs), hi=cutoff, slope=1.0)
    return out * adsr(n, 0.5, 0.3, 0.85, 0.5) * amp

def bass(f, dur=0.5, amp=0.4):
    t = tline(dur); x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / 0.08)
    return np.tanh(x * 1.7) * np.exp(-t / 0.25) * amp

# ---------- score ----------
BEAT = 0.5; BAR = 2.0
M = {'A2': 110.0, 'C3': 130.81, 'E3': 164.81, 'A3': 220.0, 'F2': 87.31, 'F3': 174.61, 'G2': 98.0, 'B2': 123.47, 'D3': 146.83, 'G3': 196.0, 'C4': 261.63, 'E4': 329.63, 'A4': 440.0, 'F4': 349.23, 'G4': 392.0, 'D4': 293.66, 'B3': 246.94, 'E5': 659.25, 'C5': 523.25, 'D5': 587.33, 'A5': 880.0, 'G5': 783.99, 'B4': 493.88}
PROG = [(['A2', 'E3', 'A3', 'C4'], 55.0), (['F2', 'C3', 'F3', 'A3'], 43.65), (['C3', 'G3', 'C4', 'E4'], 65.41), (['G2', 'D3', 'G3', 'B3'], 49.0)]
pad_l = np.zeros(N); pad_r = np.zeros(N); drum_l = np.zeros(N); drum_r = np.zeros(N)
# strings: dawn hum from 1.0, full from 5.0
bar = 0; t = 1.0
while t < 46.5:
    notes, root = PROG[bar % 4]
    cutoff = 700 if t < 5 else 1400 if t < 12 else 2200 if t < 33.6 else 2600 if t < 39.5 else 1800
    amp = 0.1 if t < 5 else 0.17 if t < 12 else 0.2
    st = strings([M[n] for n in notes], BAR + 0.6, amp=amp, cutoff=cutoff, seed=bar)
    add(pad_l, pad_r, st, t, pan=-0.3); add(pad_l, pad_r, st, t + 0.014, pan=0.3)
    for b in range(4):
        tb = t + b * BEAT
        if 9 <= tb < 46: add(drum_l, drum_r, bass(root, amp=0.3 if tb >= 20 else 0.22), tb)
    bar += 1; t += BAR
# piano motif (sparse, cinematic): phrases at hero, report, golden hour, end
PH = [('A4', 1), ('C5', 1), ('E5', 2), ('D5', 1), ('C5', 1), ('B4', 2), ('A4', 1), ('B4', 1), ('C5', 4)]
for start, amp in ((5.2, 0.12), (15.6, 0.1), (33.7, 0.14), (43.7, 0.12)):
    tm = start
    for n, d in PH:
        if tm + d * BEAT > min(start + 8, 47): break
        add(pad_l, pad_r, piano(M[n], d * BEAT + 0.6, amp=amp), tm, pan=0.2); tm += d * BEAT
# clock hat (ticking) during pre-hero and inspections, straight hats during the groove
t = 1.0; i = 0; kick_times = []
while t < 43.4:
    bib = i % 4
    if t < 5: add(drum_l, drum_r, clockhat(0.09), t, pan=0.4 if bib % 2 else -0.4)
    elif t < 9:
        if bib in (0, 2): add(drum_l, drum_r, taiko(0.6), t); kick_times.append(t)
        add(drum_l, drum_r, clockhat(0.07), t, pan=0.4 if bib % 2 else -0.4)
    elif t < 20:
        add(drum_l, drum_r, kick(0.6 if bib == 0 else 0.48), t); kick_times.append(t)
        if bib in (1, 3): add(drum_l, drum_r, clap(0.24), t, pan=0.1)
        add(drum_l, drum_r, hat(0.11), t, pan=0.3); add(drum_l, drum_r, hat(0.15), t + BEAT / 2, pan=0.35)
    elif t < 24.5:
        if bib in (0, 2): add(drum_l, drum_r, taiko(0.7), t); kick_times.append(t)
        add(drum_l, drum_r, clockhat(0.08), t, pan=0.4 if bib % 2 else -0.4)
    elif t < 33.6:
        add(drum_l, drum_r, kick(0.64 if bib == 0 else 0.52), t); kick_times.append(t)
        if bib in (1, 3): add(drum_l, drum_r, clap(0.3), t, pan=0.1)
        add(drum_l, drum_r, hat(0.12), t, pan=0.3); add(drum_l, drum_r, hat(0.17), t + BEAT / 2, pan=0.35)
        add(drum_l, drum_r, hat(0.08), t + BEAT / 4, pan=0.25); add(drum_l, drum_r, hat(0.08), t + 3 * BEAT / 4, pan=0.25)
        if bib == 3: add(drum_l, drum_r, hat(0.15, 0.3, open_=True), t + BEAT / 2, pan=0.4)
    elif t < 39.5:
        if bib in (0, 2): add(drum_l, drum_r, taiko(0.75), t); kick_times.append(t)
        if bib == 1: add(drum_l, drum_r, clap(0.26), t, pan=0.1)
        add(drum_l, drum_r, hat(0.1), t + BEAT / 2, pan=0.35)
    else:
        if bib in (0, 2): add(drum_l, drum_r, kick(0.48), t); kick_times.append(t)
        add(drum_l, drum_r, clockhat(0.06), t, pan=0.4 if bib % 2 else -0.4)
    i += 1; t += BEAT
# taiko fills before hits
for t0 in (4.0, 22.5, 32.0, 42.6):
    for k in range(6): add(drum_l, drum_r, taiko(0.35 + k * 0.08, 0.5, 140, 60), t0 + k * (1.0 / 6), pan=-0.5 + k * 0.2)
# sidechain
env = np.ones(N)
for kt in kick_times:
    i0 = int(kt * SR); seg = np.arange(min(int(0.4 * SR), N - i0)) / SR
    env[i0:i0 + len(seg)] *= 1 - 0.5 * np.exp(-seg / 0.1)
pad_l *= env; pad_r *= env
music_l = pad_l + drum_l; music_r = pad_r + drum_r
# braams + impacts
for t0, a in [(5.0, 0.55), (23.5, 0.6), (33.0, 0.5), (33.6, 0.4), (43.6, 0.6)]: add(music_l, music_r, braam(55, 2.4, a), t0)
for t0 in [5.0, 43.6]: add(music_l, music_r, boom(0.8), t0); add(music_l, music_r, subdrop(0.7), t0)
for t0 in [23.5, 33.0]: add(music_l, music_r, slam(0.95), t0)
for t0, d in [(3.6, 1.4), (22.2, 1.3), (31.7, 1.3), (38.3, 1.2), (42.3, 1.3)]: add(music_l, music_r, riser(d, 0.4), t0)
for t0 in [9.0, 12.0, 15.5, 20.0, 24.5, 27.5, 30.0, 39.5]: add(music_l, music_r, rev_cymbal(0.9, 0.2), t0 - 0.9)
pc = strings([M[n] for n in ['A2', 'E3', 'A3', 'C4', 'E4', 'A4']], 4.6, amp=0.32, cutoff=2600, seed=99)
add(music_l, music_r, pc, 43.6, pan=-0.3); add(music_l, music_r, pc, 43.615, pan=0.3)
add(music_l, music_r, fft_filter(noise(4.0), lo=5000, hi=12000) * np.linspace(1, 0, int(4.0 * SR)) ** 2 * 0.05, 43.6)

# ---------- sfx ----------
sfx_l = np.zeros(N); sfx_r = np.zeros(N)
# intro: wind bed 0–5, type ticks
wind = onepole_sweep(noise(5.0), 300, 900, 1.0) * 0.05 * np.sin(np.linspace(0, np.pi, int(5 * SR))) ** 0.5
add(sfx_l, sfx_r, wind, 0.0)
for c in range(26): add(sfx_l, sfx_r, keytick(0.05), 0.15 + c / 28, pan=-0.2)
# corridor flybys: one per plane passed (opening 13 planes over 1.0–5.0, eased)
import math
def ease_inout_q(x): return 2 * x * x if x < .5 else 1 - (-2 * x + 2) ** 2 / 2
for i in range(1, 14):
    # find t where camera passes plane i (camZ = i*ZS)
    frac = (i * 900 - 600) / (13 * 900 - 700 - 600)
    # invert easing numerically
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if ease_inout_q(mid) < frac: lo = mid
        else: hi = mid
    tp = 1.0 + lo * 4.0
    add(sfx_l, sfx_r, flyby(0.18 + 0.02 * i, 0.6), tp - 0.3, pan=(-0.7 if i % 2 else 0.7))
# hero: duotone resolve, altitude ticks, outline fills, cert ticks
add(sfx_l, sfx_r, whoosh(0.3, dur=0.6, f0=600, f1=5000), 5.15)
tc = 5.3
while tc < 7.1: add(sfx_l, sfx_r, tick(0.08), tc, pan=0.55); tc += 0.04 + 0.1 * ((tc - 5.3) / 1.8) ** 2
add(sfx_l, sfx_r, blip(0.2, 1200, 1200, 0.18), 7.2, pan=0.55)
for t0, t1 in [(5.9, 6.5), (6.2, 6.8), (6.6, 7.3)]:
    add(sfx_l, sfx_r, whoosh(0.16, dur=t1 - t0, f0=1500, f1=4500), t0, pan=-0.3)
for n in range(4): add(sfx_l, sfx_r, blip(0.16, 1300, 1700, 0.1), 7.0 + n * 0.18, pan=-0.4 + n * 0.25)
add(sfx_l, sfx_r, whoosh(0.3, dur=0.45, f0=5000, f1=500), 8.55, pan=0.5)
# typewriter kickers
KICK = [(9.05, 52), (12.05, 40), (15.55, 46), (20.05, 38), (24.55, 48), (27.55, 40), (30.05, 42), (39.55, 44)]
for t0, nch in KICK:
    for c in range(min(nch, 34)): add(sfx_l, sfx_r, keytick(0.045 + 0.02 * (c % 3 == 0)), t0 + c / 48, pan=-0.35)
# slider wipes
for t0 in (9.7, 10.2, 10.7, 11.2):
    add(sfx_l, sfx_r, whoosh(0.22, dur=0.4, f0=4000, f1=900), t0, pan=-0.5); add(sfx_l, sfx_r, thud(0.18), t0 + 0.38, pan=0.4)
tc = 9.6
while tc < 11.7: add(sfx_l, sfx_r, tick(0.05), tc, pan=0.6); tc += 0.06
add(sfx_l, sfx_r, blip(0.22, 1400, 1900, 0.16), 11.7, pan=0.6)
# visit ring: rotation whoosh + shutter per visit
for t0 in (12.2, 12.8, 13.4, 14.0, 14.6):
    add(sfx_l, sfx_r, whoosh(0.16, dur=0.45, f0=2500, f1=800), t0, pan=0.3); add(sfx_l, sfx_r, shutter(0.3), t0 + 0.4, pan=0.2)
# stakeholder ring: chips, deliveries, stamp, ring hum
hum = np.sin(2 * np.pi * 110 * tline(4.0)) * 0.03 * np.sin(np.linspace(0, np.pi, int(4.0 * SR))) ** 0.5
add(sfx_l, sfx_r, hum, 16.0)
for n in range(7): add(sfx_l, sfx_r, blip(0.14, 1300, 900), 16.4 + n * 0.14, pan=-0.5 + n * 0.17)
add(sfx_l, sfx_r, thud(0.35), 17.7); add(sfx_l, sfx_r, blip(0.2, 1800, 1400, 0.14), 17.75, pan=0.3)
for n in range(7): add(sfx_l, sfx_r, blip(0.16, 1700, 2300, 0.1), 18.0 + n * 0.2, pan=-0.5 + n * 0.17)
# issue: zoom whoosh, duotone resolve, alarm tag, bars
add(sfx_l, sfx_r, whoosh(0.25, dur=1.0, f0=600, f1=4000), 20.1)
add(sfx_l, sfx_r, blip(0.18, 700, 500, 0.18), 21.1, pan=0.5); add(sfx_l, sfx_r, blip(0.14, 700, 500, 0.18), 21.3, pan=0.5)
tc = 21.8
while tc < 23.1: add(sfx_l, sfx_r, tick(0.07), tc, pan=-0.4); tc += 0.05
add(sfx_l, sfx_r, blip(0.24, 500, 380, 0.3), 23.1, pan=-0.4)
# roof x-ray sweep + tag locks
scan = onepole_sweep(fft_filter(noise(1.4), lo=800), 1200, 6000, 1.0) * 0.06 * np.sin(np.linspace(0, np.pi, int(1.4 * SR))) ** 0.4
i0 = int(24.6 * SR); pan_t = np.linspace(-0.8, 0.8, len(scan))
sfx_l[i0:i0 + len(scan)] += scan * np.cos((pan_t + 1) * np.pi / 4); sfx_r[i0:i0 + len(scan)] += scan * np.sin((pan_t + 1) * np.pi / 4)
for t0, pn in ((24.95, -0.5), (24.95, -0.5), (25.6, 0.4), (25.6, 0.4)):
    add(sfx_l, sfx_r, blip(0.16, 1100, 900, 0.07), t0, pan=pn); add(sfx_l, sfx_r, blip(0.2, 1700, 1500, 0.1), t0 + 0.07, pan=pn)
# facade filmstrip ticks
for n in range(30): add(sfx_l, sfx_r, tick(0.06), 27.6 + n / 14, pan=0.5)
add(sfx_l, sfx_r, blip(0.22, 1400, 1900, 0.16), 27.6 + 30 / 14, pan=0.5)
# marketing gallery: devices arrive, orbit air
for n, t0 in enumerate([30.2, 30.45, 30.7]): add(sfx_l, sfx_r, whoosh(0.14, dur=0.35, f0=5000, f1=900), t0, pan=[0, -0.6, 0.6][n]); add(sfx_l, sfx_r, thud(0.2), t0 + 0.2, pan=[0, -0.6, 0.6][n])
# golden hour flybys (5 planes over 33.6–39.5 linear)
for i in range(1, 5):
    tp = 33.6 + (i * 1440 - 500) / (4 * 1440 + 500 - 500) * 5.9
    add(sfx_l, sfx_r, flyby(0.26, 0.8), tp - 0.4, pan=(-0.6 if i % 2 else 0.6)); add(sfx_l, sfx_r, thud(0.22), tp)
add(sfx_l, sfx_r, thud(0.22), 33.6 + 0.6)
# values + marquee
for n in range(3): add(sfx_l, sfx_r, whoosh(0.14, dur=0.5, f0=1500, f1=5000), 39.8 + n * 0.35, pan=-0.6 + n * 0.6); add(sfx_l, sfx_r, blip(0.14, 1400, 1900, 0.12), 40.5 + n * 0.35, pan=-0.6 + n * 0.6)
add(sfx_l, sfx_r, whoosh(0.2, dur=0.5, f0=800, f1=4000), 41.3)
# ending: reverse flythrough flybys + logo
for i in range(12, 0, -3):
    frac = ((13 * 900 - 700) - i * 900) / ((13 * 900 - 700) + 400)
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if ease_inout_q(mid) < frac: lo = mid
        else: hi = mid
    add(sfx_l, sfx_r, flyby(0.14, 0.7), 43.5 + lo * 3.1 - 0.35, pan=(-0.6 if i % 2 else 0.6))
add(sfx_l, sfx_r, whoosh(0.3, dur=1.2, f0=800, f1=7000), 43.65); add(sfx_l, sfx_r, blip(0.12, 900, 1300, 0.2), 45.4)
for t0 in [9.3, 12.5, 12.9, 15.8, 16.05, 20.3, 20.55, 26.2, 26.4, 27.8, 28.0, 30.2, 30.45, 45.6]: add(sfx_l, sfx_r, thud(0.16), t0)

# ---------- mix ----------
mix_l = music_l * 0.85 + sfx_l; mix_r = music_r * 0.85 + sfx_r
fo_s = int(47.0 * SR); fo = np.ones(N); fo[fo_s:] = np.linspace(1, 0, N - fo_s) ** 1.5; mix_l *= fo; mix_r *= fo
for arr in (mix_l, mix_r): np.tanh(arr * 1.15, out=arr)
peak = max(np.abs(mix_l).max(), np.abs(mix_r).max()); g = 10 ** (-1 / 20) / peak
out = np.stack([mix_l * g, mix_r * g], axis=1)
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype('<i2').tobytes())
rms = 20 * np.log10(np.sqrt((out ** 2).mean()) + 1e-9)
print(f'wrote {sys.argv[1]}  peak -1 dBFS  rms {rms:.1f} dBFS')
