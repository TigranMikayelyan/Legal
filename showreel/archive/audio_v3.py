"""Synthesize the showreel v3 soundtrack (120 BPM) with numpy only.
usage: python3 audio.py <out.wav>
Event times mirror the timeline in showreel.html.
"""
import sys, wave, numpy as np
SR = 48000; DUR = 48.0; N = int(SR * DUR)
rng = np.random.default_rng(11)

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

def hat(amp=0.12, dur=0.09, open_=False):
    n = int(dur * SR); return fft_filter(noise(dur), lo=7000) * env_exp(n, 0.09 if open_ else 0.018) * amp

def boom(amp=0.9, dur=1.8, f=48):
    t = tline(dur); sub = np.sin(2 * np.pi * (f + 30 * np.exp(-t / 0.08)) * t) * np.exp(-t / 0.55)
    air = fft_filter(noise(dur), lo=200, hi=3000) * np.exp(-t / 0.12) * 0.5
    return np.tanh((sub + air) * 1.5) * amp

def subdrop(amp=0.8, dur=1.2, f0=110, f1=30):
    t = tline(dur); f = f1 + (f0 - f1) * np.exp(-t / 0.25); ph = 2 * np.pi * np.cumsum(f) / SR
    return np.tanh(np.sin(ph) * 2) * np.exp(-t / 0.5) * amp

def slam(amp=1.0):
    """hard impact for the teal slam frames: boom + noise burst + sub drop"""
    out = np.zeros(int(1.8 * SR)); b = boom(1.0, 1.8); out[:len(b)] += b
    nb = fft_filter(noise(0.25), lo=300, hi=9000) * env_exp(int(0.25 * SR), 0.05); out[:len(nb)] += nb * 0.9
    sd = subdrop(0.9); out[:len(sd)] += sd
    return np.tanh(out * 1.2) * amp

def whoosh(amp=0.35, dur=0.5, f0=6000, f1=400):
    t = tline(dur); x = onepole_sweep(fft_filter(noise(dur), lo=250), f0, f1, 1.3)
    return x * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5 * amp

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

def pluck(f, dur=0.3, amp=0.08):
    t = tline(dur); tri = 2 / np.pi * np.arcsin(np.sin(2 * np.pi * f * t)); s2 = np.sin(2 * np.pi * f * 2 * t) * 0.35
    return (tri + s2) * np.exp(-t / 0.1) * amp

def lead(f, dur, amp=0.12):
    t = tline(dur); vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.25, 0, 1)
    saw = 2 * ((f * vib * t) % 1) - 1; saw2 = 2 * ((f * 1.005 * vib * t) % 1) - 1
    x = fft_filter((saw + saw2) / 2, hi=2800, slope=1.2)
    a = int(0.02 * SR); r = int(0.12 * SR); e = np.ones(len(t)); e[:a] = np.linspace(0, 1, a); e[-r:] = np.linspace(1, 0, r)
    return x * e * amp

def bass(f, dur=0.45, amp=0.4):
    t = tline(dur); x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / 0.08)
    return np.tanh(x * 1.7) * np.exp(-t / 0.22) * amp

def pad_chord(freqs, dur, amp=0.1, cutoff=800, seed=0):
    n = int(dur * SR); t = np.arange(n) / SR; out = np.zeros(n); r = np.random.default_rng(seed)
    for f in freqs:
        for det in (-0.004, 0.0, 0.004):
            ph = r.uniform(0, 1); out += (2 * ((f * (1 + det) * t + ph) % 1) - 1) / 3
    out = fft_filter(out / len(freqs), hi=cutoff, slope=1.2)
    e = np.ones(n); a = int(0.3 * SR); d = int(0.3 * SR)
    e[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2; e[-d:] = np.cos(np.linspace(0, np.pi / 2, d)) ** 2
    return out * e * amp

# ---------------- music (120 BPM) ----------------
BEAT = 0.5; BAR = 2.0
M = {'A2': 110.0, 'C3': 130.81, 'E3': 164.81, 'A3': 220.0, 'F2': 87.31, 'F3': 174.61, 'G2': 98.0, 'B2': 123.47, 'D3': 146.83, 'G3': 196.0, 'C4': 261.63, 'E4': 329.63, 'A4': 440.0, 'F4': 349.23, 'G4': 392.0, 'D4': 293.66, 'B3': 246.94, 'E5': 659.25, 'C5': 523.25, 'D5': 587.33, 'A5': 880.0, 'G5': 783.99}
PROG = [(['A2', 'C3', 'E3', 'A3'], 55.0, ['A3', 'C4', 'E4', 'A4']), (['F2', 'A2', 'C3', 'F3'], 43.65, ['F3', 'A3', 'C4', 'F4']),
        (['C3', 'E3', 'G3', 'C4'], 65.41, ['C4', 'E4', 'G4', 'C4']), (['G2', 'B2', 'D3', 'G3'], 49.0, ['G3', 'B3', 'D4', 'G4'])]
pad_l = np.zeros(N); pad_r = np.zeros(N); drum_l = np.zeros(N); drum_r = np.zeros(N)
# pads from 4.0 (pre-flight) to 46
bar = 0; t = 4.0
while t < 46.0:
    notes, root, arpn = PROG[bar % 4]
    cutoff = 800 if t < 10 else 1600 if t < 21 else 2400 if t < 38.5 else 1500
    amp = 0.14 if t < 10 else 0.2 if t < 38.5 else 0.18
    pc = pad_chord([M[n] for n in notes], BAR + 0.3, amp=amp, cutoff=cutoff, seed=bar)
    add(pad_l, pad_r, pc, t, pan=-0.25); add(pad_l, pad_r, pc, t + 0.012, pan=0.25)
    for b in range(4):
        tb = t + b * BEAT
        if 10 <= tb < 46: add(drum_l, drum_r, bass(root, amp=0.32 if tb >= 21 else 0.24), tb)
        if 13 <= tb < 38.5:
            for s_ in range(2):
                ts = tb + s_ * BEAT / 2; f = M[arpn[(b * 2 + s_) % 4]] * (2 if (b * 2 + s_) % 4 == 3 else 1)
                add(pad_l, pad_r, pluck(f, amp=0.08 if s_ else 0.11), ts, pan=0.5 * np.sin(ts * 2.1))
    bar += 1; t += BAR
# lead motif across kinetic run and the values
MOTIF = [('E4', 1), ('G4', 1), ('A4', 2), ('G4', 1), ('E4', 1), ('D4', 2), ('C4', 1), ('D4', 1), ('E4', 4)]
for start, amp in ((33.5, 0.12), (38.8, 0.08)):
    tm = start
    for n, d in MOTIF:
        if tm + d * BEAT > 42.3: break
        add(pad_l, pad_r, lead(M[n], d * BEAT * 0.95, amp=amp), tm, pan=0.15); tm += d * BEAT
# sub drone under logo + preflight
dr = np.sin(2 * np.pi * 55 * tline(8.0)) * np.sin(np.linspace(0, np.pi, int(8.0 * SR))) ** 0.6 * 0.16
add(pad_l, pad_r, dr, 2.0)
# drums
kick_times = []
# montage: kick on every cut 0–2
for k in range(8): add(drum_l, drum_r, kick(0.7 if k % 2 == 0 else 0.5, f0=170), k * 0.25); kick_times.append(k * 0.25)
t = 6.0; i = 0
while t < 42.4:
    bib = i % 4
    if 6.0 <= t < 10: k = 0.5 if bib in (0, 2) else None       # half-time during takeoff
    elif t < 21: k = 0.62 if bib == 0 else 0.5                      # four on the floor
    elif t < 24: k = 0.66 if bib == 0 else 0.54
    elif t < 38.5: k = 0.66 if bib == 0 else 0.54
    else: k = 0.5 if bib in (0, 2) else None
    if k: add(drum_l, drum_r, kick(k), t); kick_times.append(t)
    if 10 <= t < 38.5 and bib in (1, 3): add(drum_l, drum_r, clap(0.24 if t < 21 else 0.32), t, pan=0.1)
    if 10 <= t < 38.5:
        add(drum_l, drum_r, hat(0.12), t, pan=0.3); add(drum_l, drum_r, hat(0.17), t + BEAT / 2, pan=0.35)
        if t >= 21:
            add(drum_l, drum_r, hat(0.08), t + BEAT / 4, pan=0.25); add(drum_l, drum_r, hat(0.08), t + 3 * BEAT / 4, pan=0.25)
            if bib == 3: add(drum_l, drum_r, hat(0.15, 0.3, open_=True), t + BEAT / 2, pan=0.4)
    i += 1; t += BEAT
# sidechain
env = np.ones(N)
for kt in kick_times:
    i0 = int(kt * SR); seg = np.arange(min(int(0.4 * SR), N - i0)) / SR
    env[i0:i0 + len(seg)] *= 1 - 0.6 * np.exp(-seg / 0.1)
pad_l *= env; pad_r *= env
music_l = pad_l + drum_l; music_r = pad_r + drum_r
# risers / reverse cymbals / impacts
for t0, d in [(8.8, 1.2), (22.6, 1.4), (31.2, 1.2), (36.3, 1.2), (41.4, 1.2)]: add(music_l, music_r, riser(d, 0.4), t0)
for t0 in [10.0, 13.0, 16.0, 21.0, 25.0, 27.5, 30.0, 33.0, 38.5, 42.6]: add(music_l, music_r, rev_cymbal(0.9, 0.22), t0 - 0.9)
for t0 in [2.0, 6.3, 42.6]: add(music_l, music_r, boom(0.8), t0); add(music_l, music_r, subdrop(0.7), t0)
for t0 in [3.5, 24.0, 32.4, 37.5]: add(music_l, music_r, slam(0.95), t0)
# final chord swell
pc = pad_chord([M[n] for n in ['A2', 'E3', 'A3', 'C4', 'E4']], 4.6, amp=0.3, cutoff=2200, seed=99)
add(music_l, music_r, pc, 42.6, pan=-0.3); add(music_l, music_r, pc, 42.615, pan=0.3)
add(music_l, music_r, fft_filter(noise(4.0), lo=5000, hi=12000) * np.linspace(1, 0, int(4.0 * SR)) ** 2 * 0.05, 42.6)

# ---------------- sound effects ----------------
sfx_l = np.zeros(N); sfx_r = np.zeros(N)
# montage shutters
for k in range(8): add(sfx_l, sfx_r, shutter(0.25), k * 0.25, pan=-0.5 + k * 0.14)
# logo letters
for k in range(8): add(sfx_l, sfx_r, blip(0.12, 1500 + k * 90, 1100 + k * 90, 0.06), 2.45 + k * 0.05, pan=-0.5 + k * 0.14)
# preflight: typewriter + checks + cleared beeps
KICK = [(4.05, 16), (7.8, 48), (10.05, 38), (13.05, 18), (16.05, 28), (21.5, 20), (21.6, 13), (25.05, 32), (27.55, 32), (30.05, 26), (33.0, 24), (38.55, 24)]
for t0, nch in KICK:
    for c in range(min(nch, 32)): add(sfx_l, sfx_r, keytick(0.05 + 0.02 * (c % 3 == 0)), t0 + c / 40, pan=-0.35)
for n in range(4):
    t0 = 4.5 + n * 0.4; add(sfx_l, sfx_r, whoosh(0.1, dur=0.25, f0=3000, f1=800), t0 - 0.3, pan=-0.4)
    add(sfx_l, sfx_r, blip(0.2, 1300, 1700, 0.12), t0, pan=-0.3); add(sfx_l, sfx_r, tick(0.1), t0 + 0.05, pan=-0.3)
for n in range(5): add(sfx_l, sfx_r, blip(0.16, 1000, 1000, 0.1), 6.0 + n * 0.17, pan=-0.3)
# takeoff: motor spool + altitude ticks
sp = np.sin(2 * np.pi * np.cumsum(120 * 2 ** (np.linspace(0, 1.8, int(2.0 * SR)))) / SR) * np.linspace(0.2, 1, int(2.0 * SR)) * 0.06
sp += fft_filter(noise(2.0), lo=300, hi=3000) * np.linspace(0, 1, int(2.0 * SR)) ** 2 * 0.05
add(sfx_l, sfx_r, sp * np.where(tline(2.0) > 1.95, 0, 1), 6.3)
tc = 6.4
while tc < 8.2: add(sfx_l, sfx_r, tick(0.08), tc, pan=0.55); tc += 0.04 + 0.1 * ((tc - 6.4) / 1.8) ** 2
add(sfx_l, sfx_r, blip(0.22, 1200, 1200, 0.18), 8.3, pan=0.55)
# visit cuts: shutter + servo
for k in range(5): add(sfx_l, sfx_r, shutter(0.32), 10 + k * 0.6, pan=0.3); add(sfx_l, sfx_r, whoosh(0.08, dur=0.25, f0=1200, f1=2400), 10.05 + k * 0.6, pan=0.3)
# carousel: card lands + percent ticks
for n in range(5): add(sfx_l, sfx_r, thud(0.14), 13.1 + n * 0.12, pan=-0.6 + n * 0.3)
tc = 13.3
while tc < 15.6: add(sfx_l, sfx_r, tick(0.05), tc, pan=0.6); tc += 0.06
add(sfx_l, sfx_r, blip(0.22, 1400, 1900, 0.16), 15.6, pan=0.6)
# hub: card pop, chips, pulses, deliveries, stamp
add(sfx_l, sfx_r, whoosh(0.18, dur=0.4, f0=2500, f1=5000), 16.3, pan=0.3)
for n in range(7): add(sfx_l, sfx_r, blip(0.14, 1300, 900), 17.0 + n * 0.18, pan=-0.5 + n * 0.17); add(sfx_l, sfx_r, whoosh(0.06, dur=0.25, f0=2000, f1=5000), 17.0 + n * 0.18)
add(sfx_l, sfx_r, thud(0.35), 18.3); add(sfx_l, sfx_r, blip(0.2, 1800, 1400, 0.14), 18.35, pan=0.3)
for n in range(7):
    add(sfx_l, sfx_r, whoosh(0.09, dur=0.5, f0=1500, f1=4500), 18.6 + n * 0.12, pan=-0.5 + n * 0.17)
    add(sfx_l, sfx_r, blip(0.16, 1700, 2300, 0.1), 19.3 + n * 0.12, pan=-0.5 + n * 0.17)
# cost race: panels slam, VS, counter ticks, alarm
add(sfx_l, sfx_r, whoosh(0.3, dur=0.5, f0=5000, f1=500), 21.0, pan=-0.5); add(sfx_l, sfx_r, whoosh(0.3, dur=0.5, f0=5000, f1=500), 21.1, pan=0.5)
add(sfx_l, sfx_r, thud(0.3), 21.5); add(sfx_l, sfx_r, thud(0.3), 21.6); add(sfx_l, sfx_r, blip(0.2, 900, 1300, 0.15), 21.65)
tc = 22.0
while tc < 23.4: add(sfx_l, sfx_r, tick(0.07), tc, pan=0.5); tc += 0.05
for n in range(6): add(sfx_l, sfx_r, blip(0.12, 620, 620, 0.12), 22.3 + n * 0.2, pan=0.5)
add(sfx_l, sfx_r, blip(0.24, 500, 380, 0.3), 23.4, pan=0.5)
# roof tiles + tags
for n in range(60): add(sfx_l, sfx_r, tick(0.04), 25.1 + n * (1.1 / 60), pan=-0.6 + (n % 10) * 0.13)
add(sfx_l, sfx_r, blip(0.2, 1400, 1900, 0.16), 26.25, pan=0.3)
for n in range(4):
    t0 = 26.3 + n * 0.22; add(sfx_l, sfx_r, blip(0.16, 1100, 900, 0.07), t0, pan=0.5); add(sfx_l, sfx_r, blip(0.2, 1700, 1500, 0.1), t0 + 0.07, pan=0.5)
# facade flips + loupe + tags
for n in range(30): add(sfx_l, sfx_r, whoosh(0.07, dur=0.12, f0=4000, f1=1500), 27.6 + n * (1.3 / 30), pan=-0.5 + (n % 5) * 0.25); add(sfx_l, sfx_r, tick(0.05), 27.6 + n * (1.3 / 30) + 0.06)
add(sfx_l, sfx_r, blip(0.22, 1400, 1900, 0.16), 28.9, pan=0.3)
add(sfx_l, sfx_r, blip(0.12, 600, 1200, 0.25), 29.0, pan=0.4)
for n, t0 in enumerate([29.1, 29.4, 29.7]): add(sfx_l, sfx_r, blip(0.16, 1100, 900, 0.07), t0, pan=0.5); add(sfx_l, sfx_r, blip(0.2, 1700, 1500, 0.1), t0 + 0.07, pan=0.5)
# marketing devices hard-cut
for n, t0 in enumerate([30.2, 30.8, 31.4]): add(sfx_l, sfx_r, thud(0.3), t0, pan=[0, -0.6, 0.6][n]); add(sfx_l, sfx_r, whoosh(0.14, dur=0.3, f0=5000, f1=900), t0, pan=[0, -0.6, 0.6][n])
# kinetic words
for k in range(5):
    t0 = 33.5 + k * 0.8; add(sfx_l, sfx_r, shutter(0.3), t0, pan=-0.4 + k * 0.2); add(sfx_l, sfx_r, whoosh(0.26, dur=0.35, f0=6000, f1=700), t0 - 0.05, pan=(-1 if k % 2 else 1) * 0.5); add(sfx_l, sfx_r, thud(0.22), t0 + 0.05)
# values icons draw + marquee
for n in range(3): add(sfx_l, sfx_r, whoosh(0.14, dur=0.5, f0=1500, f1=5000), 38.8 + n * 0.35, pan=-0.6 + n * 0.6); add(sfx_l, sfx_r, blip(0.14, 1400, 1900, 0.12), 39.5 + n * 0.35, pan=-0.6 + n * 0.6)
add(sfx_l, sfx_r, whoosh(0.2, dur=0.5, f0=800, f1=4000), 40.3)
# logo
add(sfx_l, sfx_r, whoosh(0.3, dur=1.2, f0=800, f1=7000), 42.65); add(sfx_l, sfx_r, blip(0.12, 900, 1300, 0.2), 43.5)
# headline thuds
for t0 in [8.0, 8.15, 8.7, 10.3, 10.55, 13.2, 13.4, 16.2, 16.45, 25.4, 25.6, 27.8, 28.0, 30.2, 30.45, 33.05, 43.7]:
    add(sfx_l, sfx_r, thud(0.16), t0)

# ---------------- mix ----------------
mix_l = music_l * 0.85 + sfx_l; mix_r = music_r * 0.85 + sfx_r
fo_s = int(47.0 * SR); fo = np.ones(N); fo[fo_s:] = np.linspace(1, 0, N - fo_s) ** 1.5; mix_l *= fo; mix_r *= fo
for arr in (mix_l, mix_r): np.tanh(arr * 1.15, out=arr)
peak = max(np.abs(mix_l).max(), np.abs(mix_r).max()); g = 10 ** (-1 / 20) / peak
out = np.stack([mix_l * g, mix_r * g], axis=1)
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767).astype('<i2').tobytes())
rms = 20 * np.log10(np.sqrt((out ** 2).mean()) + 1e-9)
print(f'wrote {sys.argv[1]}  peak -1 dBFS  rms {rms:.1f} dBFS')
