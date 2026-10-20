import sys, os
from PIL import Image
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
names = {
 '1.jpg':'golf','2.jpg':'cap_slab','3.jpg':'cap_steel','4.jpg':'cap_plaza','5.jpg':'cap_house','6.jpg':'cap_foundation',
 '7.jpg':'opening_tower','8.jpg':'stage_sitework','9.jpg':'stage_foundations','10.jpg':'stage_framing','11.jpg':'stage_roof',
 '12.jpg':'townhouses','13.jpg':'office','14.jpg':'res_tower_dusk','15.jpg':'piccolos','16.jpg':'roof_hvac',
 '20.jpg':'tiltup_warehouse','21.jpg':'waterfront_sunset','22.webp':'industrial_warehouse','23.webp':'facade_closeup',
}
for f, n in names.items():
    im = Image.open(os.path.join(src, f)).convert('RGB')
    w, h = im.size
    if max(w, h) > 2400:
        s = 2400 / max(w, h); im = im.resize((round(w*s), round(h*s)), Image.LANCZOS)
    im.save(os.path.join(dst, n + '.jpg'), quality=93, subsampling=0, optimize=True)
    print(n, im.size)
# logo: key out flat dark background of 18.webp -> teal mark + white wordmark on transparent
im = Image.open(os.path.join(src, '18.webp')).convert('RGB')
a = np.asarray(im).astype(np.float32)
bg = np.median(a[:50, :50].reshape(-1, 3), axis=0)
print('bg colour', bg)
d = np.sqrt(((a - bg) ** 2).sum(axis=2))
alpha = np.clip((d - 8) / 40.0, 0, 1)  # soft key
# un-premultiply colour against bg so edges stay clean
rgb = np.clip((a - bg * (1 - alpha[..., None])) / np.maximum(alpha[..., None], 1e-3), 0, 255)
rgb = np.where(alpha[..., None] > 0.02, rgb, 0)
out = np.dstack([rgb, alpha * 255]).astype(np.uint8)
Image.fromarray(out, 'RGBA').save(os.path.join(dst, 'logo_full.png'))
# mark only (crop above the wordmark)
ys = np.where(alpha.max(axis=1) > 0.5)[0]; xs = np.where(alpha.max(axis=0) > 0.5)[0]
print('logo bbox', xs.min(), ys.min(), xs.max(), ys.max())
# find gap row between mark and wordmark
rows = alpha.max(axis=1) > 0.5
gap = [y for y in range(ys.min(), ys.max()) if not rows[y]]
split = gap[len(gap)//2] if gap else ys.max()
mark = Image.fromarray(out, 'RGBA').crop((xs.min()-20, ys.min()-20, xs.max()+20, split))
mark.save(os.path.join(dst, 'logo_mark.png')); print('mark', mark.size, 'split row', split)
word = Image.fromarray(out, 'RGBA').crop((xs.min()-20, split, xs.max()+20, ys.max()+20))
word.save(os.path.join(dst, 'logo_word.png')); print('word', word.size)
