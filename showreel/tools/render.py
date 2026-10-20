"""Render showreel.html to PNG frames with headless Chromium.

usage: python3 render.py <html> <outdir> [--fps 30] [--dur 48] [--times 1.5,2.0,...]
"""
import sys, os, asyncio, argparse, time
from playwright.async_api import async_playwright

ap = argparse.ArgumentParser()
ap.add_argument('html'); ap.add_argument('outdir')
ap.add_argument('--fps', type=int, default=30); ap.add_argument('--dur', type=float, default=48.0)
ap.add_argument('--times', default=None, help='comma list of seconds to render as preview')
ap.add_argument('--start', type=int, default=0)
ap.add_argument('--width', type=int, default=1920); ap.add_argument('--height', type=int, default=1080)
ap.add_argument('--query', default='', help='query string appended to the page URL, e.g. ?v=1')
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

async def main():
    async with async_playwright() as p:
        exe = '/opt/pw-browsers/chromium' if os.path.exists('/opt/pw-browsers/chromium') else None
        b = await p.chromium.launch(executable_path=exe, args=['--disable-gpu', '--font-render-hinting=none', '--force-color-profile=srgb', '--hide-scrollbars'])
        pg = await b.new_page(viewport={'width': a.width, 'height': a.height}, device_scale_factor=1)
        await pg.goto('file://' + os.path.abspath(a.html) + a.query)
        pg.on('pageerror', lambda e: print('PAGE ERROR:', e, flush=True))
        await pg.evaluate('document.fonts.ready')
        await pg.evaluate('''() => Promise.all([...document.images].map(i => i.complete ? 1 : new Promise(r => { i.onload = i.onerror = r; })))''')
        # also force background images to load by seeking through every scene once
        for t in [0, 1.2, 6, 11, 14.5, 19, 23.5, 27, 30, 33, 35, 38, 42, 45]:
            await pg.evaluate(f'seek({t})')
        await pg.wait_for_timeout(1500)
        if a.times:
            for t in [float(x) for x in a.times.split(',')]:
                await pg.evaluate(f'seek({t})')
                await pg.screenshot(path=os.path.join(a.outdir, f'p_{t:05.2f}.png'))
        else:
            n = int(round(a.dur * a.fps)); t0 = time.time()
            for i in range(a.start, n):
                await pg.evaluate(f'seek({i / a.fps})')
                await pg.screenshot(path=os.path.join(a.outdir, f'f_{i:05d}.png'))
                if i % 150 == 0:
                    print(f'{i}/{n} {time.time() - t0:.0f}s', flush=True)
        await b.close()
asyncio.run(main())
