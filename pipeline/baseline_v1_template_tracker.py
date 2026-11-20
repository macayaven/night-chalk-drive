#!/usr/bin/env python3
"""v1 baseline 3-DoF template matcher (uniform scale + translation on
cartoonized frames), preserved alongside the v2 8-DoF planar tracker."""
import json, os, sys
import cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import media_base as B

HEAD = os.path.join(B.WORK, "final3_parts", "head_cartoon.mp4")
# panel, start, end, seed_time, seed content quad measured on zoomed seed frames
SPECS = [
 (1, 11.9, 12.6, 12.25, [(500,622),(675,618),(675,695),(502,698)]),
 (2, 91.15, 91.85, 91.5, [(405,648),(575,643),(575,725),(405,727)]),
 (3, 141.4, 141.85, 141.5, [(410,550),(537,545),(537,637),(410,642)]),
 (4, 204.2, 204.6, 204.25, [(487,602),(632,602),(632,697),(487,697)]),
 (5, 237.9, 238.15, 238.0, [(505,662),(645,660),(645,748),(505,750)]),
 (6, 240.65, 241.35, 241.0, [(502,607),(670,607),(670,702),(502,702)]),
]

def main():
    tmp = os.path.join(B.WORK, "tmpl_parts"); os.makedirs(tmp, exist_ok=True)
    out = {}; diag = os.path.join(B.WORK, "track_diag"); os.makedirs(diag, exist_ok=True)
    for panel, start, end, seed_t, quad in SPECS:
        seg = os.path.join(tmp, f"seg{panel}.mp4")
        B.extract(HEAD, start, end - start, seg)
        frames, fps = T_read(seg)
        seed_i = int(round((seed_t - start) * 24)); seed_i = max(0, min(seed_i, len(frames) - 1))
        q0 = np.array(quad, np.float32)
        x0 = int(max(0, q0[:,0].min() - 24)); y0 = int(max(0, q0[:,1].min() - 24))
        x1 = int(min(720, q0[:,0].max() + 24)); y1 = int(min(960, q0[:,1].max() + 24))
        tpl0 = cv2.cvtColor(frames[seed_i][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        tpls = {s: cv2.resize(tpl0, (int(tpl0.shape[1] * s), int(tpl0.shape[0] * s)), interpolation=cv2.INTER_LINEAR)
                for s in (0.85, 0.95, 1.05, 1.15, 1.25)}
        tpls[1.0] = tpl0
        quads = []; cur_s = 1.0; prev_origin = np.array([x0, y0], np.float32); prev_s = 1.0
        for f in frames:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sx0 = max(0, x0 - 115); sy0 = max(0, y0 - 85)
            sx1 = min(720, x1 + 115); sy1 = min(960, y1 + 85)
            best = None
            for s, tpl in tpls.items():
                if abs(s - cur_s) > 0.21: continue
                res = cv2.matchTemplate(g[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, ml = cv2.minMaxLoc(res)
                if best is None or mx > best[0]: best = (mx, s, ml)
            mx, s, ml = best; cur_s = s
            origin = np.array([sx0 + ml[0], sy0 + ml[1]], np.float32)
            q = (q0 - np.array([x0, y0], np.float32)) * s + origin
            quads.append(q.tolist())
            print(panel, "score", round(float(mx), 3), "s", s, "loc", origin.astype(int).tolist()) if len(quads) % 8 == 1 else None
        out[str(panel)] = dict(start=start, fps=24.0, quads=quads)
        imgs = []
        for idx in (0, len(frames)//2, len(frames)-1):
            f = frames[idx].copy(); cv2.polylines(f, [np.array(quads[idx]).astype(np.int32)], True, (0,0,255), 3)
            imgs.append(cv2.resize(f, (360, 480)))
        cv2.imwrite(os.path.join(diag, f"panel{panel}.jpg"), np.hstack(imgs))
    with open(os.path.join(B.WORK, "track_quads.json"), "w") as fh: json.dump(out, fh)
    print("wrote track_quads.json (template)")

def T_read(path):
    import subprocess
    dec = subprocess.Popen(["ffmpeg","-v","error","-i",path,"-vf","fps=24","-f","rawvideo","-pix_fmt","bgr24","-"],
                           stdout=subprocess.PIPE)
    n = 720 * 960 * 3; frames = []
    while True:
        buf = dec.stdout.read(n)
        if len(buf) < n: break
        frames.append(np.frombuffer(buf, np.uint8).reshape(960, 720, 3).copy())
    dec.wait(); return frames, 24.0

if __name__ == "__main__":
    main()
