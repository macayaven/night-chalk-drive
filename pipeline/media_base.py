#!/usr/bin/env python3
"""V2: continuous story cut (going -> elevator -> back), colour chalk night.

Story: every second of the 5 chronological clips appears, in order, with no
sampled jumps. Roads at 9x (hyperlapse), story beats slower:
  clip 1439 from 70s (park, get out, EV charger, corridor, elevator, face) 3x
  clip 1446 last ~50s (gate, enter parking) 5x
  clip 1453 first 40s (in garage, set off again) 4x
Result retimed to exactly 144 s to fit the edited soundtracks.

Looks (all preserve the night: black stays black, colour retained):
 01 colour chalk  - posterized warm/cool chalk, sharp dark edges, 24 fps
 02 mixed         - graded real <-> colour chalk, 18 s cycle, 0.8 s morphs
 03 abstract      - stronger chalk + slow hue drift + light trails, still
                    a recognizable night drive (no kaleidoscope)
Each in forward and backwards-picture versions; music always forward.
Audio: per-style edited track, fades, static gain to -14 LUFS / -1.3 dBTP.
"""
import json, os, subprocess, sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths

ROOT, VID, WORK, OUT, MUSIC = paths.ROOT, paths.VID, paths.WORK, paths.OUT, paths.MUSIC
W, H, FPS, LEN = 720, 960, 24, 288

def run(cmd, quiet=False):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-4000:]); raise SystemExit(f"failed: {cmd[:4]}")
    if not quiet: print("+", os.path.basename(cmd[-2] if len(cmd)>1 else cmd[0]), flush=True)
    return r

def dur(path):
    r = run(["ffprobe","-v","error","-show_entries","format=duration","-of","json",path], quiet=True)
    return float(json.loads(r.stdout)["format"]["duration"])

ORDER = ["1453", "1430", "1460", "1446", "1439"]  # user-corrected chronology
def clips():
    by_id = {}
    for name in sorted(os.listdir(VID)):
        if name.lower().endswith((".mov",".mp4")):
            p = os.path.join(VID, name)
            r = run(["ffprobe","-v","error","-show_entries","format=duration","-of","json",p], quiet=True)
            d = json.loads(r.stdout)["format"]
            cid = name.split("-")[1].split("_")[0]
            by_id[cid] = (cid, name, p, float(d["duration"]))
    return [by_id[c] for c in ORDER]

# by clip id: list of (start, end-or-None, speed), contiguous, no gaps.
# Speeds are HALF the approved sample's (50% less jumpy hyperlapse).
PLAN = {
 "1453": [(0, 40, 2), (40, None, 4.5)],                  # in garage -> set off
 "1430": [(0, None, 4.5)],
 "1460": [(0, None, 4.5)],
 "1446": [(0, 250, 4.5), (250, None, 2.5)],              # gate -> enter parking
 "1439": [(0, 70, 3.5), (70, 105, 1.5), (105, None, 1.0)],  # finale: park -> walk -> elevator/face
}
FACE_CLIP, FACE_SRC_START = "1439", 105  # cubist treatment from here to the end

def face_start_time(cs, raw_total):
    """Output-timeline start (after retime to LEN) of the face finale chunk."""
    t = 0.0
    for cid, _n, _p, d in cs:
        for s, e, sp in PLAN[cid]:
            if cid == FACE_CLIP and s == FACE_SRC_START:
                return t * LEN / raw_total
            ee = d if e is None else min(e, d); t += (ee - s) / sp
    raise ValueError("face chunk not found")

def build_story():
    final = os.path.join(WORK, "story288.mp4")
    if os.path.exists(final): return final
    cs = clips()
    print("clips:", [(c[1], round(c[3],1)) for c in cs])
    cdir = os.path.join(WORK, "story_chunks288"); os.makedirs(cdir, exist_ok=True)
    chunks, expected = [], 0.0
    vf_tpl = ("setpts=(PTS-STARTPTS)/{sp},fps=24,scale=720:960:force_original_aspect_ratio=increase,"
              "crop=720:960,eq=gamma=1.12:saturation=1.05,format=yuv420p")
    for ci, (cid, name, path, d) in enumerate(cs):
        for si, (s, e, sp) in enumerate(PLAN[cid]):
            e = d if e is None else min(e, d); length = e - s; expected += length / sp
            out = os.path.join(cdir, f"c{ci}_{si}.mp4"); chunks.append(out)
            if not os.path.exists(out):
                run(["ffmpeg","-hide_banner","-loglevel","error","-hwaccel","videotoolbox",
                     "-ss",str(s),"-t",f"{length:.3f}","-i",path,"-an","-vf",vf_tpl.format(sp=sp),
                     "-c:v","libx264","-crf","13","-preset","veryfast",out,"-y"])
            print(f"chunk {name} {s:.0f}-{e:.0f}s @{sp}x -> {length/sp:.1f}s")
    lst = os.path.join(cdir, "list.txt")
    with open(lst, "w") as f:
        for c in chunks: f.write(f"file '{c}'\n")
    raw = os.path.join(WORK, "story_raw288.mp4")
    run(["ffmpeg","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",lst,
         "-c:v","libx264","-crf","13","-preset","veryfast","-r","24",raw,"-y"])
    rd = dur(raw); print(f"story raw {rd:.2f}s (expected {expected:.1f}) -> retime to {LEN}")
    run(["ffmpeg","-hide_banner","-loglevel","error","-i",raw,"-vf",
         f"setpts={LEN/rd:.6f}*(PTS-STARTPTS),fps=24,format=yuv420p","-an",
         "-c:v","libx264","-crf","13","-preset","veryfast",final,"-y"])
    print("story288", dur(final), "face finale at", round(face_start_time(cs, rd), 2)); return final

def cartoon_body(src, dst, step=28, sat=2.0, con=1.14, extra_yuv="", shift=""):
    """Drawn cartoon: flat cel colour x sketch-tone shading x bold black lines.
    No photographic detail is blended back. Cel base is black at night, so the
    multiplied sketch tone models form without lifting the night to paper."""
    half = step // 2; p = src
    lut = f"lutrgb=r='trunc(val/{step})*{step}+{half}':g='trunc(val/{step})*{step}+{half}':b='trunc(val/{step})*{step}+{half}'"
    return (
     f"[{src}]fps=12,split=3[{p}C][{p}T][{p}E];"
     f"[{p}C]format=yuv420p,bilateral=sigmaS=7:sigmaR=0.34:planes=15,"
     f"eq=contrast={con}:saturation={sat}:gamma=1.45:brightness=0.05,colortemperature=temperature=4800:mix=0.65:pl=0.3{extra_yuv},format=gbrp,{lut}[{p}cel];"
     f"[{p}T]split=3[{p}Ti][{p}Tg][{p}Ts];"
     f"[{p}Ti]format=gray,eq=gamma=1.35:contrast=1.05:brightness=0.05,negate,gblur=sigma=6[{p}inv];"
     f"[{p}Tg]format=gray,eq=gamma=1.35:contrast=1.05:brightness=0.05[{p}gry];"
     f"[{p}gry][{p}inv]blend=all_mode=dodge[{p}sk0];"
     f"[{p}Ts]format=gray,eq=gamma=2.0:brightness=0.30:contrast=0.92[{p}shd];"
     f"[{p}sk0][{p}shd]blend=all_expr='A*0.45+B*0.55',eq=brightness=0.26:contrast=1.05,format=gbrp[{p}tone];"
     f"[{p}cel][{p}tone]blend=all_expr='A*(0.30+0.70*B/255)'{shift}[{p}col];"
     f"[{p}E]format=gray,eq=gamma=2.0:contrast=1.15,edgedetect=low=0.03:high=0.16,negate,erosion,eq=contrast=1.7:brightness=0.04,format=gbrp[{p}edge];"
     f"[{p}col][{p}edge]blend=all_mode=multiply,fps=24[{dst}]"
    )

CARTOON = "[0:v]null[vin];" + cartoon_body("vin", "v") + ";[v]format=yuv420p,noise=c0s=14:c0f=t+u[vv]"
ABSTRACT = ("[0:v]null[vin];" + cartoon_body("vin", "v", step=38, sat=2.5, con=1.22,
                       extra_yuv=",hue=h='40*sin(2*PI*t/14)':s=1.3,lagfun=decay=0.82",
                       shift=",rgbashift=rh=3:bh=-3")
            + ";[v]format=yuv420p,noise=c0s=12:c0f=t+u[vv]")
MIX_EXPR = ("if(lt(mod(T\\,18)\\,9),0,if(lt(mod(T\\,18)\\,9.8),(mod(T\\,18)-9)/0.8,"
            "if(lt(mod(T\\,18)\\,17.2),1,(18-mod(T\\,18))/0.8)))")
MIXED = ("[0:v]split=2[rin][cin];"
         "[rin]fps=24,format=yuv420p,eq=contrast=1.12:saturation=1.30:gamma=1.05,"
         "colortemperature=temperature=5200:mix=0.4:pl=0.4,format=gbrp[vreal];"
         + cartoon_body("cin", "vct") +
         f";[vreal][vct]blend=all_expr='A*(1-({MIX_EXPR}))+B*({MIX_EXPR})',format=yuv420p,noise=c0s=6:c0f=t+u[vv]")
STYLES = {"01_colour-cartoon_baseline": CARTOON, "02_mixed_real-and-cartoon": MIXED, "03_abstract_colour-cartoon": ABSTRACT}

def render(graph, src, dst, d=None):
    total = d if d else dur(src)
    g = graph + f";[vv]fade=t=in:st=0:d=0.9,fade=t=out:st={total-1.5:.2f}:d=1.5,format=yuv420p[vout]"
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-i",src]
    if d: cmd += ["-t", str(d)]
    cmd += ["-filter_complex",g,"-map","[vout]","-an","-c:v","libx264","-crf","16",
            "-preset","veryfast","-pix_fmt","yuv420p","-r","24",dst,"-y"]
    run(cmd); print(dst, round(dur(dst),2))

def reverse_video(src, dst, chunk=12):
    d = dur(src); n = int(d // chunk)
    tmp = os.path.join(WORK, "rev2_"+os.path.basename(dst)); os.makedirs(tmp, exist_ok=True)
    parts = []
    for i in range(n):
        p = os.path.join(tmp, f"r{i:02d}.mp4"); parts.append(p)
        if not os.path.exists(p):
            run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(i*chunk),"-i",src,"-t",str(chunk),
                 "-vf","reverse,format=yuv420p","-an","-c:v","libx264","-crf","15","-preset","veryfast",p,"-y"])
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        for p in reversed(parts): f.write(f"file '{p}'\n")
    run(["ffmpeg","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",lst,
         "-c:v","libx264","-crf","16","-preset","veryfast","-r","24",dst,"-y"])
    print(dst, round(dur(dst),2))

# ---- audio ---------------------------------------------------------------
# Two tracks per version, crossfaded mid-film, edited to exactly LEN.
AUDIO = {
 "01_colour-cartoon_baseline": ([("Dream Boat - Coyote Hearing.mp3", 0, 144),
                                   ("Blue Dream - Cheel.mp3", 16, 148)], 4),
 "02_mixed_real-and-cartoon": ([("Night Drive - Quincas Moreira.mp3", 0, 160.5),
                                ("Take the Hill - The Whole Other.mp3", 43, 132.5)], 5),
 "03_abstract_colour-cartoon": ([("Sift - Density & Time.mp3", 15, 167.3),
                                 ("Glass Pond - Public Memory.mp3", 0, 126.7)], 6),
}
def soundtrack(key):
    adir = os.path.join(WORK, "audio288"); os.makedirs(adir, exist_ok=True)
    final = os.path.join(adir, key + "_master.wav")
    if os.path.exists(final): return final
    segs, xf = AUDIO[key]
    raw = os.path.join(adir, key + "_raw.wav")
    fades = f"afade=t=in:st=0:d=1.2:curve=cbr,afade=t=out:st={LEN-6}:d=6:curve=cbr,atrim=0:{LEN},asetpts=N/SR/TB"
    if len(segs) == 1:
        fname, s, l = segs[0]
        run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(s),"-t",str(l),"-i",os.path.join(MUSIC, fname),
             "-af",fades,"-ar","48000","-ac","2",raw,"-y"])
    else:
        cmd = ["ffmpeg","-hide_banner","-loglevel","error"]
        for fname, s, l in segs: cmd += ["-ss",str(s),"-t",str(l),"-i",os.path.join(MUSIC, fname)]
        cmd += ["-filter_complex",f"[0:a][1:a]acrossfade=d={xf}:c1=tri:c2=tri,{fades}[a]",
                "-map","[a]","-ar","48000","-ac","2",raw,"-y"]
        run(cmd)
    # static-gain loudness master: preserve dynamics, hit -14 LUFS / -1.3 dBTP
    r = subprocess.run(["ffmpeg","-hide_banner","-i",raw,"-af","loudnorm=I=-14:TP=-1.3:LRA=11:print_format=json","-f","null","-"], capture_output=True, text=True)
    info = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}")+1])
    gain = -14 - float(info["input_i"])  # loudness-first; limiter is the ceiling
    print(f"audio {key}: I={info['input_i']} TP={info['input_tp']} -> gain {gain:+.2f} dB")
    run(["ffmpeg","-hide_banner","-loglevel","error","-i",raw,"-af",f"volume={gain:.2f}dB,alimiter=limit=0.84:level=false",
         "-ar","48000","-ac","2",final,"-y"])
    return final

def mux(picture, audio, dst):
    run(["ffmpeg","-hide_banner","-loglevel","error","-i",picture,"-i",audio,"-map","0:v","-map","1:a",
         "-c:v","copy","-c:a","aac","-b:a","256k","-ar","48000","-shortest","-movflags","+faststart",dst,"-y"])
    print(dst, round(dur(dst),2))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    base = build_story()
    if mode == "preview":
        prev = os.path.join(WORK, "previews2"); os.makedirs(prev, exist_ok=True)
        for name, g in STYLES.items():
            render(g, base, os.path.join(prev, name+"_preview30s.mp4"), d=30)
    else:
        pic = os.path.join(WORK, "picture2"); os.makedirs(pic, exist_ok=True); os.makedirs(OUT, exist_ok=True)
        for name, g in STYLES.items():
            audio = soundtrack(name)
            fwd_p = os.path.join(pic, name+"_fwd.mp4")
            if not os.path.exists(fwd_p): render(g, base, fwd_p)
            rev_p = os.path.join(pic, name+"_rev.mp4")
            if not os.path.exists(rev_p): reverse_video(fwd_p, rev_p)
            mux(fwd_p, audio, os.path.join(OUT, name+"_forward.mp4"))
            mux(rev_p, audio, os.path.join(OUT, name+"_backwards.mp4"))


def extract(src, start, length, out):
    """Extract a constant-24fps video slice from a base clip."""
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-i", src, "-t", str(length),
        "-vf", "setpts=N/(24*TB),fps=24,format=yuv420p",
        "-an", "-c:v", "libx264", "-crf", "13", "-preset", "veryfast", out, "-y",
    ])
