"""Pipeline figure for the BRepVLM RL section.  python figures/make_pipeline_figure.py
Writes figures/pipeline_concept.svg (editable in Illustrator / Inkscape / PowerPoint) and, if
playwright is installed, figures/pipeline_concept.png. Shape renders come from figures/assets/.

Panel (a) shows chunk-9 model 00094357 (lines from its .seq.txt, spaces compressed, and its annotation). Panel (b)'s
rollouts and advantage bars are schematic. Panels (c) and (d) are measured values from run01
(logs/run01_prep_*.log, 2026-10-26); update them when the data or the reward changes."""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
W, H = 1600, 800
SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, DejaVu Sans Mono, monospace"
out = []


def img(name):
    with open(os.path.join(A, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def rect(x, y, w, h, fill, stroke="none", sw=1, rx=10, dash=None, op=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{op}"/>')


def text(x, y, s, size=13, fill="#222", weight=400, anchor="start", family=SANS, italic=False):
    st = ' font-style="italic"' if italic else ""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{st}>{s}</text>')


def arrow(pts, color="#555", sw=2, dash=None):
    d = " ".join(("M" if i == 0 else "L") + f"{x},{y}" for i, (x, y) in enumerate(pts))
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    mid = {"#555": "ah", "#D85A30": "ahc", "#0F6E56": "aht"}.get(color, "ah")
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{ds} marker-end="url(#{mid})"/>')


def image(x, y, w, h, name):
    out.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{img(name)}"/>')


# ---------------------------------------------------------------- canvas + markers
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
out.append('<defs>' + "".join(
    f'<marker id="{i}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M1,1 L9,5 L1,9 z" fill="{c}"/></marker>' for i, c in (("ah", "#555"), ("ahc", "#D85A30"), ("aht", "#0F6E56"))) + '</defs>')
rect(0, 0, W, H, "#ffffff", rx=0)

# ---------------------------------------------------------------- (1) data
rect(20, 20, 470, 370, "#EAF3FB", "#85B7EB", 1.5, 16)
text(40, 52, "(a) Data: ABC models as B-rep code + text", 17, "#0C447C", 700)
rect(40, 72, 150, 38, "#ffffff", "#378ADD", 1.2, 8)
text(115, 96, "ABC 00094357 (STEP)", 13, "#0C447C", 600, "middle")
arrow([(115, 110), (115, 124)])
image(40, 120, 150, 150, "target.png")
rect(40, 276, 150, 50, "#ffffff", "#378ADD", 1.2, 8)
text(115, 296, "normalize to [-1, 1]", 12, "#0C447C", 500, "middle")
text(115, 314, "12-bit integer grid", 12, "#0C447C", 500, "middle")
text(115, 350, "22 faces · 64 edges", 12, "#185FA5", 400, "middle", italic=True)
text(115, 366, "planes + cylinders", 12, "#185FA5", 400, "middle", italic=True)

rect(210, 72, 265, 178, "#1E2333", rx=10)
code = [("model(3)", "#C9D1D9"),
        ("v(0,0,585,3509) v(1,585,0,3509) ...", "#79C0FF"),
        ("f8 = cylinder(8,", "#7EE787"),
        ("  axis_org=[585,585,2047],", "#7EE787"),
        ("  axis_dir=[2047,2047,4094], r=585)", "#7EE787"),
        ("wire(0,face=8,r=0,edges=[-1,2,3,-4])", "#FFA657"),
        ("e0 = circle(0, r=585, sweep=0,", "#D2A8FF"),
        ("  center=[585,585,3509], v=[0,1])", "#D2A8FF"),
        ("end_model()", "#C9D1D9")]
for i, (s, c) in enumerate(code):
    text(222, 94 + i * 18, s, 11.5, c, 400, family=MONO)
rect(210, 258, 265, 117, "#ffffff", "#85B7EB", 1.2, 10)
for i, s in enumerate(["Construction: a 2.0 x 2.0 x 1.43 box with", "filleted edges, a central recess and an",
                       "arched cutout on the front face ...", "Topology: 22 faces (12 planar, 10 cylindrical),",
                       "64 edges and 43 vertices ..."]):
    text(222, 280 + i * 18, s, 12, "#2C2C2A" if i not in (0, 3) else "#0C447C", 600 if i in (0, 3) else 400)

# ---------------------------------------------------------------- (2) GRPO loop
rect(510, 20, 1070, 370, "#EEEDFE", "#AFA9EC", 1.5, 16)
text(530, 52, "(b) GRPO with a BRep compiler in the loop", 17, "#3C3489", 700)
# prompt
rect(530, 72, 160, 126, "#ffffff", "#7F77DD", 1.2, 10)
text(610, 96, "text prompt q", 14, "#3C3489", 700, "middle")
for i, s in enumerate(["Construction: ...", "Shape: ...", "Topology: 22 faces,", "64 edges, 43 vertices"]):
    text(545, 120 + i * 18, s, 12, "#534AB7", 400, italic=True)
arrow([(690, 135), (708, 135)])
# policy
rect(710, 95, 150, 80, "#534AB7", rx=12)
text(785, 128, "BRepVLM  πθ", 16, "#ffffff", 700, "middle")
text(785, 150, "InternVL3-8B, SFT init", 12, "#E5E2FF", 400, "middle")
arrow([(860, 135), (888, 135)])
# rollouts
rect(890, 72, 160, 126, "#ffffff", "#7F77DD", 1.2, 10)
cols = ["#7F77DD", "#378ADD", "#1D9E75", "#EF9F27", "#D85A30", "#7F77DD", "#378ADD", "#1D9E75"]
for i, c in enumerate(cols):
    rect(902, 80 + i * 11.5, 6, 8, c, rx=1)
    rect(912, 81 + i * 11.5, 120 - (i * 7) % 40, 6, "#D9D6F5", rx=3)
text(970, 186, "G = 8 sampled B-reps", 12, "#3C3489", 600, "middle")
arrow([(1050, 135), (1068, 135)])
# compiler
rect(1070, 95, 150, 80, "#0F6E56", rx=12)
text(1145, 127, "BRep compiler", 15, "#ffffff", 700, "middle")
text(1145, 149, "OpenCascade, ShapeFix", 11, "#D4F2E7", 400, "middle")
arrow([(1220, 135), (1248, 135)])
# solids + rewards
rect(1250, 72, 310, 126, "#ffffff", "#7F77DD", 1.2, 10)
cards = [("roll_good.png", "matches", "#1D9E75", "valid solid"), ("roll_rough.png", "close", "#BA7517", "valid solid"),
         ("roll_wrong.png", "wrong shape", "#5F5E5A", "valid solid"), ("roll_broken.png", "no solid", "#D85A30", "invalid")]
for i, (f, r, c, v) in enumerate(cards):   # schematic: the reward ladder in (c) gives the actual values
    x = 1258 + i * 75
    image(x, 74, 72, 72, f)
    text(x + 36, 162, r, 11.5, c, 700, "middle")
    text(x + 36, 180, v, 11, c, 600, "middle")
arrow([(1405, 198), (1405, 228)])
# advantages
rect(1250, 230, 310, 145, "#ffffff", "#7F77DD", 1.2, 10)
text(1405, 252, "group-relative advantage (schematic)", 13, "#3C3489", 700, "middle")
text(1405, 270, "Aᵢ = (rᵢ − mean r) / std r", 12, "#534AB7", 400, "middle", italic=True)
base = 322
out.append(f'<line x1="1270" y1="{base}" x2="1545" y2="{base}" stroke="#AFA9EC" stroke-width="1"/>')
for i, a in enumerate([1.4, 0.6, -0.4, -1.1, 0.9, -0.2, -0.7, -0.5]):
    x = 1282 + i * 33
    h = abs(a) * 30
    rect(x, base - h if a > 0 else base, 22, h, "#1D9E75" if a > 0 else "#D85A30", rx=2)
text(1405, 366, "reinforce what compiled and fits", 11, "#534AB7", 400, "middle", italic=True)
arrow([(1250, 330), (1042, 330)])
# update
rect(710, 255, 330, 100, "#ffffff", "#534AB7", 1.6, 12)
text(875, 285, "policy update (GRPO)", 15, "#3C3489", 700, "middle")
text(875, 307, "clipped ratio 0.2 / 0.28 + KL 0.001", 12, "#534AB7", 400, "middle")
text(875, 325, "verl + vLLM, 8 rollouts per prompt", 12, "#534AB7", 400, "middle")
arrow([(785, 255), (785, 178)], "#555", 2.2)
text(795, 220, "θ ← θ + ∇J", 12, "#3C3489", 600)
# guidance
rect(1070, 230, 150, 76, "#FFF4EF", "#D85A30", 1.6, 10, dash="6,4")
text(1145, 252, "GT guidance", 13, "#993C1D", 700, "middle")
text(1145, 270, "hard prompts: GT seq", 11, "#993C1D", 400, "middle")
text(1145, 286, "joins the group", 11, "#993C1D", 400, "middle")
text(1145, 300, "(ReCAD / LUFFY)", 10.5, "#993C1D", 400, "middle", italic=True)
arrow([(1220, 268), (1248, 268)], "#D85A30", 1.8, dash="5,4")
# recipe chips
for i, s in enumerate(["curriculum order", "rewards in a sandbox", "no alignment: frame", "must be right too"]):
    if i < 2:
        rect(530, 232 + i * 40, 160, 30, "#ffffff", "#AFA9EC", 1, 15)
        text(610, 252 + i * 40, s, 12, "#3C3489", 600, "middle")
rect(530, 312, 160, 46, "#ffffff", "#AFA9EC", 1, 15)
text(610, 331, "no alignment: pose", 12, "#3C3489", 600, "middle")
text(610, 348, "and scale count too", 12, "#3C3489", 600, "middle")

# ---------------------------------------------------------------- (3) reward ladder
rect(20, 410, 760, 370, "#E1F5EE", "#5DCAA5", 1.5, 16)
text(40, 442, "(c) Verifiable reward: validity first, then geometry", 17, "#085041", 700)
rungs = [("valid closed solid", "0.25 + 0.65 min(IoU, CD) + 0.10 structure", "#0F6E56", "#ffffff", 400),
         ("edges match the target", "+ wireframe Chamfer  (≤ 0.40)", "#1D9E75", "#ffffff", 370),
         ("parses, loops close, edges used twice", "no OpenCascade needed  (≤ 0.20)", "#5DCAA5", "#04342C", 340),
         ("truncated or unparseable", "reward 0", "#D3D1C7", "#2C2C2A", 310)]
for i, (t1, t2, fill, tc, w) in enumerate(rungs):
    y = 462 + i * 76
    rect(40, y, w, 64, fill, rx=10)
    text(56, y + 27, t1, 14, tc, 700)
    text(56, y + 48, t2, 12.5, tc, 400)
text(470, 470, "Sanity check, 80 shapes (run01)", 13, "#085041", 700)
bars = [("ground truth", 1.000, "#0F6E56"), ("jitter ±2 steps", 0.974, "#1D9E75"), ("jitter ±8", 0.516, "#5DCAA5"),
        ("jitter ±128", 0.418, "#9FE1CB"), ("face deleted", 0.385, "#EF9F27"), ("other model", 0.321, "#B4B2A9"),
        ("simple box", 0.320, "#888780"), ("truncated", 0.0, "#D85A30")]
for i, (lab, v, c) in enumerate(bars):
    y = 486 + i * 33
    text(470, y + 17, lab, 12, "#04342C", 500)
    rect(570, y + 4, max(2, 180 * v), 18, c, rx=4)
    text(576 + 180 * v, y + 18, f"{v:.2f}", 11.5, "#04342C", 600)

# ---------------------------------------------------------------- (4) curriculum + data funnel
rect(800, 410, 780, 370, "#FAEEDA", "#EF9F27", 1.5, 16)
text(820, 442, "(d) Curriculum and data budget (chunk 9)", 17, "#633806", 700)
segs = [(594, "#7F77DD", "small planar"), (1719, "#378ADD", "planes + cylinders"),
        (591, "#BA7517", "cones, tori, splines"), (414, "#D85A30", "complex (30-70 faces)")]
tot = sum(s[0] for s in segs)
x = 820
for n, c, lab in segs:
    w = 740 * n / tot
    rect(x, 462, w - 3, 40, c, rx=6)
    text(x + w / 2, 487, lab if w > 110 else lab.split(" (")[0], 12, "#ffffff", 700, "middle")
    text(x + w / 2, 520, f"{n}", 12, "#633806", 600, "middle")
    x += w
text(1190, 452, "prompts by tier, 16k stage", 11.5, "#633806", 400, "middle", italic=True)
out.append('<path d="M820,532 L820,540 L1462,540 L1462,532" fill="none" stroke="#633806" stroke-width="1.4"/>')
text(1141, 558, "stage 1: answers up to 8k tokens (2,877 prompts)", 12.5, "#633806", 600, "middle")
out.append('<path d="M820,566 L820,574 L1560,574 L1560,566" fill="none" stroke="#633806" stroke-width="1.4"/>')
text(1190, 592, "stage 2: up to 16k tokens (3,318 prompts)", 12.5, "#633806", 600, "middle")
fun = [("ABC models in chunk 9", 10000), ("v3 sequences", 4477), ("unique shapes", 3956), ("with text annotations", 3856),
       ("GT rebuilds into a valid solid", 3686), ("RL prompts (8k stage)", 2877)]
for i, (lab, n) in enumerate(fun):
    y = 608 + i * 28
    w = 460 * n / 10000
    rect(1060, y, w, 24, ["#FAC775", "#EF9F27", "#D08B1E", "#BA7517", "#854F0B", "#633806"][i], rx=5)
    text(1050, y + 17, lab, 12, "#412402", 500, "end")
    text(1066 + w, y + 17, f"{n:,}", 12, "#412402", 700)

out.append("</svg>")
svg = "\n".join(out)
with open(os.path.join(HERE, "pipeline_concept.svg"), "w") as f:
    f.write(svg)
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        pg.set_content(f'<html><body style="margin:0">{svg}</body></html>')
        pg.screenshot(path=os.path.join(HERE, "pipeline_concept.png"), clip=dict(x=0, y=0, width=W, height=H))
        b.close()
    print("wrote pipeline_concept.svg and .png")
except Exception as e:  # noqa: BLE001
    print("wrote pipeline_concept.svg (png needs playwright):", e)
