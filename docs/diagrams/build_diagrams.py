#!/usr/bin/env python3
"""Model diagrams of the README and docs: shared core, Task A, Task B.

    python docs/diagrams/build_diagrams.py            # writes docs/diagrams/*.html
    python docs/diagrams/build_diagrams.py --png      # also renders docs/assets/diagram-*.png with headless Chrome
                                                      # (CHROME=path/to/chrome; on WSL the Windows Chrome is used)

Grammar: the diagram-design skill (architecture type: orthogonal r=8 connectors, arrows drawn before nodes, masked
arrow labels, bottom legend strip, at most 9 nodes per figure). Colour encodes meaning in two families:
data sources (outage records, weather forecasts, radar) and model functions (network, trees, tail model, blend).
The Σ blend is the single focal element of each figure.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / 'assets'

PAPER, INK, MUTED, SOFT = '#ffffff', '#0f172a', '#475569', '#64748b'
SANS = "'Geist', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'Geist Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace"
SERIF = "'Instrument Serif', Georgia, serif"
FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600'
         '&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">')

# kind -> (text/stroke colour, fill, border, legend label). Text colours clear 4.5:1 on white and on their fill.
KIND = {
    'outage':  ('#4338ca', '#eef2ff', '#6366f1', 'Outage records'),
    'weather': ('#0e7490', '#ecfeff', '#06b6d4', 'Weather forecasts'),
    'radar':   ('#047857', '#ecfdf5', '#10b981', 'Radar'),
    'base':    ('#475569', '#f8fafc', '#94a3b8', 'Calendar, county'),
    'nn':      ('#6d28d9', '#f5f3ff', '#8b5cf6', 'Neural network'),
    'tree':    ('#c2410c', '#fff7ed', '#f97316', 'Boosted trees'),
    'tail':    ('#be123c', '#fff1f2', '#f43f5e', 'Tail classifier'),
    'out':     (INK, '#f1f5f9', '#334155', 'Blend and output'),
    # system overview: one colour per component
    'pipe':    ('#475569', '#f8fafc', '#94a3b8', 'Pipeline'),
    'infer':   ('#6d28d9', '#f5f3ff', '#8b5cf6', 'Frozen ensemble'),
    'sql':     ('#1d4ed8', '#eff6ff', '#3b82f6', 'Warehouse'),
    'train':   ('#c2410c', '#fff7ed', '#f97316', 'Trainer'),
    'eval':    (INK, '#f1f5f9', '#334155', 'Evaluation'),
}


class Figure:
    def __init__(self, slug: str, title: str, desc: str, width: int, height: int):
        self.slug, self.title, self.desc, self.w, self.h = slug, title, desc, width, height
        self.layers = {'zones': [], 'arrows': [], 'labels': [], 'nodes': [], 'top': []}
        self.used = []

    # ------------------------------------------------------------------ primitives
    def _use(self, kind):
        if kind not in self.used:
            self.used.append(kind)

    def text(self, x, y, s, size=12, color=INK, weight=400, family=SANS, anchor='middle', spacing=None, layer='nodes',
             italic=False):
        extra = f' letter-spacing="{spacing}"' if spacing else ''
        extra += ' font-style="italic"' if italic else ''
        self.layers[layer].append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" '
                                  f'font-family="{family}" text-anchor="{anchor}"{extra}>{s}</text>')

    def zone(self, x, y, w, h, label, kind):
        color, fill, border, _ = KIND[kind]
        self.layers['zones'].append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
                                    f'fill-opacity="0.55" stroke="{border}" stroke-opacity="0.45" stroke-width="0.8"/>')
        self.text(x + 12, y + 18, label, 9, color, 600, MONO, 'start', '0.14em', 'zones')

    def path(self, pts, kind, dash=False, width=1.4, hop=None):
        """Orthogonal polyline with r=8 quarter-arc bends. hop=(index, axis_value): bridge on segment `index`."""
        color = KIND[kind][0]
        self._use(kind)
        d = f'M {pts[0][0]},{pts[0][1]}'
        for i in range(1, len(pts)):
            (x0, y0), (x1, y1) = pts[i - 1], pts[i]
            if hop and hop[0] == i:            # bridge over a crossing line on this segment
                if x0 == x1:                   # vertical segment: hop at y = hop[1]
                    s = 1 if y1 > y0 else -1
                    d += f' L {x1},{hop[1] - 8 * s} a 8,8 0 0,{1 if s < 0 else 0} 0,{16 * s}'
                else:
                    s = 1 if x1 > x0 else -1
                    d += f' L {hop[1] - 8 * s},{y1} a 8,8 0 0,{1 if s > 0 else 0} {16 * s},0'
            if i < len(pts) - 1:
                x2, y2 = pts[i + 1]
                l1, l2 = abs(x1 - x0) + abs(y1 - y0), abs(x2 - x1) + abs(y2 - y1)
                r = min(8, l1 / 2, l2 / 2)
                ux, uy = (x1 - x0) / l1, (y1 - y0) / l1
                vx, vy = (x2 - x1) / l2, (y2 - y1) / l2
                d += f' L {x1 - ux * r:g},{y1 - uy * r:g} Q {x1},{y1} {x1 + vx * r:g},{y1 + vy * r:g}'
            else:
                d += f' L {x1},{y1}'
        self.layers['arrows'].append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
            + (' stroke-dasharray="5,4"' if dash else '') + f' marker-end="url(#{self.slug}-arrow-{kind})"/>')

    def label(self, cx, cy, s, kind='base', size=9):
        """Arrow label centred at (cx, cy) on an opaque mask; the caller keeps it 6-10 px off the stroke."""
        w = int(len(s) * size * 0.62 + 12) // 4 * 4 + 4
        h = 16
        self.layers['labels'].append(f'<rect x="{cx - w / 2:g}" y="{cy - h / 2:g}" width="{w}" height="{h}" rx="2" fill="{PAPER}"/>')
        self.text(cx, cy + size * 0.36, s, size, KIND[kind][0], 500, MONO, 'middle', '0.06em', 'labels')

    def node(self, x, y, w, h, kind, name, subs=(), tag=None, dash=False, name_size=13):
        color, fill, border, _ = KIND[kind]
        self._use(kind)
        n = self.layers['nodes']
        n.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>')
        n.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{border}" stroke-width="1.2"'
                 + (' stroke-dasharray="4,3"' if dash else '') + '/>')
        top = y
        if tag:
            tw = int(len(tag) * 7 * 0.62 + 10) // 2 * 2 + 2
            n.append(f'<rect x="{x + 10}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="{PAPER}" stroke="{border}" stroke-width="0.8"/>')
            self.text(x + 10 + tw / 2, y + 17, tag, 7, color, 600, MONO, 'middle', '0.08em')
            top = y + 20
        block = 10 + 15 * len(subs)                      # cap height of the name + one 15 px row per sublabel
        ty = top + (y + h - top - block) / 2 + 10
        self.text(x + w / 2, ty, name, name_size, color, 600)
        for i, s in enumerate(subs):
            self.text(x + w / 2, ty + 15 + 13 * i, s, 9.5, MUTED, 400, MONO)

    def sigma(self, cx, cy, r=26):
        n = self.layers['nodes']
        n.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{INK}"/>')
        self.text(cx, cy + 8, 'Σ', 22, '#ffffff', 600, SANS)
        self._use('out')

    def note(self, x, y, s, size=11, color=MUTED, anchor='start', italic=False, family=SANS):
        self.text(x, y, s, size, color, 400, family, anchor, layer='top', italic=italic)

    def legend(self, y):
        top = self.layers['top']
        top.append(f'<line x1="24" y1="{y}" x2="{self.w - 24}" y2="{y}" stroke="rgba(15,23,42,0.12)" stroke-width="0.8"/>')
        self.text(24, y + 26, 'LEGEND', 8, SOFT, 600, MONO, 'start', '0.16em', 'top')
        x = 96
        for kind in KIND:
            if kind not in self.used:
                continue
            color, fill, border, label = KIND[kind]
            top.append(f'<rect x="{x}" y="{y + 16}" width="14" height="14" rx="3" fill="{fill}" stroke="{border}" stroke-width="1.2"/>')
            self.text(x + 20, y + 27, label, 10.5, color, 500, SANS, 'start', layer='top')
            x += 20 + int(len(label) * 10.5 * 0.55) + 24

    # ------------------------------------------------------------------ output
    def svg(self, extra_attrs='') -> str:
        markers = ''.join(
            f'<marker id="{self.slug}-arrow-{k}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{KIND[k][0]}"/></marker>' for k in KIND)
        body = '\n'.join(self.layers['zones'] + self.layers['arrows'] + self.layers['labels'] + self.layers['nodes']
                         + self.layers['top'])
        return (f'<svg viewBox="0 0 {self.w} {self.h}" xmlns="http://www.w3.org/2000/svg" role="img" '
                f'aria-labelledby="{self.slug}-title {self.slug}-desc"{extra_attrs}>\n'
                f'<title id="{self.slug}-title">{self.title}</title>\n<desc id="{self.slug}-desc">{self.desc}</desc>\n'
                f'<defs>{markers}</defs>\n<rect width="100%" height="100%" fill="{PAPER}"/>\n{body}\n</svg>')

    def html(self, eyebrow: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self.title}</title>
{FONTS}
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: {SANS}; background: {PAPER}; color: {INK}; padding: 3rem 2rem; display: flex; justify-content: center; }}
.frame {{ max-width: {self.w + 40}px; width: 100%; }}
.eyebrow {{ font-family: {MONO}; font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: {MUTED}; margin-bottom: 0.5rem; }}
h1 {{ font-family: {SERIF}; font-size: clamp(1.5rem, 2.4vw + 0.75rem, 2rem); font-weight: 400; letter-spacing: -0.02em; margin-bottom: 1.5rem; }}
.diagram-container {{ width: 100%; overflow-x: auto; }}
svg {{ width: 100%; min-width: {self.w}px; display: block; }}
@media print {{ .diagram-container {{ overflow-x: visible; }} svg {{ min-width: 0; }} }}
</style>
</head>
<body>
<div class="frame">
<p class="eyebrow">{eyebrow}</p>
<h1>{self.title}</h1>
<div class="diagram-container">
{self.svg()}
</div>
</div>
</body>
</html>
"""

    def bare_html(self) -> str:
        """Only the SVG at its native size, for rasterising."""
        return (f'<!DOCTYPE html><html><head><meta charset="UTF-8">{FONTS}<style>html,body{{margin:0;background:{PAPER}}}'
                f'svg{{display:block;width:{self.w}px;height:{self.h}px}}</style></head><body>{self.svg()}</body></html>')


# ====================================================================== figures
def shared_core() -> Figure:
    f = Figure('diagram-shared-core', 'Shared core: history network and joint feature tree',
               'Outage history, weather forecasts and radar, all as of the issue time, feed a GRU history network and a '
               'joint XGBoost feature tree; a fixed-weight blend of the two gives the outage share and the risk of each '
               'data-center architecture.', 960, 436)
    f.zone(232, 44, 456, 144, 'GRU HISTORY NETWORK', 'nn')
    f.zone(232, 212, 456, 140, 'JOINT FEATURE TREE', 'tree')
    # arrows first
    f.path([(208, 112), (252, 112)], 'outage')                                            # history -> encoder
    f.path([(208, 128), (224, 128), (224, 272), (252, 272)], 'outage')                    # history -> feature table
    f.path([(208, 200), (552, 200), (552, 156)], 'weather', width=1.2, hop=(1, 224))      # weather -> decoder (per lead)
    f.path([(208, 216), (216, 216), (216, 292), (252, 292)], 'weather')                   # weather -> feature table
    f.path([(208, 312), (252, 312)], 'radar')                                             # radar -> feature table
    f.path([(436, 120), (476, 120)], 'nn')                                                # encoder -> decoder
    f.path([(436, 292), (476, 292)], 'tree')                                              # table -> booster
    f.path([(668, 120), (728, 120), (728, 176)], 'nn')                                    # decoder -> blend
    f.path([(668, 292), (728, 292), (728, 236)], 'tree')                                  # booster -> blend
    f.path([(754, 206), (788, 206)], 'out', width=1.8)                                    # blend -> risk
    # labels
    f.label(368, 200 - 14, 'PER-LEAD WEATHER', 'weather')
    f.label(780, 150, 'A 0.4 · B 0.5', 'nn')
    f.label(780, 262, 'A 0.4 · B 0.5', 'tree')
    # nodes
    f.node(24, 88, 184, 56, 'outage', 'Outage history', ['7 days · 15 min · EAGLE-I'], name_size=13)
    f.node(24, 180, 184, 56, 'weather', 'Weather forecasts', ['HRRR · 14 vars · 48 h'])
    f.node(24, 288, 184, 48, 'radar', 'Radar exposure', ['MRMS · last 24 h'])
    f.node(252, 84, 184, 72, 'nn', 'GRU history encoder', ['7-day sequence', '+ 11 event-state values'], tag='GRU')
    f.node(476, 84, 192, 72, 'nn', 'Lead-wise decoder', ['context added at every lead', 'temporal conv · 2 heads'], tag='CONV')
    f.node(252, 252, 184, 80, 'tree', 'Point-in-time features', ['lags · rolling · events', 'weather · radar · county'],
           tag='150-226 COLS')
    f.node(476, 252, 192, 80, 'tree', 'XGBoost, Tweedie', ['one booster per task', 'all counties and leads'], tag='TREE')
    f.sigma(728, 206)
    f.node(788, 168, 148, 76, 'out', 'Architecture risk', ['x̂ → σ(4000 (x̂ − x₀))', 'ARC01 … ARC04'])
    f.note(24, 380, 'Every input is filtered to what was published by the issue time T. '
                    'Weights are fixed (Task A adds a Q95 tail member).', 11)
    f.legend(392)
    return f


def task_a() -> Figure:
    f = Figure('diagram-task-a', 'Task A: three members for 48-hour forecasts',
               'Task A blends a GRU history network, a joint feature tree and a Q95 tail member with fixed weights '
               '0.4, 0.4 and 0.2 into 48 hourly outage forecasts per county, each mapped to four architecture risks.',
               960, 492)
    # arrows
    f.path([(232, 80), (328, 80)], 'outage')                                              # sequence -> GRU
    f.path([(232, 164), (252, 164), (252, 176), (328, 176)], 'outage')                    # event features -> tree
    f.path([(232, 228), (264, 228), (264, 206), (328, 206)], 'radar')                     # radar -> tree
    f.path([(232, 292), (276, 292), (276, 236), (328, 236)], 'weather')                   # averages -> tree
    f.path([(232, 308), (328, 308)], 'weather')                                           # averages -> Q95
    f.path([(600, 80), (688, 80), (688, 174)], 'nn')                                      # GRU -> blend
    f.path([(600, 206), (662, 206)], 'tree')                                              # tree -> blend
    f.path([(600, 332), (688, 332), (688, 232)], 'tail')                                  # Q95 -> blend
    f.path([(714, 206), (760, 206)], 'out', width=1.8)                                    # blend -> output
    # labels
    f.label(716, 120, '× 0.4', 'nn')
    f.label(630, 192, '× 0.4', 'tree')
    f.label(716, 292, '× 0.2', 'tail')
    # nodes
    f.node(24, 56, 208, 48, 'outage', '7-day outage sequence', ['+ event state, as of T'])
    f.node(24, 140, 208, 48, 'outage', 'Tree event features', ['9 · peak, duration, recovery'])
    f.node(24, 204, 208, 48, 'radar', 'Radar exposure', ['20 · last 24 h before T'])
    f.node(24, 276, 208, 48, 'weather', 'Two-run weather average', ['12 · 00Z and previous 18Z run'])
    f.node(328, 40, 272, 80, 'nn', 'GRU history network', ['185 inputs + 7-day sequence', '48 hourly leads at once'], tag='NETWORK')
    f.node(328, 148, 272, 116, 'tree', 'Joint feature tree', ['XGBoost · Tweedie · 226 inputs', 'base + events + averages', '+ radar exposure'],
           tag='XGBOOST')
    f.node(328, 292, 272, 80, 'tail', 'Q95 tail member', ['P(x ≥ q95) mixes two levels', '185 inputs'], tag='CLASSIFIER')
    f.sigma(688, 206)
    f.node(760, 164, 176, 84, 'out', '48 hourly forecasts', ['x̂ per county, 00 UTC', '→ 4 architecture risks'])
    f.note(24, 400, 'All members also read the 150 base columns (calendar, county, 30 outage-history features, HRRR statistics).', 11)
    f.note(24, 420, 'Fallbacks, per county and issue: no radar → radar-free event tree · no weather → weather-free tree · '
                    'no history → history-free tree.', 11)
    f.legend(440)
    return f


def task_b() -> Figure:
    f = Figure('diagram-task-b', 'Task B: two members on a 15-minute grid',
               'Task B blends a joint Tweedie tree and a GRU history network with equal weights on 24 native 15-minute '
               'leads, then holds each value over its interval to give 72 five-minute forecasts per issue.',
               960, 436)
    # arrows
    f.path([(224, 56), (256, 56), (256, 72), (296, 72)], 'weather')                      # forecast evolution -> tree
    f.path([(224, 120), (264, 120), (264, 104), (296, 104)], 'outage')                   # outage change -> tree
    f.path([(224, 184), (272, 184), (272, 136), (296, 136)], 'radar')                    # radar -> tree
    f.path([(224, 256), (296, 256)], 'outage')                                            # sequence -> GRU
    f.path([(560, 104), (616, 104), (616, 148)], 'tree')                                  # tree -> blend
    f.path([(560, 256), (616, 256), (616, 200)], 'nn')                                    # GRU -> blend
    f.path([(642, 174), (688, 174)], 'out', width=1.8)                                    # blend -> native grid
    f.path([(812, 216), (812, 260)], 'out', width=1.8)                                    # native -> 5-minute
    # labels
    f.label(644, 128, '× 0.5', 'tree')
    f.label(644, 224, '× 0.5', 'nn')
    f.label(868, 238, 'HOLD 15 MIN', 'out')
    # nodes
    f.node(24, 32, 200, 48, 'weather', 'Forecast evolution', ['rolling maxima, run changes'])
    f.node(24, 96, 200, 48, 'outage', 'Outage changes', ['now vs recent levels'])
    f.node(24, 160, 200, 48, 'radar', 'Radar exposure', ['storm cells, last 24 h'])
    f.node(24, 232, 200, 48, 'outage', '7-day outage sequence', ['+ event state, as of T'])
    f.node(296, 48, 264, 112, 'tree', 'Joint Tweedie tree', ['XGBoost · 192 inputs', 'one model: counties × leads', 'base + process + radar'],
           tag='XGBOOST')
    f.node(296, 216, 264, 80, 'nn', 'GRU history network', ['150 base inputs + sequence', '24 native leads'], tag='NETWORK')
    f.sigma(616, 174)
    f.node(688, 132, 248, 84, 'out', '24 native forecasts', ['+15 … +360 min, 15-min grid', 'issued 00/06/12/18 UTC'])
    f.node(688, 260, 248, 84, 'out', '72 five-minute outputs', ['each value held for 15 min', '+5, +10: latest observation'])
    f.note(24, 344, 'All members also read the 150 base columns. Fallbacks: no radar → five-family radar-free tree system;',
           11)
    f.note(24, 362, 'no history → county × month climatology.', 11)
    f.legend(384)
    return f


def system_overview() -> Figure:
    f = Figure('diagram-system-overview', 'System overview',
               'Public outage, weather and radar archives are turned into county input tables with availability times; '
               'the frozen ensemble and the DuckDB warehouse read them, the warehouse also feeds the reference trainer, '
               'and both prediction paths are scored against the same truth and a persistence baseline.', 960, 452)
    # arrows
    f.path([(168, 64), (184, 64), (184, 112), (200, 112)], 'outage')
    f.path([(168, 128), (200, 128)], 'weather')
    f.path([(168, 192), (184, 192), (184, 144), (200, 144)], 'radar')
    f.path([(336, 128), (368, 128)], 'pipe', width=1.6)
    f.path([(504, 116), (520, 116), (520, 64), (552, 64)], 'pipe', width=1.6)
    f.path([(504, 140), (532, 140), (532, 200), (552, 200)], 'pipe', width=1.6)
    f.path([(636, 240), (636, 300)], 'sql')
    f.path([(720, 64), (852, 64), (852, 160)], 'infer')
    f.path([(720, 200), (768, 200)], 'sql')
    f.path([(720, 336), (852, 336), (852, 240)], 'train')
    # labels
    f.label(702, 270, 'GOLD.EXAMPLES', 'sql')
    f.label(786, 52, 'PREDICTIONS', 'infer')
    f.label(786, 324, 'PREDICTIONS', 'train')
    f.label(744, 186, 'TRUTH', 'sql')
    # nodes
    f.node(24, 40, 144, 48, 'outage', 'EAGLE-I outages', ['ORNL · 15 min'])
    f.node(24, 104, 144, 48, 'weather', 'HRRR forecasts', ['NOAA · 00Z, 18Z runs'])
    f.node(24, 168, 144, 48, 'radar', 'MRMS radar', ['NOAA · 10 min'])
    f.node(200, 92, 136, 72, 'pipe', 'Data pipeline', ['download, decode', 'county polygons'], name_size=12.5)
    f.node(368, 92, 136, 72, 'pipe', 'Input tables', ['Parquet, 5 tables', 'with availability'], name_size=12.5)
    f.node(552, 24, 168, 80, 'infer', 'Inference', ['frozen GRU + XGBoost', 'infer.py'], tag='ENSEMBLE')
    f.node(552, 160, 168, 80, 'sql', 'Warehouse', ['bronze → silver → gold', 'sql/'], tag='DUCKDB')
    f.node(552, 300, 168, 72, 'train', 'Reference trainer', ['XGBoost, 2022–2024', 'reference.py'], tag='TRAIN')
    f.node(768, 160, 168, 80, 'eval', 'Evaluation', ['challenge score', 'vs persistence'], tag='SQL SCORING')
    f.note(24, 400, 'Every row keeps its availability time; features at issue time T only read rows available by T.', 11)
    f.legend(408)
    return f


FIGURES = {'system-overview': (system_overview, 'Architecture · whole repository'), 'shared-core': (shared_core, 'Model · both tasks'), 'task-a': (task_a, 'Model · Task A, 48 h hourly'),
           'task-b': (task_b, 'Model · Task B, 6 h every 5 min')}


def chrome_binary() -> str | None:
    for c in (os.environ.get('CHROME'), shutil.which('google-chrome'), shutil.which('chromium'),
              '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe'):
        if c and (Path(c).is_file() or shutil.which(c)):
            return c
    return None


def render_png(fig: Figure, out: Path, scale: int = 2) -> None:
    chrome = chrome_binary()
    if chrome is None:
        sys.exit('no Chrome found: set CHROME=/path/to/chrome')
    windows = chrome.endswith('.exe')
    tmp = Path('/mnt/c/Users/Public/diagram_render') if windows else Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    page, png = tmp / f'{fig.slug}.html', tmp / f'{fig.slug}.png'
    page.write_text(fig.bare_html(), encoding='utf-8')
    to_arg = (lambda p: subprocess.run(['wslpath', '-w', str(p)], capture_output=True, text=True).stdout.strip()) if windows else str
    url = ('file:///' + to_arg(page).replace('\\', '/')) if windows else page.as_uri()
    subprocess.run([chrome, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--force-device-scale-factor={scale}',
                    f'--window-size={fig.w},{fig.h}', '--virtual-time-budget=8000', f'--screenshot={to_arg(png)}', url],
                   check=True, capture_output=True)
    shutil.copyfile(png, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--png', action='store_true', help='also render PNGs into docs/assets/')
    a = p.parse_args()
    for name, (build, eyebrow) in FIGURES.items():
        fig = build()
        (HERE / f'{name}.html').write_text(fig.html(eyebrow), encoding='utf-8')
        print(HERE / f'{name}.html')
        if a.png:
            ASSETS.mkdir(parents=True, exist_ok=True)
            render_png(fig, ASSETS / f'diagram-{name}.png')
            print(ASSETS / f'diagram-{name}.png')


if __name__ == '__main__':
    main()
