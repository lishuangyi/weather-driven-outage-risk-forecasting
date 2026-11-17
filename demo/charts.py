"""Forecast-against-observation strip charts of the demo summary page (demo tool; never used by infer.py).

For each task one block of strips, one strip per county, all on the same time axis: the observed share of customers
without power (orange, with a light fill) and the forecast share (blue line, predicted_x), with the band of the four
architecture thresholds x0 (risk = 0.5).

Vertical scale: asinh(share / 0.05 %), i.e. linear from zero up to about 0.05 % and logarithmic above. Zero is on the
baseline (a plain logarithmic axis would throw every near-zero value to the bottom as a spike), the threshold band sits
in the readable middle, and large events are still in range. The top of the scale follows the data of the task.

A target time that several issues forecast shows the forecast with the shortest lead. Long windows are reduced to at
most MAX_POINTS buckets per strip, each drawn at its maximum, so peaks are kept. The lines break where no target time
was requested (forecast) or no observation exists (observed); an isolated observation is drawn as a dot. predicted_x
does not depend on the architecture; the first architecture's rows are used.

series() returns plain numbers (JSON-ready); section() returns the HTML: inline SVG that needs no script, plus a small
script for the pointer readout. Every coordinate is checked to be finite and inside the plot before the HTML is
returned (ChartError otherwise), so a broken chart is never shown. Nothing here changes a prediction.
"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime, timedelta, timezone

import polars as pl

MAX_POINTS = 900
KNEE = 5e-4                      # 0.05 % of customers: below it the axis is linear, above it logarithmic
W, LEFT, RIGHT, TOP, PLOT_H = 1000, 52, 10, 8, 116
AXIS_H = 22                      # extra height of the last strip of a task (time labels)
TOPS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
Y_TICKS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
TASK_TITLES = {'A': ('Task A', 'hourly values, forecast up to 48 hours ahead, issued daily at 00 UTC'),
               'B': ('Task B', '5-minute values, forecast up to 6 hours ahead, issued at 00, 06, 12 and 18 UTC')}
UTC = timezone.utc


class ChartError(ValueError):
    """The chart could not be drawn correctly; the summary page is written without it."""


def series(joined: dict) -> dict:
    """{task: {'bucket_minutes': m, 'counties': {fips: {'t': [...], 'pred': [...], 'obs': [...]}}}}; values are outage
    proportions (None = no observation), t is the target time in epoch seconds (start of the bucket)."""
    out = {}
    for task in ('A', 'B'):
        f = joined.get(task)
        if f is None or not f.height:
            continue
        f = f.filter(pl.col('arch') == f['arch'].min())
        if 'truth' not in f.columns:
            f = f.with_columns(pl.lit(None, dtype=pl.Float64).alias('truth'))
        f = (f.with_columns((pl.col('target') - pl.col('issue')).alias('lead'),
                            pl.when(pl.col('truth').is_finite()).then(pl.col('truth')).otherwise(None).alias('truth'))
              .sort('county', 'target', 'lead').group_by('county', 'target', maintain_order=True).first())
        step = f.group_by('county').agg(pl.col('target').diff().drop_nulls().median().alias('s'))['s'].drop_nulls()
        native = float(step.min()) if len(step) else 0.0
        longest = int(f.group_by('county').len()['len'].max())
        per = max(1, math.ceil(longest / MAX_POINTS))
        counties = {}
        for (county,), g in f.group_by('county', maintain_order=True):
            g = (g.with_row_index('i').with_columns((pl.col('i') // per).alias('bucket')).group_by('bucket', maintain_order=True)
                  .agg(pl.col('target').first().alias('t'), pl.col('px').max().alias('pred'), pl.col('truth').max().alias('obs')))
            counties[county] = {'t': [int(v) for v in g['t'].to_list()], 'pred': g['pred'].to_list(), 'obs': g['obs'].to_list()}
        out[task] = {'bucket_minutes': round(per * native / 60, 1), 'buckets_merged': per, 'step_seconds': per * native,
                     'counties': counties}
    return out


def _scale(v: float) -> float:
    return math.asinh(max(v, 0.0) / KNEE)


def _pct(v: float) -> str:
    return '0' if v == 0 else f'{100 * v:g}%'


def _time_ticks(t0: int, t1: int) -> tuple[list[int], str]:
    """Tick times on round UTC boundaries (at most about ten) and their label format."""
    span = max(t1 - t0, 1)
    for step, fmt in ((3 * 3600, '%d %b %H:%M'), (6 * 3600, '%d %b %H:%M'), (12 * 3600, '%d %b %H:%M'), (86400, '%d %b'),
                      (2 * 86400, '%d %b'), (7 * 86400, '%d %b'), (14 * 86400, '%d %b')):
        if span / step <= 10:
            first = (t0 + step - 1) // step * step
            return list(range(first, t1 + 1, step)), fmt
    ticks, d = [], datetime.fromtimestamp(t0, UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    every = 1 if span <= 400 * 86400 else 3
    while d.timestamp() <= t1:
        if d.timestamp() >= t0 and (d.month - 1) % every == 0:
            ticks.append(int(d.timestamp()))
        d = (d + timedelta(days=32)).replace(day=1)
    return ticks, '%b %Y' if span > 300 * 86400 else '%d %b'


def _segments(times, values, step):
    """Runs of consecutive available points: a run ends at a missing value or where no target time was requested."""
    runs, run, last = [], [], None
    for t, v in zip(times, values):
        if v is None or (last is not None and step > 0 and t - last > 1.5 * step):
            if run:
                runs.append(run)
            run = []
        if v is not None:
            run.append((t, v))
            last = t
        else:
            last = None
    if run:
        runs.append(run)
    return runs


def _strip(key: str, d: dict, t0: int, t1: int, y_top: float, band: tuple[float, float], step: float, ticks, fmt, last: bool) -> str:
    height = TOP + PLOT_H + (AXIS_H if last else 6)
    span, top = max(t1 - t0, 1), _scale(y_top)
    base = TOP + PLOT_H

    def px(t):
        return LEFT + (W - LEFT - RIGHT) * (t - t0) / span

    def py(v):
        return TOP + PLOT_H * (1 - min(_scale(v), top) / top)

    def check(x, y):
        if not (math.isfinite(x) and math.isfinite(y) and LEFT - 0.5 <= x <= W - RIGHT + 0.5 and TOP - 0.5 <= y <= base + 0.5):
            raise ChartError(f'coordinate outside the plot in strip {key}: ({x}, {y})')
        return f'{x:.1f} {y:.1f}'

    grid, previous = '', None
    for v in Y_TICKS:
        if v > y_top * 1.0001:
            break
        y = py(v)
        if previous is not None and previous - y < 13:      # labels would touch
            continue
        previous = y
        grid += (f"<line class='grid{' base' if v == 0 else ''}' x1='{LEFT}' x2='{W - RIGHT}' y1='{y:.1f}' y2='{y:.1f}'/>"
                 f"<text class='tick' x='{LEFT - 7}' y='{y + 3.5:.1f}' text-anchor='end'>{_pct(v)}</text>")
    for t in ticks:
        x = px(t)
        grid += f"<line class='grid time' x1='{x:.1f}' x2='{x:.1f}' y1='{TOP}' y2='{base}'/>"
        if last:
            anchor = 'start' if x < LEFT + 30 else 'end' if x > W - RIGHT - 30 else 'middle'
            grid += f"<text class='tick' x='{x:.1f}' y='{base + 15}' text-anchor='{anchor}'>{datetime.fromtimestamp(t, UTC).strftime(fmt).lstrip('0')}</text>"
    band_top, band_bottom = py(band[1]), py(band[0])
    marks = f"<rect class='band' x='{LEFT}' width='{W - LEFT - RIGHT}' y='{band_top:.1f}' height='{max(band_bottom - band_top, 1.5):.1f}'/>"
    area, line, dots = '', '', ''
    for run in _segments(d['t'], d['obs'], step):
        points = [check(px(t), py(v)) for t, v in run]
        if len(run) == 1:
            x, y = points[0].split()
            dots += f"<circle class='obs-dot' cx='{x}' cy='{y}' r='2'/>"
            continue
        line += 'M' + ' L'.join(points) + ' '
        area += f"M{px(run[0][0]):.1f} {base} L" + ' L'.join(points) + f" L{px(run[-1][0]):.1f} {base} Z "
    forecast = ''
    for run in _segments(d['t'], d['pred'], step):
        points = [check(px(t), py(v)) for t, v in run]
        forecast += ('M' + ' L'.join(points) + ' ') if len(run) > 1 else f"M{points[0]} h0.01 "
    if not forecast:
        raise ChartError(f'no forecast to draw in strip {key}')
    if area:
        marks += f"<path class='obs-area' d='{area.strip()}'/>"
    if line:
        marks += f"<path class='obs' d='{line.strip()}'/>"
    marks += dots + f"<path class='pred' d='{forecast.strip()}'/>"
    return (f"<svg viewBox='0 0 {W} {height}' data-key='{html.escape(key)}' role='img' "
            f"aria-label='Forecast and observed share of customers without power over time'>{grid}{marks}"
            f"<line class='hair' y1='{TOP}' y2='{base}' visibility='hidden'/>"
            f"<rect class='hit' x='{LEFT}' y='{TOP}' width='{W - LEFT - RIGHT}' height='{PLOT_H}'/></svg>")


def section(data: dict, county_names: dict, x0: dict) -> str:
    """HTML of the chart section ('' when there is nothing to draw). Raises ChartError when a chart would be wrong."""
    if not data:
        return ''
    band = (min(x0.values()), max(x0.values()))
    lo, hi = f'{100 * band[0]:.2f}', f'{100 * band[1]:.2f}'
    blocks, payload = '', {}
    for task, entry in data.items():
        counties = entry['counties']
        values = [v for c in counties.values() for v in c['pred'] + c['obs'] if v is not None]
        if not values or not all(math.isfinite(v) and v >= 0 for v in values):
            raise ChartError(f'Task {task}: no finite value to draw')
        peak = max(max(values), band[1] * 1.5)
        y_top = next((t for t in TOPS if t >= peak), 1.0)
        t0 = min(c['t'][0] for c in counties.values())
        t1 = max(c['t'][-1] for c in counties.values())
        ticks, fmt = _time_ticks(t0, t1)
        has_obs = any(v is not None for c in counties.values() for v in c['obs'])
        strips, names = '', list(counties)
        for i, county in enumerate(names):
            d = counties[county]
            key = f'{task}-{county}'
            payload[key] = d
            obs = [v for v in d['obs'] if v is not None]
            over = sum(v > band[0] for v in obs)
            facts = (f"observed above {lo}% in {100 * over / len(obs):.0f}% of the times; peak {100 * max(obs):.2f}%"
                     if obs else 'no observation in the input')
            strips += (f"<div class='strip'><div class='who'><b>{html.escape(county_names.get(county, county))}</b>"
                       f"<span>{html.escape(county)}</span><span>{facts}</span></div>"
                       + _strip(key, d, t0, t1, y_top, band, entry.get('step_seconds', 0), ticks, fmt, i == len(names) - 1) + '</div>')
        title, sub = TASK_TITLES.get(task, (f'Task {task}', ''))
        note = ('' if entry['buckets_merged'] == 1 else
                f"The period is long, so each point is the highest value within {entry['bucket_minutes']:g} minutes, for both "
                f"lines. Choose a shorter period (--start, --end) to see every value. ")
        note += '' if has_obs else 'The input directory has no observation for these times, so only the forecast is drawn.'
        blocks += (f"<section class='task'><h3>{html.escape(title)}<small>{html.escape(sub)}</small></h3>"
                   f"<div class='strips'>{strips}</div>" + (f"<p class='note'>{note.strip()}</p>" if note.strip() else '') + '</section>')
    legend = ("<ul class='legend'><li><span class='key pred'></span>Forecast: the share of customers the model predicted to be without power</li>"
              "<li><span class='key obs'></span>Observed: the share that actually was without power</li>"
              f"<li><span class='key band'></span>Risk thresholds of the four architectures, {lo}% to {hi}%</li></ul>")
    explain = f"""<details class='explain' open><summary>How to read the charts / 图表说明</summary><div class='cols'>
<div>
<p>Each strip is one county over the selected period; the strips of a task share the time axis (UTC). The vertical
axis is the share of the county's electricity customers without power: customers out divided by all customers of the
county.</p>
<p><b>Orange, observed.</b> What actually happened, from the EAGLE-I outage records in the input directory. Task A uses
the mean of the four 15-minute records of the hour ending at each time, Task B the record at each time. Where the
orange line breaks there is no record; a single record between two gaps is a dot.</p>
<p><b>Blue, forecast.</b> What the model predicted for the same time and the same quantity before it happened (column
<code>predicted_x</code> of predictions.csv). Where a time was forecast more than once, the most recent forecast is
drawn.</p>
<p><b>Grey band, risk thresholds.</b> A data-centre power architecture (ARC01 to ARC04) counts as at risk when the
share exceeds its threshold, {lo}% to {hi}% of customers. Above the band all four are at risk, below it none. The score
counts an event whenever the observed value is above the threshold.</p>
<p><b>What to look for.</b> Blue close to orange is a good forecast. Orange above the band with blue below it is a
missed event; blue above the band with orange below it is a false alarm. The axis is linear near zero and logarithmic
above about 0.05%, so quiet periods lie on the baseline and large events still fit. Move the pointer over a strip to
read both values.</p>
</div>
<div lang='zh'>
<p>每一条是一个县在所选时段内的情况，同一任务的各条共用时间轴（UTC）。纵轴是该县停电用户占全部用户的比例。</p>
<p><b>橙色：实际观测。</b>真实发生的停电比例，来自输入目录里的 EAGLE-I 停电记录。Task A 取每个时刻前一小时内四条
15 分钟记录的平均，Task B 取该时刻的记录。橙线中断表示没有记录；两段空缺之间的单条记录画成一个点。</p>
<p><b>蓝色：模型预测。</b>模型在事前对同一时刻、同一个量给出的预测（predictions.csv 的 <code>predicted_x</code> 列）。
同一时刻被多次预测时，画最近一次。</p>
<p><b>灰带：风险阈值。</b>四种数据中心供电架构（ARC01 至 ARC04）在停电比例超过各自阈值（{lo}% 至 {hi}%）时视为有风险。
高于灰带四种架构都有风险，低于灰带都没有。评分时，实际值超过阈值算一次事件。</p>
<p><b>怎么看。</b>蓝线贴近橙线说明预测准。橙色在灰带之上而蓝线在下方是漏报；蓝线在灰带之上而橙色在下方是误报。
纵轴在零附近是线性的，约 0.05% 以上是对数的，所以平静时段贴着底线，大事件也能画得下。鼠标移到图上可读出两个数值。</p>
</div></div></details>"""
    script = _SCRIPT % {'left': LEFT, 'right': RIGHT, 'width': W}
    return (f"<section class='charts'><h2>Forecast and observation</h2>{legend}{blocks}{explain}"
            f"<div id='tip' hidden></div><script type='application/json' id='chart-data'>{json.dumps(payload, separators=(',', ':'))}</script>"
            f"<script>{script}</script></section>")


STYLE = """
.legend{list-style:none;display:flex;flex-wrap:wrap;gap:6px 28px;margin:0 0 6px;padding:0;font-size:14px}
.key{display:inline-block;width:22px;height:0;border-top:3px solid;vertical-align:middle;margin-right:8px;border-radius:2px}
.key.pred{border-color:var(--series-1)}.key.obs{border-color:var(--series-2)}
.key.band{height:10px;border:0;background:var(--band);border-radius:0}
.task{margin:22px 0 0}.task h3{margin:0 0 8px}
.strips{background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:6px 10px 4px 0;overflow-x:auto}
.strip{display:grid;grid-template-columns:168px minmax(560px,1fr);align-items:start}
.strip+.strip{border-top:1px solid var(--line)}
.who{padding:10px 8px 0 14px;font-size:12px;line-height:1.35;color:var(--muted)}
.who b{display:block;font:600 15px/1.2 var(--display);color:var(--ink);letter-spacing:.01em}
.who span{display:block}.who span:first-of-type{font-variant-numeric:tabular-nums;margin-bottom:3px}
.strip svg{display:block;width:100%;height:auto}
.grid{stroke:var(--line);stroke-width:1;vector-effect:non-scaling-stroke}.grid.base{stroke:var(--muted)}
.grid.time{stroke-opacity:.55}
.tick{fill:var(--muted);font-size:11px;font-family:inherit;font-variant-numeric:tabular-nums}
.band{fill:var(--band)}
.pred,.obs{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round;vector-effect:non-scaling-stroke}
.pred{stroke:var(--series-1)}.obs{stroke:var(--series-2);stroke-width:1.5}
.obs-area{fill:var(--series-2);fill-opacity:.13;stroke:none}.obs-dot{fill:var(--series-2)}
.hair{stroke:var(--ink);stroke-width:1;stroke-opacity:.5;vector-effect:non-scaling-stroke}.hit{fill:transparent}
.explain{margin:26px 0 0;border-top:1px solid var(--line);padding-top:10px}
.explain summary{cursor:pointer;font:600 15px var(--display);letter-spacing:.01em}
.explain .cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:0 40px;font-size:14px;line-height:1.6;max-width:none}
.explain p{margin:10px 0}.explain [lang=zh]{line-height:1.75}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--muted);border-radius:4px;padding:7px 10px;font-size:13px;box-shadow:0 4px 14px rgba(10,20,30,.18);z-index:5;white-space:nowrap}
#tip div:first-child{color:var(--muted);font-size:12px;margin-bottom:2px}
#tip b{font-variant-numeric:tabular-nums;display:inline-block;min-width:4.6em}
#tip .k{display:inline-block;width:12px;border-top:3px solid;vertical-align:middle;margin-right:7px;border-radius:2px}
"""

_SCRIPT = """
(function(){var D=JSON.parse(document.getElementById('chart-data').textContent),tip=document.getElementById('tip');
function pct(v){return v==null?'none':(100*v).toFixed(v<0.001?4:3)+'%%'}
function row(cls,label,v){var r=document.createElement('div'),k=document.createElement('span'),b=document.createElement('b');
k.className='k';k.style.borderColor='var(--series-'+cls+')';b.textContent=pct(v);r.appendChild(k);r.appendChild(b);
r.appendChild(document.createTextNode(label));return r}
document.querySelectorAll('.strip svg').forEach(function(svg){var d=D[svg.dataset.key],hair=svg.querySelector('.hair'),n=d.t.length;
var t0=d.t[0],t1=d.t[n-1];document.querySelectorAll('.strip svg').forEach(function(o){var e=D[o.dataset.key];
if(o.dataset.key.charAt(0)===svg.dataset.key.charAt(0)){t0=Math.min(t0,e.t[0]);t1=Math.max(t1,e.t[e.t.length-1])}});
function move(ev){var box=svg.getBoundingClientRect(),x=(ev.clientX-box.left)/box.width*%(width)d,
f=Math.min(1,Math.max(0,(x-%(left)d)/(%(width)d-%(left)d-%(right)d))),t=t0+f*(t1-t0),lo=0,hi=n-1;
while(hi-lo>1){var m=(lo+hi)>>1;if(d.t[m]<=t)lo=m;else hi=m}
var i=(t-d.t[lo]<=d.t[hi]-t)?lo:hi,hx=%(left)d+(%(width)d-%(left)d-%(right)d)*(t1>t0?(d.t[i]-t0)/(t1-t0):0);
hair.setAttribute('x1',hx);hair.setAttribute('x2',hx);hair.setAttribute('visibility','visible');
tip.textContent='';var h=document.createElement('div');
h.textContent=new Date(d.t[i]*1000).toISOString().slice(0,16).replace('T',' ')+' UTC';tip.appendChild(h);
tip.appendChild(row(1,'forecast',d.pred[i]));tip.appendChild(row(2,'observed',d.obs[i]));tip.hidden=false;
var w=tip.offsetWidth;tip.style.left=Math.min(ev.clientX+14,window.innerWidth-w-8)+'px';tip.style.top=(ev.clientY+14)+'px'}
svg.addEventListener('pointermove',move);svg.addEventListener('pointerdown',move);
svg.addEventListener('pointerleave',function(){tip.hidden=true;hair.setAttribute('visibility','hidden')})})})();
"""
