# -*- coding: utf-8 -*-
"""Build a clean, self-contained HTML index for the two-colour diversity series
(two_color_series/index.html). Embeds the static contact sheet, summarises the
corpus, maps the directory layout, and tabulates every volume."""
import base64, glob, json, os

OUT = os.path.join(os.path.dirname(__file__), "output")
SERIES = os.path.join(OUT, "two_color_series")
VOLS = os.path.join(SERIES, "volumes")
REPORTS = os.path.join(SERIES, "reports")
INDEX = os.path.join(SERIES, "index.html")


def b64(p):
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def gather():
    rows = []
    for vd in sorted(glob.glob(os.path.join(VOLS, "seed*"))):
        seed = int(os.path.basename(vd).replace("seed", ""))
        r = {"seed": seed, "dff": None, "spikes": None, "vessel": None, "gcamp": False, "tdt": False}
        gm = os.path.join(vd, "gcamp", "metadata.json")
        if os.path.exists(gm):
            m = json.load(open(gm)); r["gcamp"] = True
            r["dff"] = m.get("dff_p99"); r["spikes"] = m.get("total_spikes")
        r["tdt"] = os.path.exists(os.path.join(vd, "tdt", "movies.npz"))
        vs = os.path.join(vd, "stub", "volume_stats.npz")
        if os.path.exists(vs):
            import numpy as np
            r["vessel"] = float(np.load(vs)["vessel_frac"])
        rows.append(r)
    return rows


def main():
    rows = gather()
    n = len(rows)
    dffs = [r["dff"] for r in rows if r["dff"]]
    dff_lo, dff_hi = (min(dffs), max(dffs)) if dffs else (0, 0)
    sheet = b64(os.path.join(REPORTS, "overview_contact_sheet.png"))
    gif_mb = os.path.getsize(os.path.join(REPORTS, "overview_all_sims.gif")) / 1e6

    trows = "\n".join(
        f'<tr><td class="n">{r["seed"]}</td>'
        f'<td class="n">{r["dff"]:.3f}</td>'
        f'<td class="n">{r["spikes"]:,}</td>'
        f'<td class="n">{r["vessel"]*100:.3f}%</td>'
        f'<td>{"●" if r["gcamp"] else "○"} {"●" if r["tdt"] else "○"}</td></tr>'
        for r in rows if r["dff"])

    html = f"""<title>Two-colour diversity series — index</title>
<style>
  :root {{
    --paper:#fbfaf8; --ink:#181a1f; --sub:#565b66; --line:#e6e2db; --card:#fff;
    --accent:#3d6b8e; --gc:#2f7d5b; --td:#b0472f; --good:#2f8f5b;
    --mono:"SF Mono","Cascadia Code","Consolas",ui-monospace,monospace;
    --serif:"Iowan Old Style","Palatino Linotype","Songti SC","Noto Serif CJK SC",serif;
    --sans:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  }}
  @media (prefers-color-scheme:dark){{:root{{--paper:#111318;--ink:#e9e7e2;--sub:#9aa0ab;--line:#282c34;--card:#181b21;--accent:#6fa3c9;--gc:#5cc194;--td:#e0705d;}}}}
  :root[data-theme="light"]{{--paper:#fbfaf8;--ink:#181a1f;--sub:#565b66;--line:#e6e2db;--card:#fff;--accent:#3d6b8e;}}
  :root[data-theme="dark"]{{--paper:#111318;--ink:#e9e7e2;--sub:#9aa0ab;--line:#282c34;--card:#181b21;--accent:#6fa3c9;--gc:#5cc194;--td:#e0705d;}}
  *{{box-sizing:border-box;}}
  body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.7;}}
  .wrap{{max-width:900px;margin:0 auto;padding:60px 26px 90px;}}
  .eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin:0 0 12px;}}
  h1{{font-family:var(--serif);font-weight:600;font-size:clamp(28px,5vw,40px);line-height:1.2;margin:0 0 14px;}}
  h2{{font-family:var(--serif);font-weight:600;font-size:23px;margin:48px 0 8px;}}
  p{{margin:11px 0;}} .lede{{font-size:18px;color:var(--sub);max-width:40em;}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:13px;margin:28px 0 6px;}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:17px 18px;}}
  .card .k{{font-size:12.5px;color:var(--sub);margin:0 0 9px;}}
  .card .v{{font-family:var(--serif);font-size:32px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1;}}
  .card .u{{font-size:12px;color:var(--sub);margin-top:7px;}}
  .pill{{display:inline-block;font-size:12.5px;font-weight:600;padding:4px 12px;border-radius:999px;color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent);}}
  figure{{margin:22px 0;}} figure img{{width:100%;border:1px solid var(--line);border-radius:10px;background:#111;}}
  figcaption{{font-size:13px;color:var(--sub);margin-top:9px;border-left:2px solid var(--accent);padding:2px 0 2px 12px;}}
  pre.tree{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;overflow-x:auto;font-family:var(--mono);font-size:13px;line-height:1.75;color:var(--ink);}}
  pre.tree b{{color:var(--accent);}} pre.tree i{{color:var(--sub);font-style:normal;}}
  .tblwrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin:18px 0;}}
  table{{width:100%;border-collapse:collapse;font-size:13.5px;}}
  th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;white-space:nowrap;}}
  th{{position:sticky;top:0;background:var(--card);font-size:11.5px;color:var(--sub);font-weight:600;}}
  td.n{{font-family:var(--mono);}}
  a{{color:var(--accent);}}
  .foot{{margin-top:54px;padding-top:18px;border-top:1px solid var(--line);font-family:var(--mono);font-size:12px;color:var(--sub);line-height:1.9;}}
</style>

<div class="wrap">
  <p class="eyebrow">calcia · 合成双色钙成像语料库</p>
  <h1>双色多样性系列 · 总索引</h1>
  <p class="lede">{n} 个独立生成的深层纹状体体积，每个都是一对逐像素共配准的
  GCaMP（动态）+ tdTomato（静态）宽场电影。不同 Phase-1 seed → 不同神经元分布 + 不同血管布局。</p>

  <div class="cards">
    <div class="card"><p class="k">体积数</p><div class="v">{n}</div><p class="u">共配准双色对</p></div>
    <div class="card"><p class="k">每体积</p><div class="v">4500</div><p class="u">神经元 · 500³µm 深层</p></div>
    <div class="card"><p class="k">dF/F p99</p><div class="v">{dff_lo:.2f}</div><p class="u">–{dff_hi:.2f}（真实 ~0.20）</p></div>
    <div class="card"><p class="k">几何多样性</p><div class="v">✓</div><p class="u">soma/血管 XY |r|≈0.02</p></div>
  </div>
  <p><span class="pill">DIVERSE · 判别性维度全部去相关</span></p>

  <h2>总览 · 一览全部 {n} 个体积</h2>
  <figure>
    <img alt="全部体积的活动足迹接触印相（每格一个体积，标注 seed）" src="{sheet}">
    <figcaption>静态接触印相：每格为一个体积 GCaMP 电影的时间最大投影（活动足迹 + 血管结构），
    各格血管/void 图样各异 → 几何多样性一目了然。动画版见
    <code>reports/overview_all_sims.gif</code>（{gif_mb:.0f} MB，随时间播放全部体积）。</figcaption>
  </figure>

  <h2>目录结构</h2>
  <pre class="tree"><b>two_color_series/</b>
├─ <b>volumes/</b>            <i>{n} 个体积，每个一个文件夹</i>
│   └─ <b>seedNNNN/</b>
│       ├─ gcamp/         <i>GCaMP 动态电影 (movies.npz, traces.npz, metadata)</i>
│       ├─ tdt/           <i>tdTomato 静态电影（共配准）</i>
│       └─ stub/          <i>Phase-1 体积句柄 + volume_stats.npz</i>
├─ <b>reports/</b>
│   ├─ overview_contact_sheet.png   <i>本页嵌入的静态总览</i>
│   ├─ overview_all_sims.gif        <i>动画总览</i>
│   ├─ diversity_report.png / .json <i>差异性验证</i>
│   ├─ two_color_series_manifest.csv
│   └─ simtrace/                    <i>sim-trace 集群招募重扫描</i>
└─ <b>index.html</b>       <i>本页</i>

<i>其它：</i> _shared/ <i>(Phase-1 缓存)</i> · _archive/ <i>(旧的一次性 run)</i> · logs/</pre>

  <h2>逐体积清单</h2>
  <div class="tblwrap"><table>
    <thead><tr><th>seed</th><th>dF/F p99</th><th>spikes</th><th>血管占比</th><th>G / T</th></tr></thead>
    <tbody>
    {trows}
    </tbody>
  </table></div>
  <p style="font-size:12.5px;color:var(--sub)">G / T = GCaMP / tdTomato 通道是否齐备（● 有 · ○ 无）。</p>

  <div class="foot">
    calcia/examples/output/two_color_series/ · {n} volumes · 500µm deep striatum · GCaMP6f + tdTomato<br>
    optics: two-scale PSF (halo 28 / w 0.8), composite off, flat illum · 200 fr @ 20 Hz
  </div>
</div>
"""
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {INDEX}  ({os.path.getsize(INDEX)/1e6:.2f} MB, {n} volumes)")


if __name__ == "__main__":
    main()
