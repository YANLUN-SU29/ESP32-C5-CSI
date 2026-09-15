"""分段分析：以「憋氣段」作為真正的負樣本。

為什麼需要這支程式
------------------
`HOLD` 檔的錄製協定是：

    0~30s 以 25 BPM 正常呼吸 → 舉手 → 30~60s 憋氣 → 舉手 → 60~90s 恢復呼吸

也就是說**整檔有 2/3 的時間受測者確實在呼吸**。早期把 HOLD 整檔當成「沒有呼吸」
的負樣本是錯的，會同時低估判別力（負樣本裡混入真訊號）與誤判誤報來源。

正確的負樣本是中間那段憋氣。但憋氣段只有 30 秒，與 GEO/EMPTY 的 120 秒不可直接比較
（突出度的尺度隨 Welch 平均段數改變）。因此本程式把**所有**檔案切成等長片段後才比較。

分段定義（頭尾各留 3 秒避開舉手造成的暫態）
    呼吸前段  3~27s      憋氣段  33~57s      呼吸後段  63~87s

GEO 與 EMPTY 取相同的三個時間窗，確保片段長度與處理方式完全一致。

用法：
    python segment_analyze.py
    python segment_analyze.py --seconds 24 --band breath
"""
import argparse
import glob
import os
import re
import sys
import warnings
from math import comb

import numpy as np
import pandas as pd
from scipy.signal import welch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzer_gui import CSIFeatureViewer, MODES  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NFFT = 8192
TRUTH_BPM = 12.5          # 節拍器 25 BPM、兩拍一次呼吸
GUARD = 3.0               # 舉手前後各避開的秒數


def segments(seconds):
    """(標籤, 起, 迄)。與 HOLD 的錄製協定對齊。"""
    return [('呼吸前段', GUARD, GUARD + seconds),
            ('憋氣段',   30 + GUARD, 30 + GUARD + seconds),
            ('呼吸後段', 60 + GUARD, 60 + GUARD + seconds)]


def parse_label(name):
    if name.startswith('CSI_EMPTY'):
        return 'EMPTY', None
    m = re.match(r'CSI_(GEO|HOLD)_(P\d+)_', name)
    return (m.group(1), m.group(2)) if m else ('OTHER', None)


def load(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.iloc[:, 0], format='%H:%M:%S.%f')
    t = (ts - ts.iloc[0]).dt.total_seconds().values
    fps = (len(t) - 1) / t[-1] if t[-1] > 0 else 60.0
    cols = list(df.columns[1:])
    a_cols = [c for c in cols if str(c).endswith('_A')] or cols
    amp = df[a_cols].values.astype(float)
    return t, fps, amp[:, ~np.all(amp == 0, axis=0)]


def seg_metric(seg, fps, band, helper):
    """回傳該片段最強子載波的 (突出度, 峰值頻率 次/分)。

    片段本身即一個 Welch 段（無分段平均），突出度會系統性偏高；
    但所有片段用同一套處理，彼此可比。
    """
    if len(seg) < 64:
        return np.nan, np.nan
    clean = np.empty_like(seg)
    for j in range(seg.shape[1]):
        clean[:, j] = helper._clean(seg[:, j])
    f, _ = welch(clean[:, 0], fs=fps, nperseg=len(clean), nfft=NFFT)
    best = (0.0, np.nan)
    for j in range(clean.shape[1]):
        _, P = welch(clean[:, j], fs=fps, nperseg=len(clean), nfft=NFFT)
        fr, pr = helper._band_peak(f, P, band)
        if not np.isnan(pr) and pr > best[0]:
            best = (pr, fr * 60)
    return best


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float('nan')
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description='CSI 分段分析：以憋氣段為負樣本')
    ap.add_argument('--seconds', type=float, default=24.0, help='每段長度（秒）')
    ap.add_argument('--band', default='breath', choices=list(MODES))
    ap.add_argument('--dir', default=DATA_DIR)
    args = ap.parse_args()
    band = MODES[args.band]['band']
    segs = segments(args.seconds)
    helper = CSIFeatureViewer.__new__(CSIFeatureViewer)

    paths = sorted(glob.glob(os.path.join(args.dir, '*.csv')))
    print(f"[分段分析] {len(paths)} 檔，每段 {args.seconds:.0f} 秒，頻段 {args.band} {band} Hz")
    rows = []
    for path in paths:
        name = os.path.basename(path)
        grp, pos = parse_label(name)
        t, fps, amp = load(path)
        day = re.search(r'_(20260\d{3})_', name)
        for lab, s0, s1 in segs:
            i0, i1 = int(np.searchsorted(t, s0)), int(np.searchsorted(t, s1))
            if i1 > len(t):
                continue
            pr, fr = seg_metric(amp[i0:i1], fps, band, helper)
            if np.isnan(pr):
                continue
            # 片段的真實狀態：只有 HOLD 的中段是「有人但沒呼吸」
            if grp == 'HOLD':
                state = '憋氣' if lab == '憋氣段' else '呼吸'
            elif grp == 'GEO':
                state = '呼吸'
            else:
                state = '空房間'
            rows.append(dict(name=name, grp=grp, pos=pos, seg=lab, state=state,
                             day=day.group(1) if day else '?', prom=pr, rate=fr))
        print(f"  {name[:40]:40s} 完成", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '.segment_metrics.csv'), index=False)

    print(f"\n{'=' * 92}\n各狀態彙總（片段長度 {args.seconds:.0f} 秒）\n{'=' * 92}")
    print(f"{'狀態':8s} {'n':>4s} {'突出度 中位':>12s} {'四分位':>16s} "
          f"{'頻率落在 12.5±1':>16s}")
    for st in ('呼吸', '憋氣', '空房間'):
        v = d[d.state == st]
        if v.empty:
            continue
        q1, q3 = v.prom.quantile([.25, .75])
        near = np.mean(np.abs(v.rate - TRUTH_BPM) <= 1.0) * 100
        print(f"{st:8s} {len(v):4d} {v.prom.median():12.1f} "
              f"{f'{q1:.1f} ~ {q3:.1f}':>16s} {near:15.0f}%")

    print(f"\n{'=' * 92}\n判別力（片段層級）\n{'=' * 92}")
    P = d[d.state == '呼吸'].prom.values
    A = d[d.state == '憋氣'].prom.values
    E = d[d.state == '空房間'].prom.values
    print(f"  呼吸 vs 憋氣（有人在場，唯一差異是呼吸）  AUC = {auc(P, A):.3f}"
          f"   n={len(P)} vs {len(A)}")
    print(f"  呼吸 vs 空房間                          AUC = {auc(P, E):.3f}"
          f"   n={len(P)} vs {len(E)}")
    print(f"  憋氣 vs 空房間（理想應接近 0.5）          AUC = {auc(A, E):.3f}"
          f"   n={len(A)} vs {len(E)}")

    # HOLD 檔內配對：同人、同位置、同日、相隔數十秒
    print(f"\n{'=' * 92}\nHOLD 檔內配對比較（最嚴格的對照：除呼吸外一切相同）\n{'=' * 92}")
    h = d[d.grp == 'HOLD']
    print(f"{'檔案':34s} {'呼吸段最高':>11s} {'憋氣段':>9s} {'比值':>8s}")
    wins = 0
    for nm, g in h.groupby('name'):
        br = g[g.state == '呼吸'].prom
        apn = g[g.state == '憋氣'].prom
        if br.empty or apn.empty:
            continue
        b, a_ = br.max(), apn.iloc[0]
        wins += b > a_
        print(f"{nm[:34]:34s} {b:11.1f} {a_:9.1f} {b / max(a_, 1e-9):7.2f}x")
    n_files = h.name.nunique()
    # 單尾符號檢定：H0 為兩段等機率較高，p = P(X >= wins | n, 0.5)
    pval = sum(comb(n_files, k) for k in range(wins, n_files + 1)) / 2 ** n_files
    print(f"\n  呼吸段 > 憋氣段的檔數：{wins}/{n_files}"
          f"　（單尾符號檢定 p = {pval:.4f}）")
    print("  註：這是最嚴格的對照——同一個人、同一位置、同一天、相隔數十秒，"
          "唯一的差異就是有沒有在呼吸。")


if __name__ == '__main__':
    main()
