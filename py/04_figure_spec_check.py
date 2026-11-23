#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_figure_spec_check.py —— 按 PLOS ONE 图片技术规范自检

PLOS ONE 要求（期刊 Figures 页面）：
  格式   TIFF / EPS 优先（PNG、JPEG 可接收但非首选）
  分辨率 300–600 dpi
  宽度   2.63–7.5 in（6.68–19.05 cm）
  高度   ≤ 8.75 in（22.23 cm）
  大小   ≤ 10 MB
  字号   图内 8–12 pt（Arial / Times / Symbol）

注意：在获得 provisional Editorial Accept 之前，PLOS 豁免全部格式要求；
      但进入制作流程后不达标的图会被退回重做，因此建议提前自查。

用法：python 04_figure_spec_check.py --dir ../figures_R2
"""

import argparse
import os
import sys

from PIL import Image

MIN_W, MAX_W, MAX_H = 2.63, 7.5, 8.75
MIN_DPI, MAX_DPI = 300, 600
MAX_MB = 10


def check(path):
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    try:
        im = Image.open(path)
    except Exception as e:
        return name, [f"无法读取：{e}"], [], None

    dpi = round(im.info.get("dpi", (0, 0))[0])
    w_in, h_in = im.size[0] / dpi, im.size[1] / dpi if dpi else (0, 0)

    issues, notes = [], []
    if not dpi:
        issues.append("缺少 dpi 元数据")
    elif not (MIN_DPI <= dpi <= MAX_DPI):
        issues.append(f"dpi={dpi:.0f}，应在 {MIN_DPI}–{MAX_DPI}")
    if w_in and not (MIN_W <= w_in <= MAX_W):
        issues.append(f"宽 {w_in:.2f} in，应在 {MIN_W}–{MAX_W} in（当前 {im.size[0]} px）")
    if h_in and h_in > MAX_H:
        issues.append(f"高 {h_in:.2f} in，超过 {MAX_H} in")
    if size_mb > MAX_MB:
        issues.append(f"文件 {size_mb:.1f} MB，超过 {MAX_MB} MB")
    if im.format not in ("TIFF", "EPS"):
        notes.append(f"格式 {im.format}（PLOS 首选 TIFF/EPS，PNG 可接收）")

    return name, issues, notes, (im.size, dpi, w_in, h_in, size_mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    exts = (".png", ".tif", ".tiff", ".eps", ".jpg", ".jpeg")
    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(exts))
    if not files:
        sys.exit(f"{args.dir} 下没有图片")

    bad = 0
    for f in files:
        name, issues, notes, info = check(os.path.join(args.dir, f))
        if info:
            (px, dpi, w, h, mb) = info
            head = f"{name:46s} {px[0]}x{px[1]}px {dpi:.0f}dpi {w:.2f}x{h:.2f}in {mb:.1f}MB"
        else:
            head = name
        print(head)
        for i in issues:
            print(f"    ! {i}")
        for n in notes:
            print(f"    · {n}")
        bad += bool(issues)

    print(f"\n合计 {len(files)} 张，其中 {bad} 张存在不合规项。")
    if bad:
        print("处理建议：宽度超限的图请按 7.0 in 宽、目标字号 9 pt 重新导出（不要用缩放降采样，")
        print("          否则图内字号会低于 8 pt）；导出后重跑本脚本确认。")


if __name__ == "__main__":
    main()
