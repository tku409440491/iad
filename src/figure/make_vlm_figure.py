#!/usr/bin/env python3
"""
make_vlm_figure.py — PatchCore-VLM 質化結果圖產生器

產出一張論文用圖：
    上排：原始影像 | PatchCore 熱力圖
    中段：VLM 判斷依據（reason）
    下段：PatchCore 分數 / VLM 分數 / Combined 分數 與判定

兩種用法
--------
1) 由 PatchCore checkpoint 即時產生熱力圖（完整流程）
   python make_vlm_figure.py --category breakfast_box --gt logical --stem 000

2) 直接餵兩張現成影像（最穩，不依賴 anomalib API）
   python make_vlm_figure.py --from-images 原圖.png 熱力圖.png \
       --category breakfast_box --gt logical --stem 000

分數與 reason 一律從 JSONL 讀取，不手動輸入，避免與論文表格不一致。
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from src.config import CODE_DIR, DATA_ROOT

# ── 預設路徑（依你的環境） ────────────────────────────────────────────
LOCO_ROOT = DATA_ROOT / "mvtec_loco"
MODELS_DIR = CODE_DIR / "results/patchcore_models"
DEFAULT_JSONL = CODE_DIR / "results/patchcore_vlm_text_combined_lqsq_counting_results_bak.jsonl"

GT_DIRNAME = {
    "good": "test/good",
    "logical": "test/logical_anomalies",
    "structural": "test/structural_anomalies",
}

# ── 配色（light surface，論文列印用） ─────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8882"
HAIRLINE = "#e3e2dd"
C_PATCHCORE = "#2a78d6"   # categorical slot 1
C_VLM = "#eb6834"         # categorical slot 2
C_ANOMALY = "#d03b3b"     # status critical
C_NORMAL = "#0ca30c"      # status good


# ── CJK 字型 ─────────────────────────────────────────────────────────
def ensure_cjk_bold(family: str = "Noto Sans CJK TC") -> bool:
    """
    確保 matplotlib 有該家族的粗體 face，否則 fontweight='bold' 會靜默退回一般字重。
    Noto CJK 的粗體封在 NotoSansCJK-Bold.ttc（集合），matplotlib 只讀第一個 face（JP），
    因此用 fontTools 從集合抽出「Noto Sans CJK TC」粗體 face 存成獨立 .otf 後註冊。
    成功回傳 True。
    """
    from matplotlib import font_manager as fm
    # 已經有該家族的粗體？
    try:
        path = fm.findfont(fm.FontProperties(family=family, weight="bold"),
                           fallback_to_default=False)
        f = fm.get_font(path)
        if family.lower() in (f.family_name or "").lower() \
           and "bold" in (f.style_name or "").lower():
            return True
    except Exception:
        pass

    bold_otf = Path.home() / ".local/share/fonts/NotoSansCJKtc-Bold.otf"
    if not bold_otf.exists():
        src = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
        if not src.exists():
            return False
        try:
            from fontTools.ttLib.ttCollection import TTCollection
            coll = TTCollection(str(src))
            target = None
            for font in coll.fonts:
                fam = (font["name"].getDebugName(1) or "").strip()
                if fam == family and "Mono" not in fam:
                    target = font
                    break
            if target is None:
                return False
            bold_otf.parent.mkdir(parents=True, exist_ok=True)
            target.save(str(bold_otf))
        except Exception as e:
            sys.stderr.write(f"[警告] 抽取 CJK Bold face 失敗：{e}\n")
            return False

    fm.fontManager.addfont(str(bold_otf))
    return True


def setup_cjk_font() -> str:
    """
    為什麼需要這步：matplotlib 預設字型（DejaVu Sans）沒有中日韓字符，
    reason 是繁體中文，直接畫會變成一排空心方框「□□□」。
    """
    # 真正的繁體中文字型（優先）。Noto Sans CJK TC 以獨立 .otf 提供時，
    # matplotlib 會直接以 TC 家族名登記（.ttc 集合則只露第一個 face，通常是 JP）。
    prefer_tc = [
        "Noto Sans CJK TC", "Noto Sans TC", "Noto Serif CJK TC",
        "Source Han Sans TW", "Microsoft JhengHei", "PingFang TC",
        "WenQuanYi Zen Hei", "AR PL UMing TW",
    ]
    # 最後手段 fallback：CJK 集合的非繁體 face（JP/SC/HK）。這些能顯示漢字、
    # 不會變空心方框，但字形是日文/簡中/港標變體，非台灣繁體寫法——
    # 選到時務必印警告，避免下次又靜默選到 JP 而沒人發現。
    fallback_nontc = [
        "Noto Sans CJK HK", "Noto Sans CJK SC", "Noto Sans CJK JP",
        "Noto Serif CJK JP", "AR PL UMing CN",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}

    for name in prefer_tc:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            if not ensure_cjk_bold(name):
                sys.stderr.write(
                    f"[警告] 找不到「{name}」的粗體 face，粗體標籤會退回一般字重。\n")
            return name

    for name in fallback_nontc:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            sys.stderr.write(
                f"\n[警告] 找不到繁體中文字型，改用 fallback「{name}」。\n"
                f"  此為 CJK 集合的非繁體 face，字形為日文/簡中/港標變體，"
                f"非台灣繁體寫法（例如「門」「骨」「直」等字形會不同）。\n"
                f"  要正確繁體字形：放一份 Noto Sans CJK TC 的 .otf 到 "
                f"~/.local/share/fonts/，再 rm -rf ~/.cache/matplotlib 後重跑。\n\n"
            )
            return name

    sys.stderr.write(
        "\n[錯誤] 找不到任何中日韓字型，中文會變成空心方框。\n"
        "  WSL2/Ubuntu 安裝方式：\n"
        "    sudo apt update && sudo apt install -y fonts-noto-cjk\n"
        "    rm -rf ~/.cache/matplotlib\n"
        "  安裝後重跑本腳本即可。\n\n"
    )
    sys.exit(1)


# ── 中文斷行（全形算 2 格，含行首禁則） ─────────────────────────────
def _w(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("F", "W", "A") else 1


# 行首禁則字元：這些標點不可出現在行首，斷行時往前一行帶
_LEADING_FORBIDDEN = set("，。、；：！？）」』》〉】,.;:!?)]}%")


def wrap_cjk(text: str, width: int = 86, max_lines: int | None = None) -> list[str]:
    lines, cur, cw = [], "", 0
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur, cw = "", 0
            continue
        w = _w(ch)
        # 禁則：若在此斷行，下一行行首會是標點，則不斷，讓標點留在本行
        if cw + w > width and ch not in _LEADING_FORBIDDEN:
            lines.append(cur)
            cur, cw = "", 0
        cur += ch
        cw += w
    if cur:
        lines.append(cur)
    lines = lines or [""]

    # 超過 max_lines 就截斷，並在最後一行結尾加「…」
    if max_lines is not None and len(lines) > max_lines:
        kept = lines[:max_lines]
        last = kept[-1]
        while last and (sum(_w(c) for c in last) + 1) > width:
            last = last[:-1]
        kept[-1] = last.rstrip() + "…"
        lines = kept
    return lines


# ── 讀取 JSONL 紀錄 ──────────────────────────────────────────────────
def load_record(jsonl: Path, category: str, gt: str, stem: str) -> dict:
    if not jsonl.exists():
        sys.exit(f"[錯誤] 找不到結果檔：{jsonl}")
    want_stem = str(stem)
    cands = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("category") != category:
                continue
            if str(r.get("stem", "")).lstrip("0") != want_stem.lstrip("0"):
                continue
            cands.append(r)

    gt_key = {"good": "good", "logical": "logical", "structural": "structural"}[gt]
    for r in cands:
        if gt_key in str(r.get("gt_class", "")):
            return r
    if cands:
        sys.stderr.write(f"[警告] 找不到 gt={gt}，改用第一筆相符紀錄\n")
        return cands[0]
    sys.exit(f"[錯誤] JSONL 中找不到 {category}/{gt}/{stem}")


# ── PatchCore 熱力圖 ─────────────────────────────────────────────────
def patchcore_heatmap(category: str, image_path: Path, subdir: str = "Patchcore",
                      save_prefix: Path | None = None):
    """
    回傳 (原圖 RGB ndarray, 熱力圖疊加 RGB ndarray)。

    PatchCore 推論**完全複用** patchcore_vlm_pipeline_combined_lqsq_counting.py 的
    既有邏輯，不自行另寫一套 anomalib 呼叫（版本對不上會失敗）：
      - load_patchcore()：checkpoint 挑選規則（latest/ 否則取最新 mtime）、
        load_from_checkpoint(weights_only=False)、model.eval() 皆照 pipeline。
      - infer_image()：影像前處理用 model.pre_processor.transform，
        經 predict_step 取得 pred_score 與 anomaly_map。
      - anomaly_map_to_image()：熱力圖正規化（pixel_min/max + jet colormap）照 pipeline。
    熱力圖只是在上述結果之上做 alpha 疊加顯示，不影響論文數值。
    若匯入或 API 失敗，請改用 --from-images 直接餵兩張現成影像。
    """
    # 複用 pipeline 的推論模組（匯入即可，無副作用、不需 API key）
    import src.pipeline.patchcore_vlm_pipeline_combined_lqsq_counting as P
    model, ckpt = P.load_patchcore(category, subdir=subdir)
    print(f"  checkpoint：{ckpt}")

    raw_score, amap = P.infer_image(model, image_path)   # amap: (H, W)
    pixel_min = float(model.post_processor.pixel_min)
    pixel_max = float(model.post_processor.pixel_max)

    # 熱力圖 jet colormap 沿用 pipeline 的 anomaly_map_to_image（相同 pixel_min/max 正規化）
    heat_pil = P.anomaly_map_to_image(amap, pixel_min, pixel_max)

    # 關鍵：維持原圖「原尺寸、原寬高比」，把 anomaly map 的 jet 圖以 bilinear 上採樣到
    # 原圖 (H0, W0) 後再疊加——不可反過來把原圖縮成 256×256 正方形（會橫向壓扁）。
    orig_full = Image.open(image_path).convert("RGB")
    W0, H0 = orig_full.size
    heat_full = heat_pil.convert("RGB").resize((W0, H0), Image.BILINEAR)
    of = np.asarray(orig_full, dtype=np.float32) / 255.0
    hf = np.asarray(heat_full, dtype=np.float32) / 255.0
    overlay_full = np.clip(0.55 * of + 0.45 * hf, 0, 1)

    orig_u8 = np.asarray(orig_full, dtype=np.uint8)              # (H0, W0, 3)
    heat_u8 = (overlay_full * 255).astype(np.uint8)             # (H0, W0, 3)

    # 另存獨立的原圖與熱力圖（論文單獨使用），即上面同一份全解析度影像。
    if save_prefix is not None:
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        orig_path = save_prefix.with_name(save_prefix.name + "_orig.png")
        heat_path = save_prefix.with_name(save_prefix.name + "_heat.png")
        orig_full.save(orig_path)
        Image.fromarray(heat_u8).save(heat_path)
        print(f"  另存原圖　：{orig_path}")
        print(f"  另存熱力圖：{heat_path}  （{W0}×{H0}）")

    return orig_u8, heat_u8


# ── 版面組合 ─────────────────────────────────────────────────────────
def compose_figure(orig, heat, rec, out_path: Path,
                   show_counting: bool = True,
                   title: str | None = None,
                   wrap_width: int = 88,
                   dpi: int = 300,
                   max_reason_lines: int = 3,
                   max_img_height: float = 2.8):
    pc = float(rec.get("patchcore_norm_score", 0.0))
    vlm = float(rec.get("vlm_score", 0.0))
    comb = float(rec.get("combined_score", 0.5 * pc + 0.5 * vlm))
    reason = (rec.get("reason") or "").strip()
    counting = (rec.get("counting_text") or "").strip()
    is_anom = comb >= 0.5

    # 去掉 counting_text 開頭重複的「計數分析：」（標題已有同名），避免重複
    for pre in ("計數分析：", "計數分析:"):
        if counting.startswith(pre):
            counting = counting[len(pre):].strip()
            break

    # ⚠️ = U+26A0 + U+FE0F，字型無 U+FE0F 會渲染成空心方框。
    # 本專案字型 Noto Sans CJK TC 無 ✗(U+2717)，會變方框；改用 ×(U+00D7)：
    # 筆畫粗細與既有 ✓(U+2713) 接近、Noto Sans CJK TC 與 DejaVu Sans 皆有此字，
    # 不像 ╳(U+2573，box-drawing 全形) 筆畫過粗且全形對齊可能異常。故：
    #   - 濾掉所有變異選擇器 U+FE0F / U+FE0E
    #   - 把 ⚠(U+26A0) 與各種 ballot-X（U+2717/2718/2715/2716）統一換成 ×(U+00D7)
    def _strip_emoji(s: str) -> str:
        s = s.translate({0xFE0F: None, 0xFE0E: None})
        for cp in ("⚠", "✗", "✘", "✕", "✖"):
            s = s.replace(cp, "×")
        return s
    counting = _strip_emoji(counting)
    reason = _strip_emoji(reason)

    log_s = rec.get("vlm_logical_score")
    str_s = rec.get("vlm_structural_score")

    body = wrap_cjk(reason, wrap_width, max_lines=max_reason_lines)
    cnt_lines = wrap_cjk(counting, wrap_width) if (show_counting and counting) else []

    # ── 影像列：以「高度」為主控（orig = (H, W, 3)）──
    # 直式類別（如 juice_bottle 0.5）若依半欄寬配置，高度會爆表；橫幅類別又太扁。
    # 改成影像列固定高度 = max_img_height，各類別圖面高度因此趨於一致。
    H_src, W_src = orig.shape[0], orig.shape[1]
    img_aspect = W_src / H_src

    # ── 尺寸（IEEE 雙欄 \textwidth ≈ 7.16 吋）──
    FIG_W = 7.16
    side, gap = 0.075, 0.028
    w_ax = (1 - 2 * side - gap) / 2            # 單一面板槽寬（figure fraction）
    panel_img_in = max_img_height             # 影像列固定高度（吋）
    label_pad_in = 0.30                        # 面板標題「(a) …」空間

    h_img = panel_img_in + label_pad_in
    h_cnt = (0.24 * len(cnt_lines) + 0.28) if cnt_lines else 0.0
    h_body = 0.24 * len(body) + 0.40
    h_score = 0.92
    h_title = 0.40 if title else 0.0
    FIG_H = h_title + h_img + h_cnt + h_body + h_score + 0.26

    print(f"  圖面尺寸：{FIG_W:.2f} × {FIG_H:.2f} 吋  "
          f"（影像列高 {max_img_height:.2f} 吋，來源比例 {img_aspect:.3f}）")

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)

    y = 1.0
    if title:
        fig.text(0.5, 1.0 - 0.08 / FIG_H, title, ha="center", va="top",
                 fontsize=11, fontweight="bold", color=INK)
        y -= h_title / FIG_H

    # ── 上排：兩張圖。槽高固定 = max_img_height，aspect='equal' 保持等比：
    #    直式→填滿高度、左右留白；橫幅→填滿寬度、上下留白。影像於槽內置中。──
    top = y - h_img / FIG_H
    for i, (img, label) in enumerate([(orig, "(a) 原始影像"),
                                      (heat, "(b) PatchCore 異常熱力圖")]):
        ax = fig.add_axes([side + i * (w_ax + gap), top + 0.02 / FIG_H,
                           w_ax, panel_img_in / FIG_H])
        ax.imshow(img, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(HAIRLINE); s.set_linewidth(0.8)
        ax.set_title(label, fontsize=9, color=INK_2, pad=5)
    y = top

    # ── 計數分析（若有） ──
    if cnt_lines:
        y -= 0.20 / FIG_H
        fig.text(side, y, "計數分析", fontsize=8, color=C_VLM,
                 fontweight="bold", va="top")
        y -= 0.24 / FIG_H
        for ln in cnt_lines:
            fig.text(side, y, ln, fontsize=8, color=INK_2, va="top")
            y -= 0.24 / FIG_H

    # ── VLM 判斷依據 ──
    y -= 0.16 / FIG_H
    fig.text(side, y, "VLM 判斷依據", fontsize=8.5, color=INK,
             fontweight="bold", va="top")
    y -= 0.28 / FIG_H
    for ln in body:
        fig.text(side, y, ln, fontsize=8.5, color=INK, va="top", linespacing=1.45)
        y -= 0.24 / FIG_H

    # ── 分數列 ──
    y -= 0.12 / FIG_H
    ax = fig.add_axes([side, y - (h_score - 0.22) / FIG_H,
                       1 - 2 * side, (h_score - 0.22) / FIG_H])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    verdict_c = C_ANOMALY if is_anom else C_NORMAL
    verdict_t = "異常" if is_anom else "正常"

    boxes = [
        (0.0, 27.0, "PatchCore 偵測器", f"{pc:.3f}", "權重 0.5", C_PATCHCORE, False),
        (36.5, 27.0, "VLM 語意判斷", f"{vlm:.3f}", "權重 0.5", C_VLM, False),
        (73.0, 27.0, "Combined 分數", f"{comb:.3f}", f"判定：{verdict_t}", verdict_c, True),
    ]
    for x0, w, lab, val, sub, color, strong in boxes:
        ax.add_patch(FancyBboxPatch(
            (x0, 8), w, 80,
            boxstyle="round,pad=0,rounding_size=3",
            linewidth=1.6 if strong else 1.1,
            edgecolor=color,
            facecolor=color + ("1f" if strong else "12"),
            transform=ax.transData, clip_on=False))
        cx = x0 + w / 2
        ax.text(cx, 76, lab, ha="center", va="center", fontsize=8, color=INK_2)
        ax.text(cx, 50, val, ha="center", va="center",
                fontsize=16.5 if strong else 15,
                fontweight="bold", color=INK)
        ax.text(cx, 22, sub, ha="center", va="center",
                fontsize=8.5 if strong else 8,
                color=color if strong else INK_MUTED,
                fontweight="bold" if strong else "normal")

    ax.text(31.8, 48, "+", ha="center", va="center", fontsize=14, color=INK_MUTED)
    ax.text(68.3, 48, "=", ha="center", va="center", fontsize=14, color=INK_MUTED)

    if log_s is not None and str_s is not None:
        ax.text(36.5 + 27.0 / 2, -4,
                f"logical {int(log_s)} / structural {int(str_s)}",
                ha="center", va="center", fontsize=7.5, color=INK_MUTED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.12)
    pdf = out_path.with_suffix(".pdf")
    fig.savefig(pdf, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path, pdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--gt", default="logical", choices=["good", "logical", "structural"])
    p.add_argument("--stem", default="000")
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--from-images", nargs=2, metavar=("ORIG", "HEAT"),
                   help="直接指定原圖與熱力圖，跳過 PatchCore 推論")
    p.add_argument("--subdir", default="Patchcore",
                   help="checkpoint 子目錄，indval 模式用 indval/Patchcore")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--no-counting", action="store_true")
    p.add_argument("--wrap", type=int, default=88)
    p.add_argument("--max-reason-lines", type=int, default=3, dest="max_reason_lines",
                   help="VLM 判斷依據最多顯示幾行，超過在末行加「…」（預設 3）")
    p.add_argument("--max-img-height", type=float, default=2.8, dest="max_img_height",
                   help="影像列固定高度（吋），直式/橫幅類別皆以此為準（預設 2.8）")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    font = setup_cjk_font()
    print(f"  中文字型：{font}")

    rec = load_record(args.jsonl, args.category, args.gt, args.stem)
    print(f"  紀錄：{rec.get('category')}/{rec.get('gt_class')}/{rec.get('stem')}")

    out = args.out or (CODE_DIR / "results/figures" /
                       f"qual_{args.category}_{args.gt}_{args.stem}.png")

    if args.from_images:
        orig = np.asarray(Image.open(args.from_images[0]).convert("RGB"))
        heat = np.asarray(Image.open(args.from_images[1]).convert("RGB"))
    else:
        img_path = LOCO_ROOT / args.category / GT_DIRNAME[args.gt] / f"{args.stem}.png"
        if not img_path.exists():
            sys.exit(f"[錯誤] 找不到影像：{img_path}")
        # save_prefix 另存獨立原圖與熱力圖（檔名 _orig.png / _heat.png）
        orig, heat = patchcore_heatmap(args.category, img_path, args.subdir,
                                       save_prefix=out.with_suffix(""))

    png, pdf = compose_figure(orig, heat, rec, out,
                              show_counting=not args.no_counting,
                              title=args.title, wrap_width=args.wrap, dpi=args.dpi,
                              max_reason_lines=args.max_reason_lines,
                              max_img_height=args.max_img_height)
    print(f"\n完成：\n  {png}\n  {pdf}")


if __name__ == "__main__":
    main()
