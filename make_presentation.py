"""HJB Ladder Pricer — presentation for a non-expert audience."""

import io, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

sys.path.insert(0, "/Users/jshellen/Repos/finance/hjb_ladder")
import ladder_pricer as lp
from app import build_piecewise_centered_q_grid

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY      = RGBColor(0x0D, 0x1B, 0x2A)
DARK_CARD = RGBColor(0x16, 0x2A, 0x3E)
GOLD      = RGBColor(0xF0, 0xB3, 0x23)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LGREY     = RGBColor(0xC8, 0xD6, 0xE5)
TEAL      = RGBColor(0x2E, 0xCC, 0xC1)
BLUE_H    = RGBColor(0x42, 0xA5, 0xF5)
RED_H     = RGBColor(0xEF, 0x53, 0x50)

BG        = "#0D1B2A"
CARD      = "#162A3E"
GOLD_H    = "#F0B323"
TEAL_H    = "#2ECCC1"
GREY_H    = "#C8D6E5"
WHITE_H   = "#FFFFFF"
BLUE_MPL  = "#42A5F5"
RED_MPL   = "#EF5350"

W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H
BLANK = prs.slide_layouts[6]


# ── Slide primitives ───────────────────────────────────────────────────────────

def bg(slide):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = NAVY

def rect(slide, x, y, w, h, color=NAVY):
    s = slide.shapes.add_shape(1, x, y, w, h)
    s.line.fill.background()
    s.fill.solid()
    s.fill.fore_color.rgb = color
    return s

def text(slide, txt, x, y, w, h, size=22, bold=False,
         color=WHITE, align=PP_ALIGN.LEFT):
    b = slide.shapes.add_textbox(x, y, w, h)
    b.word_wrap = True
    tf = b.text_frame
    tf.word_wrap = True
    p  = tf.paragraphs[0]
    p.alignment = align
    r  = p.add_run()
    r.text = txt
    r.font.size  = Pt(size)
    r.font.bold  = bold
    r.font.color.rgb = color
    return b

def top_rule(slide):
    rect(slide, Inches(0), Inches(0), W, Inches(0.06), GOLD)

def slide_title(slide, title):
    top_rule(slide)
    text(slide, title, Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.75),
         size=32, bold=True, color=GOLD)
    rect(slide, Inches(0.4), Inches(0.9), Inches(12.5), Inches(0.025), DARK_CARD)


# ── Figure helpers ─────────────────────────────────────────────────────────────

def style_ax(ax, xlabel="", ylabel="", title="", title_color=GOLD_H):
    ax.set_facecolor(CARD)
    if xlabel: ax.set_xlabel(xlabel, color=GREY_H, fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, color=GREY_H, fontsize=11)
    if title:  ax.set_title(title, color=title_color, fontsize=13, fontweight="bold", pad=8)
    ax.tick_params(colors=GREY_H, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color(GREY_H); sp.set_alpha(0.25)
    ax.grid(True, alpha=0.1, color=GREY_H)

def to_stream(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf

def img(slide, stream, x=Inches(0.4), y=Inches(1.0),
        w=Inches(12.53), h=Inches(6.15)):
    slide.shapes.add_picture(stream, x, y, w, h)


# ── Narrative slide ────────────────────────────────────────────────────────────

def narrative(slide, title, body_lines, callout=None):
    """
    body_lines: list of (text, style) where style ∈ 'heading', 'body', 'detail'
    callout: (big_text, label) shown in a gold box on the right  — optional
    """
    bg(slide)
    slide_title(slide, title)

    body_w = Inches(7.6) if callout else Inches(12.2)

    txb = slide.shapes.add_textbox(Inches(0.4), Inches(1.05), body_w, Inches(6.1))
    txb.word_wrap = True
    tf = txb.text_frame
    tf.word_wrap = True

    first = True
    for line, style in body_lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        r = p.add_run()
        if style == "heading":
            r.text = line
            r.font.size  = Pt(22)
            r.font.bold  = True
            r.font.color.rgb = GOLD
            p.space_before = Pt(14)
        elif style == "body":
            r.text = "  " + line
            r.font.size  = Pt(20)
            r.font.color.rgb = WHITE
            p.space_before = Pt(6)
        else:   # detail
            r.text = "      — " + line
            r.font.size  = Pt(17)
            r.font.color.rgb = LGREY
            p.space_before = Pt(3)

    if callout:
        big, label = callout
        rect(slide, Inches(8.3), Inches(1.4), Inches(4.5), Inches(4.6), DARK_CARD)
        rect(slide, Inches(8.3), Inches(1.4), Inches(0.12), Inches(4.6), GOLD)
        text(slide, big,
             Inches(8.5), Inches(2.0), Inches(4.1), Inches(3.0),
             size=52, bold=True, color=GOLD, align=PP_ALIGN.CENTER)
        text(slide, label,
             Inches(8.5), Inches(5.0), Inches(4.1), Inches(0.8),
             size=16, color=LGREY, align=PP_ALIGN.CENTER)


def section_break(slide, title, subtitle=None):
    bg(slide)
    rect(slide, Inches(1.5), Inches(2.4), Inches(10.3), Inches(2.7), DARK_CARD)
    rect(slide, Inches(1.5), Inches(2.4), Inches(0.12), Inches(2.7), GOLD)
    text(slide, title, Inches(1.9), Inches(2.8), Inches(9.8), Inches(1.0),
         size=40, bold=True, color=GOLD)
    if subtitle:
        text(slide, subtitle, Inches(1.9), Inches(3.85), Inches(9.8), Inches(0.8),
             size=21, color=LGREY)


def title_card(slide, title, subtitle=None):
    bg(slide)
    rect(slide, Inches(0), Inches(3.1), Inches(0.14), Inches(1.5), GOLD)
    text(slide, title, Inches(0.45), Inches(2.85), Inches(12), Inches(1.3),
         size=54, bold=True, color=WHITE)
    if subtitle:
        text(slide, subtitle, Inches(0.45), Inches(4.2), Inches(12), Inches(0.9),
             size=26, color=LGREY)
    rect(slide, Inches(0), Inches(7.2), W, Inches(0.05), GOLD)


def caption_slide(slide, title, stream, caption):
    bg(slide)
    slide_title(slide, title)
    slide.shapes.add_picture(stream, Inches(0.4), Inches(1.05), Inches(12.53), Inches(5.75))
    text(slide, caption, Inches(0.4), Inches(6.9), Inches(12.5), Inches(0.5),
         size=14, color=LGREY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# Run the solver once — used by several figures
# ══════════════════════════════════════════════════════════════════════════════

print("Running solver…")

q_arr = build_piecewise_centered_q_grid(
    q_abs_max=20.0, fine_half_width=3.0, fine_step=0.25, coarse_step=1.0)
Q = [float(q) for q in q_arr]
SPOT   = 11.5
SPREAD = 20 / 10_000
SIGMA  = 20 / 10_000

cfg = lp.SolverConfig()
cfg.q_grid    = Q;  cfg.dt = 0.002;  cfg.n_iter = 140
cfg.spread    = SPREAD;  cfg.spot = SPOT;  cfg.spot_drift = 0.0
cfg.early_stop = True;  cfg.tol_h = 1e-5;  cfg.tol_rhs = 1e-4
cfg.min_iter  = 5;  cfg.consecutive_passes_required = 3

penalty = lp.PolynomialInventoryPenalty(
    lp.CarryCost(risk_aversion=10.0, sigma=SIGMA),
    lp.PolynomialInternalizationTime(tau0=4.0, tau1=0.070, tau2=0.0084),
)

tier1 = lp.MDPTier(
    name="Tier 1",
    sizes=[1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
    flow_curve=lp.LogisticFlowCurve(
        A0=0.0155, theta=0.144, beta=0.0857,
        shift=0.52, steepness=8.42, volume_shift=0.026),
    markout_model=lp.SqrtTimeMarkoutModel(
        a0=-0.1785/10_000, a1=-0.1819/10_000, a2=0.0052/10_000,
        b0= 0.0089/10_000, b1= 0.0068/10_000, b2=-0.0030/10_000),
    delta_min=-100.0, delta_max=100.0, use_markout=True,
)

solver = lp.HJBLadderSolver(config=cfg, penalty=penalty, mdp_tiers=[tier1])
sol    = solver.solve()
print(f"  converged={sol.diagnostics.converged}  iters={sol.diagnostics.iterations_used}")

# Dark pool venue (app.py defaults)
dp_venue = lp.DarkPoolVenue(
    lambda_bid=2.0, lambda_ask=2.0,
    p_bid=0.5,      p_ask=0.5,
    fee_per_unit_bid=0.0, fee_per_unit_ask=0.0,
    posted_sizes=[1.0, 2.0, 3.0, 5.0],
    allow_both_sides=False,
    min_fill_value=0.0,
)
dp_venue.reset_policy_shape(Q)
solver.build_dark_pool_policy(dp_venue, list(sol.h))
print("  dark pool policy built")


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig 1: The spread ──────────────────────────────────────────────────────────
def fig_spread():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10);  ax.set_ylim(0, 10)
    ax.axis("off")

    mid = 5.0
    spread_px = 1.4   # visual half-spread
    bid = mid - spread_px
    ask = mid + spread_px

    # colour bands
    ax.axhspan(0, bid,  alpha=0.18, color=BLUE_MPL, zorder=0)
    ax.axhspan(ask, 10, alpha=0.18, color=RED_MPL,  zorder=0)

    # price lines
    ax.axhline(mid,  color=GREY_H,   lw=1.5, linestyle="--", alpha=0.6)
    ax.axhline(bid,  color=BLUE_MPL, lw=3.0)
    ax.axhline(ask,  color=RED_MPL,  lw=3.0)

    ax.text(9.6, mid + 0.25, "Mid  11.500", color=GREY_H,   ha="right", va="bottom", fontsize=13)
    ax.text(9.6, bid + 0.25, "Bid  11.499", color=BLUE_MPL, ha="right", va="bottom", fontsize=14, fontweight="bold")
    ax.text(9.6, ask + 0.25, "Ask  11.501", color=RED_MPL,  ha="right", va="bottom", fontsize=14, fontweight="bold")

    # double-headed spread arrow
    ax.annotate("", xy=(7.5, ask), xytext=(7.5, bid),
                arrowprops=dict(arrowstyle="<->", color=GOLD_H, lw=2.2))
    ax.text(7.7, mid, "Spread\n0.002 SEK\n= 2 pips", color=GOLD_H,
            va="center", fontsize=12, fontweight="bold")

    # client sell arrow (client → us, bid side)
    ax.annotate("", xy=(4.3, bid), xytext=(2.5, bid),
                arrowprops=dict(arrowstyle="->", color=BLUE_MPL, lw=2.5))
    ax.text(0.2, bid - 0.7,
            "Client sells EUR\nWe pay 11.499\nWe receive inventory (+EUR)",
            color=BLUE_MPL, va="top", fontsize=11.5)

    # client buy arrow (us → client, ask side)
    ax.annotate("", xy=(2.5, ask), xytext=(4.3, ask),
                arrowprops=dict(arrowstyle="->", color=RED_MPL, lw=2.5))
    ax.text(0.2, ask + 0.15,
            "Client buys EUR\nWe receive 11.501\nWe lose inventory (−EUR)",
            color=RED_MPL, va="bottom", fontsize=11.5)

    # profit label
    ax.text(5.0, 0.5,
            "Each trade earns us ½ spread = 0.001 SEK per EUR traded  ≈  1,000 SEK per 1M EUR",
            color=GOLD_H, ha="center", fontsize=12, fontweight="bold")

    fig.tight_layout(pad=0.3)
    return to_stream(fig)


# ── Fig 2: Why inventory hurts ─────────────────────────────────────────────────
def fig_inventory_risk():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.4)

    # Left: inventory path under random spot moves
    ax = axes[0]
    ax.set_facecolor(CARD)
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    spot = 11.50 + 0.015 * np.cumsum(np.random.randn(200)) * (t[1] - t[0]) ** 0.5
    q = 5.0  # holding 5 lots long

    pnl = q * (spot - spot[0])
    ax.fill_between(t, pnl, 0, where=pnl < 0, color=RED_MPL,   alpha=0.45, label="Unrealised loss")
    ax.fill_between(t, pnl, 0, where=pnl >= 0, color=TEAL_H,   alpha=0.25, label="Unrealised gain")
    ax.plot(t, pnl, color=WHITE_H, lw=1.8)
    ax.axhline(0, color=GREY_H, lw=1, alpha=0.5, linestyle="--")
    style_ax(ax, xlabel="Time holding inventory  [min]",
             ylabel="Mark-to-market P&L  [SEK / lot]",
             title="Holding 5 lots long: random P&L walk")
    ax.legend(facecolor=CARD, edgecolor=GREY_H, labelcolor=WHITE_H, fontsize=10)

    # Right: how cost grows with inventory size
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    tau0, tau1, tau2 = 4.0, 0.070, 0.0084
    gamma, sigma = 10.0, SIGMA
    q_vals = np.linspace(0, 20, 300)
    t_vals = tau0 + tau1 * q_vals + tau2 * q_vals ** 2
    pi     = gamma * sigma ** 2 * q_vals ** 2 * t_vals

    ax2.plot(q_vals, pi * 1e6, color=GOLD_H, lw=2.8)
    ax2.fill_between(q_vals, pi * 1e6, alpha=0.15, color=GOLD_H)

    # annotate: 10 lots costs 4× more than 5 lots
    y5  = gamma * sigma**2 * 5**2  * (tau0 + tau1*5  + tau2*25)  * 1e6
    y10 = gamma * sigma**2 * 10**2 * (tau0 + tau1*10 + tau2*100) * 1e6
    ax2.annotate(f"{y5:.1f}", xy=(5, y5), xytext=(6.5, y5 * 1.3),
                 arrowprops=dict(arrowstyle="->", color=GREY_H, lw=1.2),
                 color=GREY_H, fontsize=11)
    ax2.annotate(f"{y10:.1f}", xy=(10, y10), xytext=(11.5, y10 * 0.85),
                 arrowprops=dict(arrowstyle="->", color=GREY_H, lw=1.2),
                 color=GREY_H, fontsize=11)
    ax2.text(14, y10 * 0.6, f"10 lots\ncosts {y10/y5:.1f}×\nmore than 5",
             color=GOLD_H, fontsize=11, fontweight="bold")

    style_ax(ax2, xlabel="Inventory held  [EUR lots]",
             ylabel="Expected total cost  [μSEK / min]",
             title="Cost grows faster than inventory size")
    return to_stream(fig)


# ── Fig 3: How clients decide ──────────────────────────────────────────────────
def fig_flow():
    A0, theta, beta_f = 0.0155, 0.144, 0.0857
    shift, steepness, vol_shift = 0.52, 8.42, 0.026

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.38)

    delta = np.linspace(0.0, 1.0, 300)
    sizes    = [1, 3, 10, 20]
    cols     = ["#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]

    for z, col in zip(sizes, cols):
        A  = A0 * z ** (-theta - beta_f * z)
        hr = 1 / (1 + np.exp(-steepness * (delta - shift + vol_shift * (z - 1))))
        axes[0].plot(delta, hr * 100,   color=col, lw=2.2, label=f"{z} lots")
        axes[1].plot(delta, A * hr * 60, color=col, lw=2.2, label=f"{z} lots")

    for ax in axes:
        ax.axvline(0.5, color=GREY_H, lw=1.0, alpha=0.4, linestyle=":")
        ax.legend(facecolor=CARD, edgecolor=GREY_H, labelcolor=WHITE_H, fontsize=10)

    axes[0].text(0.52, 52, "Quote at mid →", color=GREY_H, fontsize=10, alpha=0.7)
    style_ax(axes[0], xlabel="How aggressively we quote  (δ)",
             ylabel="Chance client trades with us  [%]",
             title="More aggressive = higher fill rate")
    style_ax(axes[1], xlabel="How aggressively we quote  (δ)",
             ylabel="Expected fills per hour",
             title="More aggressive = more trades per hour")

    axes[0].set_xlim(0, 1);  axes[1].set_xlim(0, 1)
    return to_stream(fig)


# ── Fig 4: The optimal policy ──────────────────────────────────────────────────
def fig_policy():
    sizes    = [1.0, 3.0, 10.0, 20.0]
    bid_cols = ["#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]
    ask_cols = ["#EF9A9A", "#EF5350", "#E53935", "#B71C1C"]

    tier = sol.mdp_tiers[0]
    q_show = [q for q in Q if -15 <= q <= 15]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.38)

    for z, bc, ac in zip(sizes, bid_cols, ask_cols):
        bd = [tier.quote(q, z, "bid") for q in q_show]
        ad = [tier.quote(q, z, "ask") for q in q_show]
        # convert δ to pips distance from mid: (0.5 − δ) × spread × 10000
        bid_pips = [(0.5 - d) * SPREAD * 10_000 for d in bd]
        ask_pips = [(0.5 - d) * SPREAD * 10_000 for d in ad]
        axes[0].plot(q_show, bid_pips, color=bc, lw=2.2, label=f"{int(z)} lots")
        axes[1].plot(q_show, ask_pips, color=ac, lw=2.2, label=f"{int(z)} lots")

    for ax, side, col in [(axes[0], "BID", BLUE_MPL), (axes[1], "ASK", RED_MPL)]:
        ax.axvline(0, color=GREY_H, lw=1.0, alpha=0.4, linestyle=":")
        ax.axhline(0, color=GREY_H, lw=1.0, alpha=0.4, linestyle=":")
        ax.legend(facecolor=CARD, edgecolor=GREY_H, labelcolor=WHITE_H, fontsize=10)
        ax.set_title(f"{side} quote: distance from mid  [pips]",
                     color=col, fontsize=13, fontweight="bold", pad=8)
        style_ax(ax, xlabel="Inventory  (positive = long EUR)")
        ax.tick_params(colors=GREY_H)

    # Annotation: when long, bid goes wider (more pips from mid)
    axes[0].annotate("Long EUR:\nbid widens →\ndiscourage\nmore buying",
                     xy=(10, axes[0].get_lines()[1].get_ydata()[
                         q_show.index(min(q_show, key=lambda q: abs(q-10)))]),
                     xytext=(6, 3.5),
                     arrowprops=dict(arrowstyle="->", color=GOLD_H, lw=1.4),
                     color=GOLD_H, fontsize=10, fontweight="bold")
    axes[1].annotate("Long EUR:\nask tightens →\nencourage\nclient buying",
                     xy=(10, axes[1].get_lines()[1].get_ydata()[
                         q_show.index(min(q_show, key=lambda q: abs(q-10)))]),
                     xytext=(4, -2.5),
                     arrowprops=dict(arrowstyle="->", color=GOLD_H, lw=1.4),
                     color=GOLD_H, fontsize=10, fontweight="bold")

    return to_stream(fig)


# ── Fig 5: Value function ──────────────────────────────────────────────────────
def fig_value():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.4)

    h_arr = np.array(sol.h)
    q_arr = np.array(Q)

    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.plot(q_arr, h_arr * 1e4, color=TEAL_H, lw=2.8)
    ax.fill_between(q_arr, h_arr * 1e4, alpha=0.15, color=TEAL_H)
    ax.axvline(0, color=GREY_H, lw=1, alpha=0.4, linestyle=":")
    ax.axhline(0, color=GREY_H, lw=1, alpha=0.4, linestyle=":")
    style_ax(ax, xlabel="Inventory  [EUR lots]",
             ylabel="Value  [×10⁻⁴ SEK]",
             title="h(q): the long-run value of each inventory level")
    ax.text(0.5, 0.05, "Zero inventory is the\nbest place to be",
            transform=ax.transAxes, color=GOLD_H, fontsize=11,
            ha="center", va="bottom", fontweight="bold")

    # Right: spread income per fill for different sizes
    tier = sol.mdp_tiers[0]
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    sizes    = [1.0, 3.0, 10.0, 20.0]
    bid_cols = ["#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]
    ask_cols = ["#EF9A9A", "#EF5350", "#E53935", "#B71C1C"]
    q_show = [q for q in Q if -15 <= q <= 15]

    for z, bc, ac in zip(sizes, bid_cols, ask_cols):
        bd = [tier.quote(q, z, "bid") for q in q_show]
        ad = [tier.quote(q, z, "ask") for q in q_show]
        bid_income = [z * SPREAD * (0.5 - d) * 10_000 for d in bd]
        ask_income = [z * SPREAD * (0.5 - d) * 10_000 for d in ad]
        ax2.plot(q_show, bid_income, color=bc, lw=1.8, label=f"bid z={int(z)}")
        ax2.plot(q_show, ask_income, color=ac, lw=1.8, linestyle="--",
                 label=f"ask z={int(z)}")

    style_ax(ax2, xlabel="Inventory  [EUR lots]",
             ylabel="Spread income per fill  [pips × lots]",
             title="Spread captured per trade — varies with inventory")
    ax2.legend(facecolor=CARD, edgecolor=GREY_H, labelcolor=WHITE_H,
               fontsize=9, ncol=2)

    return to_stream(fig)


# ── Fig 6: Internalization time ────────────────────────────────────────────────
def fig_internaliz():
    tau0, tau1, tau2 = 4.0, 0.070, 0.0084
    q_pos = np.linspace(0, 20, 300)
    t_q   = tau0 + tau1 * q_pos + tau2 * q_pos ** 2

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.plot(q_pos, t_q, color=TEAL_H, lw=2.8)
    ax.fill_between(q_pos, t_q, alpha=0.15, color=TEAL_H)
    ax.axhline(tau0, color=GOLD_H, lw=1.2, linestyle="--", alpha=0.7)
    ax.text(0.5, tau0 + 0.15, f"Minimum: τ₀ = {tau0} min", color=GOLD_H, fontsize=11)

    for q_mark in [5, 10, 15]:
        t_mark = tau0 + tau1 * q_mark + tau2 * q_mark**2
        ax.plot(q_mark, t_mark, "o", color=WHITE_H, ms=6, zorder=5)
        ax.text(q_mark + 0.4, t_mark + 0.1, f"{t_mark:.1f} min", color=GREY_H, fontsize=10)

    style_ax(ax, xlabel="Inventory size  [EUR lots]",
             ylabel="Expected minutes to reach zero",
             title="How long does it take to work out of inventory?")
    ax.set_xlim(0, 20)
    fig.tight_layout(pad=0.4)
    return to_stream(fig)


# ── Fig 7: Implied hit ratios ──────────────────────────────────────────────────
def fig_hit_ratios():
    tier     = sol.mdp_tiers[0]
    sizes    = [1.0, 3.0, 10.0, 20.0]
    bid_cols = ["#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]
    ask_cols = ["#EF9A9A", "#EF5350", "#E53935", "#B71C1C"]
    q_show   = [q for q in Q if -15 <= q <= 15]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.38)

    for z, bc, ac in zip(sizes, bid_cols, ask_cols):
        bid_hr = [tier.hit_ratio(tier.quote(q, z, "bid"), z) * 100 for q in q_show]
        ask_hr = [tier.hit_ratio(tier.quote(q, z, "ask"), z) * 100 for q in q_show]
        axes[0].plot(q_show, bid_hr, color=bc, lw=2.2, label=f"{int(z)} lots")
        axes[1].plot(q_show, ask_hr, color=ac, lw=2.2, label=f"{int(z)} lots")

    for ax, side_col in [(axes[0], BLUE_MPL), (axes[1], RED_MPL)]:
        ax.axvline(0, color=GREY_H, lw=1.0, alpha=0.4, linestyle=":")
        ax.set_ylim(0, 100)
        ax.legend(facecolor=CARD, edgecolor=GREY_H, labelcolor=WHITE_H, fontsize=10)
        style_ax(ax, xlabel="Inventory  [EUR lots]",
                 ylabel="Client fill probability  [%]")

    axes[0].set_title("BID: how often clients sell to us", color=BLUE_MPL,
                      fontsize=13, fontweight="bold", pad=8)
    axes[1].set_title("ASK: how often clients buy from us", color=RED_MPL,
                      fontsize=13, fontweight="bold", pad=8)

    # annotations
    axes[0].annotate("Long EUR: we widen\nbid → fewer clients\nsell to us",
                     xy=(10, axes[0].get_lines()[1].get_ydata()[
                         q_show.index(min(q_show, key=lambda q: abs(q-10)))]),
                     xytext=(5, 65),
                     arrowprops=dict(arrowstyle="->", color=GOLD_H, lw=1.3),
                     color=GOLD_H, fontsize=10, fontweight="bold")
    axes[1].annotate("Long EUR: we tighten\nask → more clients\nbuy from us",
                     xy=(10, axes[1].get_lines()[1].get_ydata()[
                         q_show.index(min(q_show, key=lambda q: abs(q-10)))]),
                     xytext=(3, 25),
                     arrowprops=dict(arrowstyle="->", color=GOLD_H, lw=1.3),
                     color=GOLD_H, fontsize=10, fontweight="bold")

    return to_stream(fig)


# ── Fig 8: Dark pool posting strategy ─────────────────────────────────────────
def fig_dark_pool():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.42)

    # Left: posted sizes across inventory
    ax = axes[0]
    ax.set_facecolor(CARD)

    bid_sizes, ask_sizes = [], []
    for i, q in enumerate(Q):
        b_sz = dp_venue.policy.bid_size[i] if dp_venue.policy.bid_active[i] else 0.0
        a_sz = dp_venue.policy.ask_size[i] if dp_venue.policy.ask_active[i] else 0.0
        bid_sizes.append(b_sz)
        ask_sizes.append(a_sz)

    q_arr = np.array(Q)
    ax.fill_between(q_arr, bid_sizes, step="mid", alpha=0.5,
                    color=BLUE_MPL, label="Bid posting (buying)", where=np.array(bid_sizes) > 0)
    ax.fill_between(q_arr, ask_sizes, step="mid", alpha=0.5,
                    color=RED_MPL,  label="Ask posting (selling)", where=np.array(ask_sizes) > 0)
    ax.step(q_arr, bid_sizes, color=BLUE_MPL, lw=1.8, where="mid")
    ax.step(q_arr, ask_sizes, color=RED_MPL,  lw=1.8, where="mid")
    ax.axvline(0, color=GREY_H, lw=1.0, alpha=0.4, linestyle=":")
    ax.set_xlim(q_arr[0], q_arr[-1])
    ax.set_ylim(bottom=0)
    style_ax(ax, xlabel="Inventory  [EUR lots]",
             ylabel="Size posted in dark pool  [lots]",
             title="What size we leave in the dark pool")
    ax.legend(facecolor=CARD, edgecolor=GREY_H, labelcolor=WHITE_H, fontsize=10)
    ax.text(-15, max(max(bid_sizes), 0.1) * 0.8,
            "Short:\npost to buy back", color=BLUE_MPL, fontsize=10,
            fontweight="bold", ha="center")
    ax.text(15, max(max(ask_sizes), 0.1) * 0.8,
            "Long:\npost to sell", color=RED_MPL, fontsize=10,
            fontweight="bold", ha="center")

    # Right: geometric fill distribution for posting size 5, p=0.5
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    p, u = 0.5, 5
    ks = np.arange(1, u + 1)
    probs = np.array([p * (1 - p) ** (k - 1) for k in range(1, u)])
    probs = np.append(probs, (1 - p) ** (u - 1))   # truncated at u
    bars = ax2.bar(ks, probs * 100, color=[TEAL_H] * (u - 1) + [GOLD_H],
                   edgecolor=CARD, width=0.7)
    ax2.bar_label(bars, fmt="%.1f%%", color=GREY_H, fontsize=10, padding=3)
    style_ax(ax2, xlabel="Lots actually filled",
             ylabel="Probability  [%]",
             title="Fill distribution when posting 5 lots  (p = 0.5)")
    ax2.text(u, probs[-1] * 100 * 0.5,
             "Full 5-lot fill\n(residual prob)",
             color=GOLD_H, fontsize=10, fontweight="bold", ha="center")
    ax2.set_xticks(ks)
    ax2.set_xticklabels([f"{k}" for k in ks], color=GREY_H)

    return to_stream(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Build slides
# ══════════════════════════════════════════════════════════════════════════════

print("Rendering charts…")
s_spread      = fig_spread()
s_inv_risk    = fig_inventory_risk()
s_flow        = fig_flow()
s_policy      = fig_policy()
s_hit_ratios  = fig_hit_ratios()
s_dark_pool   = fig_dark_pool()
s_value       = fig_value()
s_internaliz  = fig_internaliz()

def S(fn, *a, **kw):
    sl = prs.slides.add_slide(BLANK)
    fn(sl, *a, **kw)
    return sl


# ── 1. Title ──────────────────────────────────────────────────────────────────
S(title_card,
  "How should we quote EUR/SEK?",
  "A model for FX market making")

# ── 2. What we do ─────────────────────────────────────────────────────────────
S(narrative, "We are a price-maker", [
    ("Our role", "heading"),
    ("Clients call us wanting to buy or sell euros.", "body"),
    ("We quote two prices — a bid (we buy) and an ask (we sell).", "body"),
    ("We make money on the gap between them, called the spread.", "body"),
    ("The challenge", "heading"),
    ("Every trade changes what we hold — our inventory.", "body"),
    ("Too much inventory exposes us to losses if the market moves.", "body"),
    ("The model tells us how to adjust our quotes to stay in control.", "body"),
], callout=("2 pips", "spread on a\n1M EUR trade\n= 2,000 SEK"))

# ── 3. The spread diagram ─────────────────────────────────────────────────────
S(caption_slide,
  "Every trade earns us the spread — but also changes our position",
  s_spread,
  "EUR/SEK mid ≈ 11.50  |  1 pip = 0.0001 SEK/EUR  |  on 1M EUR, 2 pips = 2,000 SEK profit")

# ── 4. Why inventory is the problem ──────────────────────────────────────────
S(narrative, "Inventory is the core risk", [
    ("What happens when we trade?", "heading"),
    ("Each trade moves our position. If we buy 5M EUR, we are now 'long'.", "body"),
    ("The euro might weaken while we hold it — that's a loss.", "body"),
    ("The key insight", "heading"),
    ("We can't control when clients call us.", "body"),
    ("But we can control how attractive our prices are.", "body"),
    ("Wider bid → fewer clients sell to us → we don't get longer.", "body"),
    ("Tighter ask → more clients buy from us → we reduce our long position.", "body"),
    ("This is the lever the model optimises.", "heading"),
])

# ── 5. Cost of holding inventory ─────────────────────────────────────────────
S(caption_slide,
  "The longer and larger the position, the more it costs",
  s_inv_risk,
  "Left: one realisation of the P&L from holding 5 lots.  Right: expected total cost — grows faster than the position size.")

# ── 6. How long until we are flat? ───────────────────────────────────────────
S(caption_slide,
  "Bigger positions take longer to work off",
  s_internaliz,
  "Estimated from historical data: how many minutes on average until inventory returns to zero, starting from a given position.")

# ── 7. How clients decide ─────────────────────────────────────────────────────
S(narrative, "Clients respond to our prices", [
    ("The fill rate", "heading"),
    ("If we quote tight (aggressive), more clients trade with us.", "body"),
    ("If we quote wide (conservative), fewer clients trade.", "body"),
    ("This is the fundamental tension: margin per trade vs. number of trades.", "body"),
    ("Size matters", "heading"),
    ("Larger trades are harder to fill — fewer counterparties exist.", "body"),
    ("Larger trades also move our inventory more, so they deserve wider quotes.", "body"),
])

# ── 8. Flow curves ────────────────────────────────────────────────────────────
S(caption_slide,
  "How fill probability and trade frequency depend on our quote",
  s_flow,
  "Left: probability a client trades with us.  Right: expected number of trades per hour.  δ = 0.5 means quoting exactly at mid.")

# ── 9. What the model solves ─────────────────────────────────────────────────
S(narrative, "One question: what is the best quote right now?", [
    ("The answer depends on inventory", "heading"),
    ("When we hold a lot of long EUR, we should discourage more buying.", "body"),
    ("We widen our bid (we pay less, so fewer clients sell to us).", "detail"),
    ("We tighten our ask (we charge less, so more clients buy from us).", "detail"),
    ("When we are short EUR, the reverse applies.", "body"),
    ("When flat, we quote symmetrically to maximise flow income.", "body"),
    ("The model finds the exact optimal quote for every inventory level", "heading"),
    ("…and for every trade size simultaneously.", "body"),
    ("It accounts for the cost of holding, the expected flow, and how prices move.", "body"),
])

# ── 10. The solved policy ─────────────────────────────────────────────────────
S(caption_slide,
  "Solved: how far from mid we quote at each inventory level",
  s_policy,
  "Positive = quoting further from mid (wider, more conservative).  The model automatically widens the side that would worsen the inventory.")

# ── 11. Implied hit ratios: narrative ────────────────────────────────────────
S(narrative, "The quote determines how often clients actually trade", [
    ("What is the implied hit ratio?", "heading"),
    ("Once the model picks a quote, we can ask: how likely is it a client trades?", "body"),
    ("This is the implied hit ratio — a direct consequence of the chosen delta.", "body"),
    ("What it reveals", "heading"),
    ("When long EUR, we widen the bid. Fewer clients sell to us — the bid hit rate falls.", "body"),
    ("At the same time, we tighten the ask. More clients buy from us — ask hit rate rises.", "body"),
    ("The model is actively steering flow to reduce inventory.", "body"),
    ("Larger sizes always have lower hit rates", "heading"),
    ("Bigger trades are riskier and harder to fill — the model quotes them wider.", "body"),
])

# ── 12. Implied hit ratios: figure ───────────────────────────────────────────
S(caption_slide,
  "Implied hit ratios: how often clients trade at each inventory level",
  s_hit_ratios,
  "The model steers flow — when long, it raises ask hit rate and lowers bid hit rate simultaneously.")

# ── 13. Dark pool: narrative ──────────────────────────────────────────────────
S(narrative, "The dark pool: a passive, anonymous venue", [
    ("What is a dark pool?", "heading"),
    ("Some clients route orders through anonymous electronic venues.", "body"),
    ("We can leave a passive order there — if a client crosses our price, we fill.", "body"),
    ("No spread cost — we trade at mid. We just need someone to be on the other side.", "body"),
    ("How we use it", "heading"),
    ("When long EUR, we post a sell order. If a client buys, we reduce our position for free.", "body"),
    ("When short EUR, we post a buy order to try to cover.", "body"),
    ("The model decides the optimal posting size at each inventory level.", "body"),
    ("The catch", "heading"),
    ("Fills are random — we post a size but might get less.", "body"),
    ("The larger the post, the less likely a full fill.", "detail"),
])

# ── 14. Dark pool: figure ─────────────────────────────────────────────────────
S(caption_slide,
  "Dark pool posting strategy and fill distribution",
  s_dark_pool,
  "Left: size posted at each inventory level — buy when short, sell when long.  Right: geometric fill distribution when posting 5 lots.")

# ── 15. The value of being flat ───────────────────────────────────────────────
S(caption_slide,
  "The model also tells us what each inventory level is worth",
  s_value,
  "Left: h(q) — the long-run value advantage of being flat vs holding inventory.  Right: how much spread income we capture per fill.")

# ── 12. Multiple sizes ────────────────────────────────────────────────────────
S(narrative, "In practice, clients trade many different sizes", [
    ("The ladder", "heading"),
    ("We quote six sizes simultaneously: 1, 2, 3, 5, 10 and 20 million EUR.", "body"),
    ("Each size has its own quote, its own fill curve, its own risk.", "body"),
    ("Consistency rules", "heading"),
    ("A 5M trade must never get a tighter quote than a 1M trade.", "body"),
    ("The model enforces this automatically — no manual intervention needed.", "body"),
    ("As inventory shifts, all six rungs update together, coherently.", "detail"),
], callout=("6", "trade sizes\nquoted\nsimultaneously"))

# ── 13. How we solve it ───────────────────────────────────────────────────────
S(narrative, "The algorithm: iterate until the quotes stabilise", [
    ("We cannot solve this analytically", "heading"),
    ("Start with flat quotes everywhere.", "body"),
    ("Compute the best response to those quotes — update the whole ladder.", "body"),
    ("Repeat. After ~100 iterations the quotes stop changing.", "body"),
    ("That is the optimal policy.", "body"),
    ("Convergence is fast", "heading"),
    ("The solver runs in C++ and completes in under a second.", "body"),
    ("It is re-run whenever market conditions change.", "body"),
], callout=("< 1s", "solve time\non a laptop"))

# ── 14. Where the numbers come from ──────────────────────────────────────────
S(narrative, "Every parameter comes from data", [
    ("Fill curves", "heading"),
    ("Fit to our RFQ history: how often clients traded at different spread levels.", "body"),
    ("Internalization time", "heading"),
    ("How long until inventory returns to zero — measured directly from trading records.", "body"),
    ("Volatility", "heading"),
    ("One-minute EUR/SEK spot volatility, estimated from price data.", "body"),
    ("Adverse selection (markout)", "heading"),
    ("How much the price moves after a client trade — a direct data regression.", "body"),
])

# ── 15. End ───────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
bg(sl)
rect(sl, Inches(0), Inches(0), W, Inches(0.06), GOLD)
rect(sl, Inches(0), H - Inches(0.06), W, Inches(0.06), GOLD)
text(sl, "Questions?", Inches(0), Inches(2.9), W, Inches(1.2),
     size=52, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
text(sl, "Model: stationary HJB  ·  C++23 solver  ·  pybind11  ·  Streamlit UI",
     Inches(0), Inches(4.1), W, Inches(0.7),
     size=19, color=LGREY, align=PP_ALIGN.CENTER)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = "hjb_ladder_pricer.pptx"
prs.save(OUT)
print(f"Saved → {OUT}  ({prs.slides.__len__()} slides)")
