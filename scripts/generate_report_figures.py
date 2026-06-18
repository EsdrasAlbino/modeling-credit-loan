#!/usr/bin/env python3
"""Generate publication-quality EDA figures for the IEEE report."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "imagens"

# IEEE-friendly style: serif fonts, white background, no heavy titles (caption lives in LaTeX)
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Single-column width for IEEEtran (~3.5 in)
FIG_W = 3.5
FIG_H = 2.6

COLORS = {
    "primary": "#2E5EAA",
    "secondary": "#D4A574",
    "neutral": "#5C6B73",
    "accent": "#8B4B6B",
    "line": "#C44E52",
}


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf")


def fig_distribuicao_inadimplencia() -> None:
    sizes = [68.7, 31.3]
    labels = ["Adimplentes", "Inadimplentes"]
    colors = [COLORS["secondary"], COLORS["primary"]]
    explode = (0, 0.03)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        pctdistance=0.55,
        labeldistance=1.12,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    _save(fig, "distribuicao_inadimplencia")


def fig_distribuicao_historico_comportamental() -> None:
    sizes = [22.1, 77.9]
    labels = ["Com histórico\ncomportamental", "Sem histórico\ncomportamental"]
    colors = [COLORS["primary"], COLORS["secondary"]]
    explode = (0.03, 0)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        pctdistance=0.55,
        labeldistance=1.15,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    _save(fig, "distribuicao_historico_comportamental")


def fig_taxa_inadimplencia_por_grupo() -> None:
    categories = [
        "Portfólio\ntotal",
        "Com dados\ncomportamentais",
        "Sem dados\ncomportamentais",
    ]
    rates = [31.3, 20.8, 33.9]
    colors = [COLORS["neutral"], COLORS["primary"], COLORS["secondary"]]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = range(len(categories))
    bars = ax.bar(x, rates, color=colors, width=0.62, edgecolor="white", linewidth=0.8)

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=7.5)
    ax.set_ylabel("Taxa de inadimplência (%)")
    ax.set_ylim(0, 42)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.axhline(31.3, color=COLORS["line"], linestyle="--", linewidth=1.0, label="Média geral (31,3%)")
    ax.legend(loc="upper right", frameon=False, handlelength=1.8)

    fig.subplots_adjust(bottom=0.22)
    _save(fig, "taxa_inadimplencia_por_grupo")


def main() -> None:
    fig_distribuicao_inadimplencia()
    fig_distribuicao_historico_comportamental()
    fig_taxa_inadimplencia_por_grupo()
    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
