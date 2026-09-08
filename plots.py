"""
plots.py — all figure construction for the Joukowski Airfoil & Wing Designer UI.

Split out of app.py so the figures can be built and inspected without starting
Streamlit.

Two plotting back-ends, on purpose:

  * matplotlib  — every 2-D figure, plus one static 3-D view. Vector-clean and
                  exportable as PNG for reports.
  * Plotly      — the interactive 3-D views. Rotatable, and the axis labels
                  cannot collide the way the matplotlib 3-D labels did.

Both back-ends are driven from the same THEME dict, so the look stays
consistent with the app's CSS.

Nothing in this module calls Streamlit, and nothing here computes aerodynamics
except the clearly-labelled Schrenk spanwise-loading approximation, which is a
classical planform estimate and NOT a solver output.
"""

from __future__ import annotations

import io

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # safe on headless Streamlit Cloud
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go

from joukowski_designer_v2 import (
    generate_joukowski_airfoil,
    reorder_airfoil_from_trailing_edge,
    upper_lower_split,
    build_wing_surface,
    approx_cl_max,
)

# =========================================================================
# THEME — single source of truth for both back-ends
# =========================================================================
THEME = {
    "ink":        "#1A2332",   # darkest text
    "ink_soft":   "#4A5A70",   # axis labels, ticks
    "title":      "#1F3A5F",
    "grid":       "#DDE3EC",
    "axis":       "#B0BCC9",
    "panel":      "#FAFBFD",   # plot interior
    "paper":      "#FFFFFF",   # figure background
    "primary":    "#2E86AB",   # airfoil / 2-D
    "primary_dk": "#1F5F82",
    "accent":     "#E07A5F",   # aircraft / 3-D
    "green":      "#5F8A5F",   # wing
    "grey":       "#5B6B7F",   # design-point marker
    "red":        "#C0392B",   # limits
    "font":       "DejaVu Sans",
}

# Diverging Cp scale that reads correctly on a white page:
# suction (negative Cp) = blue, stagnation (positive Cp) = red.
CP_SCALE = [
    [0.00, "#1F5F82"],
    [0.25, "#2E86AB"],
    [0.50, "#EEF2F7"],
    [0.75, "#E07A5F"],
    [1.00, "#C0392B"],
]

FIGSIZE_WIDE = (9.0, 3.6)
FIGSIZE_STD = (8.6, 4.3)


# =========================================================================
# matplotlib helpers
# =========================================================================
def _style_axes(ax, title=None, xlabel=None, ylabel=None, legend=False):
    ax.set_facecolor(THEME["panel"])
    ax.grid(True, color=THEME["grid"], alpha=0.9, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(THEME["axis"])
    if title:
        ax.set_title(title, color=THEME["title"], fontsize=11, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=THEME["ink_soft"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=THEME["ink_soft"], fontsize=10)
    ax.tick_params(colors=THEME["ink_soft"], labelsize=9)
    if legend:
        leg = ax.legend(frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(THEME["ink"])


def _new_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(THEME["paper"])
    return fig, ax


def fig_to_png(fig, dpi=170, tight=True):
    """
    Serialize a matplotlib figure to PNG bytes (for the download ZIP).

    tight=False for 3-D figures: matplotlib's tight bounding box does not
    reliably account for 3-D axis labels and clips them.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight" if tight else None)
    buf.seek(0)
    return buf.getvalue()


# =========================================================================
# 2-D FIGURES — airfoil geometry
# =========================================================================
def fig_airfoil_shape(best, chord_dim):
    x_n, z_n, _, _ = generate_joukowski_airfoil(
        best["a"], best["x_c"], best["y_c"], n_points=1400)
    fig, ax = _new_fig(FIGSIZE_WIDE)
    ax.fill(x_n * chord_dim, z_n * chord_dim,
            color=THEME["primary"], alpha=0.13, zorder=2)
    ax.plot(x_n * chord_dim, z_n * chord_dim,
            color=THEME["primary"], linewidth=2, zorder=3)
    ax.axhline(0.0, color=THEME["axis"], linewidth=0.6, zorder=1)
    ax.set_aspect("equal")
    _style_axes(
        ax,
        title=(f"Optimized Joukowski airfoil — chord {chord_dim:.3f} m, "
               f"t/c {100 * best['tc_actual']:.2f}%, "
               f"camber {100 * best['camber']:.2f}%"),
        xlabel="x [m]", ylabel="z [m]")
    fig.tight_layout()
    return fig


def fig_thickness_camber(best):
    x_n, z_n, _, _ = generate_joukowski_airfoil(
        best["a"], best["x_c"], best["y_c"], n_points=1400)
    x_grid, _, _, thick, camb = upper_lower_split(x_n, z_n, n_grid=400)
    fig, ax = _new_fig(FIGSIZE_STD)
    ax.plot(x_grid, thick, color=THEME["primary"], linewidth=2,
            label="Thickness t(x)/c")
    ax.plot(x_grid, camb, color=THEME["accent"], linewidth=2,
            label="Camber z_c(x)/c")
    i_t, i_c = int(np.argmax(thick)), int(np.argmax(camb))
    ax.plot(x_grid[i_t], thick[i_t], "o", ms=5, color=THEME["primary_dk"])
    ax.annotate(f"max t/c {thick[i_t]:.4f} at x/c {x_grid[i_t]:.3f}",
                (x_grid[i_t], thick[i_t]), textcoords="offset points",
                xytext=(8, 8), fontsize=8.5, color=THEME["ink_soft"])
    ax.plot(x_grid[i_c], camb[i_c], "o", ms=5, color=THEME["accent"])
    ax.annotate(f"max camber {camb[i_c]:.4f} at x/c {x_grid[i_c]:.3f}",
                (x_grid[i_c], camb[i_c]), textcoords="offset points",
                xytext=(8, -14), fontsize=8.5, color=THEME["ink_soft"])
    ax.axhline(0.0, color=THEME["axis"], linewidth=0.6)
    _style_axes(ax, title="Chordwise thickness and camber distribution",
                xlabel="x / c", ylabel="t/c and camber/c", legend=True)
    fig.tight_layout()
    return fig


def fig_cp(x_ord, cp_ord, alpha_sec, mach):
    fig, ax = _new_fig(FIGSIZE_STD)
    x_ord = np.asarray(x_ord)
    cp_ord = np.asarray(cp_ord)
    ax.plot(x_ord, cp_ord, color=THEME["primary"], linewidth=1.5)
    ax.axhline(0.0, color=THEME["axis"], linewidth=0.6)
    # sonic line, only where it is physically meaningful
    cp_crit = _cp_critical(mach)
    if cp_crit is not None and cp_ord.min() < 0.4 * cp_crit:
        ax.axhline(cp_crit, linestyle=":", color=THEME["red"], linewidth=1.2,
                   label=f"Cp* sonic limit ({cp_crit:.2f})")
    ax.invert_yaxis()
    _style_axes(
        ax,
        title=(f"Pressure coefficient at section α = {alpha_sec:.2f}°, "
               f"M = {mach:.2f}"),
        xlabel="x / c", ylabel="Cp",
        legend=(cp_crit is not None and cp_ord.min() < 0.4 * cp_crit))
    fig.tight_layout()
    return fig


def _cp_critical(mach, gamma=1.4):
    """Critical pressure coefficient (Cp at which local flow reaches M=1)."""
    if mach <= 0.05 or mach >= 1.0:
        return None
    term = (1 + 0.5 * (gamma - 1) * mach ** 2) / (1 + 0.5 * (gamma - 1))
    return float((2 / (gamma * mach ** 2)) * (term ** (gamma / (gamma - 1)) - 1))


# =========================================================================
# 2-D FIGURES — polars
# =========================================================================
def _polar(ax, x, series, design_alpha, xlabel, ylabel, title,
           hline=None, hline_label=None):
    for y, lab, c in series:
        ax.plot(x, y, color=c, linewidth=2, label=lab)
    ax.axvline(design_alpha, linestyle="--", color=THEME["grey"],
               linewidth=1.2, label=f"Design α = {design_alpha:.2f}°")
    if hline is not None:
        ax.axhline(hline, linestyle=":", color=THEME["red"],
                   linewidth=1.2, label=hline_label)
    _style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel, legend=True)


def fig_cl_polar(sweep, best, design_alpha):
    fig, ax = _new_fig(FIGSIZE_STD)
    series = [(sweep["cl_airfoil"], "Cl (airfoil, 2-D)", THEME["primary"])]
    if "CL_aircraft" in sweep:
        series.append((sweep["CL_aircraft"], "CL (3-D aircraft)", THEME["accent"]))
    cl_max = approx_cl_max(best["Re"])
    _polar(ax, sweep["alpha_aircraft"], series, design_alpha,
           "Aircraft angle of attack [deg]", "Lift coefficient",
           f"Lift curve — M {best['max_mach']:.2f}, Re {best['Re']:.2e}",
           hline=cl_max, hline_label=f"Approx Cl_max ({cl_max:.2f})")
    fig.tight_layout()
    return fig


def fig_cd_polar(sweep, best, design_alpha):
    fig, ax = _new_fig(FIGSIZE_STD)
    series = [(sweep["cd_airfoil"], "Cd (airfoil)", THEME["primary"])]
    if "CD_wing" in sweep:
        series.append((sweep["CD_wing"], "CD (wing)", THEME["green"]))
    if "CD_aircraft" in sweep:
        series.append((sweep["CD_aircraft"], "CD (aircraft)", THEME["accent"]))
    _polar(ax, sweep["alpha_aircraft"], series, design_alpha,
           "Aircraft angle of attack [deg]", "Drag coefficient",
           "Drag build-up")
    fig.tight_layout()
    return fig


def fig_ld_polar(sweep, best, design_alpha):
    fig, ax = _new_fig(FIGSIZE_STD)
    series = [(sweep["ld_airfoil"], "L/D (airfoil)", THEME["primary"])]
    if "LD_wing" in sweep:
        series.append((sweep["LD_wing"], "L/D (wing)", THEME["green"]))
    if "LD_aircraft" in sweep:
        series.append((sweep["LD_aircraft"], "L/D (aircraft)", THEME["accent"]))
    _polar(ax, sweep["alpha_aircraft"], series, design_alpha,
           "Aircraft angle of attack [deg]", "L / D",
           "Aerodynamic efficiency")
    # mark the peak of the most complete curve available
    y, lab, _ = series[-1]
    i = int(np.nanargmax(y))
    ax.plot(sweep["alpha_aircraft"][i], y[i], "o", ms=5, color=THEME["ink"])
    ax.annotate(f"peak {y[i]:.1f} at α {sweep['alpha_aircraft'][i]:.2f}°",
                (sweep["alpha_aircraft"][i], y[i]), textcoords="offset points",
                xytext=(8, 6), fontsize=8.5, color=THEME["ink_soft"])
    fig.tight_layout()
    return fig


def fig_cm_polar(sweep, best, design_alpha):
    fig, ax = _new_fig(FIGSIZE_STD)
    series = [(sweep["Cm_airfoil"], "Cm airfoil (c/4)", THEME["primary"])]
    if "Cm_aircraft" in sweep:
        series.append((sweep["Cm_aircraft"], "Cm aircraft (CG)", THEME["accent"]))
    _polar(ax, sweep["alpha_aircraft"], series, design_alpha,
           "Aircraft angle of attack [deg]", "Cm", "Pitching moment")
    ax.axhline(0.0, color=THEME["axis"], linewidth=0.6)
    fig.tight_layout()
    return fig


def fig_drag_polar(sweep, design_alpha):
    """CL against CD — the classical drag polar, previously not plotted."""
    fig, ax = _new_fig(FIGSIZE_STD)
    if "CL_aircraft" in sweep:
        ax.plot(sweep["CD_aircraft"], sweep["CL_aircraft"],
                color=THEME["accent"], linewidth=2, label="Aircraft")
    if "CD_wing" in sweep:
        ax.plot(sweep["CD_wing"], sweep["CL_wing"],
                color=THEME["green"], linewidth=2, label="Wing")
    ax.plot(sweep["cd_airfoil"], sweep["cl_airfoil"],
            color=THEME["primary"], linewidth=2, label="Airfoil (2-D)")
    i = int(np.argmin(np.abs(sweep["alpha_aircraft"] - design_alpha)))
    cd_k = "CD_aircraft" if "CD_aircraft" in sweep else "cd_airfoil"
    cl_k = "CL_aircraft" if "CL_aircraft" in sweep else "cl_airfoil"
    ax.plot(sweep[cd_k][i], sweep[cl_k][i], "o", ms=6, color=THEME["ink"],
            label="Design point")
    _style_axes(ax, title="Drag polar", xlabel="CD", ylabel="CL", legend=True)
    fig.tight_layout()
    return fig


# =========================================================================
# SPANWISE LOADING — Schrenk approximation (NOT a solver output)
# =========================================================================
def schrenk_loading(wing, CL, n=201):
    """
    Classical Schrenk approximation: the additional-lift distribution is the
    average of the planform chord distribution and an ellipse of equal area.

    Exactly area-normalized, so integrating cl(y)·c(y) over the span returns
    CL·S. Ignores sweep, twist, and compressibility, and assumes an unflapped,
    untwisted trapezoidal wing — treat it as a first-cut tip-stall check only.
    """
    b = float(wing["span"])
    S = float(wing["area"])
    y = np.linspace(-b / 2, b / 2, n)
    eta = np.abs(y) / (b / 2)
    c_local = wing["root_chord"] * (1 - (1 - wing["taper_ratio"]) * eta)
    c_ell = (4 * S / (np.pi * b)) * np.sqrt(np.clip(1 - eta ** 2, 0, None))
    c_eff = 0.5 * (c_local + c_ell)
    cl_local = CL * c_eff / np.maximum(c_local, 1e-9)
    return y, c_local, c_ell, cl_local


def fig_spanwise_loading(wing, CL, Re):
    y, c_local, c_ell, cl_local = schrenk_loading(wing, CL)
    cl_max = approx_cl_max(Re)
    fig, ax = _new_fig(FIGSIZE_STD)
    ax.plot(y, cl_local, color=THEME["primary"], linewidth=2,
            label="Local section cl (Schrenk)")
    ax.axhline(CL, linestyle="--", color=THEME["grey"], linewidth=1.2,
               label=f"Wing CL = {CL:.3f}")
    ax.axhline(cl_max, linestyle=":", color=THEME["red"], linewidth=1.2,
               label=f"Approx Cl_max ({cl_max:.2f})")
    i = int(np.argmax(cl_local))
    ax.plot(y[i], cl_local[i], "o", ms=5, color=THEME["ink"])
    ax.annotate(f"first to stall: y {y[i]:+.2f} m, cl {cl_local[i]:.3f}",
                (y[i], cl_local[i]), textcoords="offset points",
                xytext=(8, 8), fontsize=8.5, color=THEME["ink_soft"])
    _style_axes(ax, title="Spanwise lift distribution — Schrenk approximation",
                xlabel="y [m] (spanwise)", ylabel="local section cl",
                legend=True)
    fig.tight_layout()
    return fig


# =========================================================================
# 3-D GEOMETRY HELPERS
# =========================================================================
def _span_axis(Y):
    """
    Which array axis of the surface mesh runs spanwise?

    build_wing_surface may return (n_airfoil, n_span) or the transpose, so
    detect it instead of assuming: the spanwise axis is the one along which
    the spanwise coordinate Y actually varies.
    """
    v0 = float(np.mean(np.ptp(Y, axis=0)))
    v1 = float(np.mean(np.ptp(Y, axis=1)))
    return 0 if v0 >= v1 else 1


def _aspect(X, Y, Z, z_exagg=1.0, z_floor=0.16):
    """
    Normalized scene box aspect, plus the vertical stretch actually applied.

    The original 3-D plot passed the raw data ranges to set_box_aspect. With
    x ~ 2 m, y ~ 10 m and z ~ 0.3 m that collapses the z axis to ~3% of the
    box height, which is why the z tick labels and the z axis title piled on
    top of each other.

    The fix stretches the *box*, never the data, so tick values stay in true
    metres and only the drawing is scaled. The returned factor is reported in
    the axis title so the stretch is never silent.
    """
    xr, yr = float(np.ptp(X)), float(np.ptp(Y))
    zr = float(np.ptp(Z))
    m = max(xr, yr, zr, 1e-12)
    z_true = max(zr / m, 1e-9)
    # The multiplier applies on top of the readable baseline, so every scale
    # option changes the drawing. Applying it before the floor made ×2 and ×3
    # identical to auto for a typical wing, since the floor dominated both.
    z_frac = min(max(z_true, z_floor) * float(z_exagg), 1.0)
    return (xr / m, yr / m, z_frac), z_frac / z_true


def _resample(values, n):
    """Resample a closed contour array to n points by normalized index."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == n:
        return v
    s_old = np.linspace(0.0, 1.0, v.size)
    s_new = np.linspace(0.0, 1.0, n)
    return np.interp(s_new, s_old, v)


def _scene(title_x="x [m] (chordwise)", z_factor=1.0, aspect=(1, 1, 0.2)):
    """Common Plotly 3-D scene styling with generous label spacing."""
    # Nested title.font — the old `titlefont` shorthand was removed in Plotly 6.
    title_font = dict(size=12, color=THEME["ink"])
    ax_common = dict(
        backgroundcolor=THEME["paper"],
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["axis"],
        showbackground=True,
        color=THEME["ink_soft"],
        tickfont=dict(size=10, color=THEME["ink_soft"]),
        nticks=6,
    )
    z_title = ("z [m]" if z_factor < 1.05
               else f"z [m] — true metres, drawn ×{z_factor:.1f}")
    return dict(
        xaxis=dict(title=dict(text=title_x, font=title_font), **ax_common),
        yaxis=dict(title=dict(text="y [m] (spanwise)", font=title_font),
                   **ax_common),
        zaxis=dict(title=dict(text=z_title, font=title_font), nticks=5,
                   **{k: v for k, v in ax_common.items() if k != "nticks"}),
        aspectmode="manual",
        aspectratio=dict(x=aspect[0], y=aspect[1], z=aspect[2]),
        camera=dict(eye=dict(x=1.5, y=-1.7, z=0.9)),
    )


def _layout(fig, title, subtitle=None, height=560, z_factor=1.0,
            aspect=(1, 1, 0.2), title_x="x [m] (chordwise)"):
    head = title if subtitle is None else f"{title}<br><sub>{subtitle}</sub>"
    fig.update_layout(
        title=dict(text=head, font=dict(size=15, color=THEME["title"]),
                   x=0.01, xanchor="left", y=0.97, yanchor="top"),
        scene=_scene(title_x=title_x, z_factor=z_factor, aspect=aspect),
        paper_bgcolor=THEME["paper"],
        plot_bgcolor=THEME["paper"],
        font=dict(family="Inter, Segoe UI, sans-serif",
                  color=THEME["ink"], size=12),
        height=height,
        # Room on all sides so no axis title can be clipped or overlapped.
        margin=dict(l=10, r=10, t=74, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",
                    bordercolor=THEME["grid"], borderwidth=1,
                    font=dict(size=11), x=0.01, y=0.01,
                    xanchor="left", yanchor="bottom"),
        hoverlabel=dict(bgcolor=THEME["paper"], font_size=12,
                        bordercolor=THEME["grid"]),
    )
    return fig


def _wing_head(wing):
    return (f"span {wing['span']:.2f} m · AR {wing['aspect_ratio']:.2f} · "
            f"taper {wing['taper_ratio']:.2f} · LE sweep "
            f"{wing['sweep_le_deg']:.1f}° · dihedral {wing['dihedral_deg']:.1f}°")


def _edge_lines(X, Y, Z, s_ax):
    """Leading edge, trailing edge and quarter-chord lines from the mesh."""
    n_span = X.shape[s_ax]
    le, te, qc = [], [], []
    for j in range(n_span):
        sl = (slice(None), j) if s_ax == 1 else (j, slice(None))
        xs, ys, zs = X[sl], Y[sl], Z[sl]
        i0, i1 = int(np.argmin(xs)), int(np.argmax(xs))
        p0 = np.array([xs[i0], ys[i0], zs[i0]])
        p1 = np.array([xs[i1], ys[i1], zs[i1]])
        le.append(p0)
        te.append(p1)
        qc.append(p0 + 0.25 * (p1 - p0))
    return np.array(le), np.array(te), np.array(qc)


# =========================================================================
# 3-D FIGURES (Plotly, interactive)
# =========================================================================
def fig3d_wing_surface(best, wing, n_span=41, n_airfoil=160, z_exagg=1.0,
                       z_floor=0.16, show_edges=True, height=580):
    X, Y, Z, _ = build_wing_surface(best, wing, n_span=n_span, n_airfoil=n_airfoil)
    aspect, z_factor = _aspect(X, Y, Z, z_exagg, z_floor)
    s_ax = _span_axis(Y)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, THEME["primary"]], [1, THEME["primary"]]],
        showscale=False, opacity=0.97,
        lighting=dict(ambient=0.62, diffuse=0.78, specular=0.16, roughness=0.62),
        lightposition=dict(x=1000, y=-800, z=1200),
        contours=dict(y=dict(show=True, color=THEME["primary_dk"],
                             width=1, start=-wing["span"] / 2,
                             end=wing["span"] / 2,
                             size=max(wing["span"] / 16, 1e-3))),
        hovertemplate="x %{x:.3f} m<br>y %{y:.3f} m<br>z %{z:.4f} m<extra></extra>",
        name="Wing surface",
    ))
    if show_edges:
        le, te, qc = _edge_lines(X, Y, Z, s_ax)
        for pts, nm, col, dash in (
                (le, "Leading edge", THEME["ink"], "solid"),
                (te, "Trailing edge", THEME["ink_soft"], "solid"),
                (qc, "Quarter-chord line", THEME["accent"], "dash")):
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines", name=nm,
                line=dict(color=col, width=4, dash=dash),
                hovertemplate=nm + "<br>y %{y:.3f} m<extra></extra>"))
    return _layout(fig, "Wing surface", _wing_head(wing), height=height,
                   z_factor=z_factor, aspect=aspect)


def fig3d_wing_ribs(best, wing, n_ribs=11, n_airfoil=200, z_exagg=1.0,
                    z_floor=0.16, height=560):
    """Spanwise section stack — the view you want before lofting in CAD."""
    n_span = max(n_ribs, 3)
    X, Y, Z, _ = build_wing_surface(best, wing, n_span=n_span, n_airfoil=n_airfoil)
    aspect, z_factor = _aspect(X, Y, Z, z_exagg, z_floor)
    s_ax = _span_axis(Y)
    fig = go.Figure()
    for j in range(X.shape[s_ax]):
        sl = (slice(None), j) if s_ax == 1 else (j, slice(None))
        y_j = float(np.mean(Y[sl]))
        edge = j in (0, X.shape[s_ax] - 1)
        fig.add_trace(go.Scatter3d(
            x=X[sl], y=Y[sl], z=Z[sl], mode="lines",
            line=dict(color=THEME["primary_dk"] if edge else THEME["primary"],
                      width=5 if edge else 2.5),
            opacity=1.0 if edge else 0.75,
            name=f"y = {y_j:+.2f} m", showlegend=edge,
            hovertemplate=(f"section y {y_j:+.2f} m<br>"
                           "x %{x:.3f} m<br>z %{z:.4f} m<extra></extra>")))
    le, te, qc = _edge_lines(X, Y, Z, s_ax)
    for pts, nm, col, dash in (
            (le, "Leading edge", THEME["ink"], "solid"),
            (te, "Trailing edge", THEME["ink_soft"], "solid"),
            (qc, "Quarter-chord line", THEME["accent"], "dash")):
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
            line=dict(color=col, width=4, dash=dash), name=nm))
    return _layout(fig, f"Section ribs ({X.shape[s_ax]} stations)",
                   _wing_head(wing), height=height, z_factor=z_factor,
                   aspect=aspect)


def fig3d_wing_cp(best, wing, cp_contour, n_span=41, n_airfoil=160,
                  z_exagg=1.0, z_floor=0.16, alpha_sec=None, height=580):
    """
    Wing surface coloured by the 2-D section Cp.

    Strip visualization: the same section Cp is painted at every spanwise
    station. There is no spanwise pressure variation, sweep relief, or tip
    effect in this colouring — it shows the chordwise pressure field wrapped
    onto the 3-D shape, not a 3-D pressure solution.
    """
    X, Y, Z, _ = build_wing_surface(best, wing, n_span=n_span, n_airfoil=n_airfoil)
    aspect, z_factor = _aspect(X, Y, Z, z_exagg, z_floor)
    s_ax = _span_axis(Y)
    n_chord = X.shape[1 - s_ax]
    cp_line = _resample(cp_contour, n_chord)
    C = (np.tile(cp_line[:, None], (1, X.shape[1])) if s_ax == 1
         else np.tile(cp_line[None, :], (X.shape[0], 1)))
    lim = float(np.nanmax(np.abs(cp_line)))
    lim = max(lim, 0.5)
    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Z, surfacecolor=C,
        colorscale=CP_SCALE, cmin=-lim, cmax=lim, reversescale=False,
        colorbar=dict(title=dict(text="Cp", side="right"),
                      tickfont=dict(size=10, color=THEME["ink_soft"]),
                      outlinecolor=THEME["grid"], outlinewidth=1,
                      len=0.62, thickness=14, x=1.0),
        lighting=dict(ambient=0.78, diffuse=0.5, specular=0.06),
        hovertemplate=("x %{x:.3f} m<br>y %{y:.3f} m<br>"
                       "Cp %{surfacecolor:.3f}<extra></extra>"),
    ))
    sub = _wing_head(wing)
    if alpha_sec is not None:
        sub = f"section α {alpha_sec:.2f}° · " + sub
    fig = _layout(fig, "Surface pressure (2-D strip approximation)", sub,
                  height=height, z_factor=z_factor, aspect=aspect)
    fig.update_layout(margin=dict(l=10, r=70, t=74, b=10))
    return fig


def fig3d_spanwise_loading(wing, CL, height=540):
    """Planform in plan view with the Schrenk load curve standing above it."""
    y, c_local, c_ell, cl_local = schrenk_loading(wing, CL, n=121)
    eta = np.abs(y) / (wing["span"] / 2)
    x_le = eta * (wing["span"] / 2) * np.tan(np.radians(wing["sweep_le_deg"]))
    x_te = x_le + c_local
    load = cl_local * c_local                      # local lift per unit span / q
    load_s = load / max(load.max(), 1e-9)
    h = 0.55 * float(np.ptp(np.concatenate([x_le, x_te])) or 1.0)

    fig = go.Figure()
    # planform outline, flat at z = 0
    fig.add_trace(go.Scatter3d(
        x=np.concatenate([x_le, x_te[::-1], x_le[:1]]),
        y=np.concatenate([y, y[::-1], y[:1]]),
        z=np.zeros(2 * y.size + 1), mode="lines",
        line=dict(color=THEME["ink_soft"], width=3), name="Planform"))
    # load curve
    fig.add_trace(go.Scatter3d(
        x=x_le + 0.25 * c_local, y=y, z=load_s * h, mode="lines",
        line=dict(color=THEME["primary"], width=6),
        name="Schrenk load l(y)",
        customdata=np.column_stack([cl_local, c_local]),
        hovertemplate=("y %{y:.3f} m<br>local cl %{customdata[0]:.3f}"
                       "<br>chord %{customdata[1]:.3f} m<extra></extra>")))
    # vertical strips
    for k in range(0, y.size, 4):
        fig.add_trace(go.Scatter3d(
            x=[x_le[k] + 0.25 * c_local[k]] * 2, y=[y[k]] * 2,
            z=[0, load_s[k] * h], mode="lines",
            line=dict(color=THEME["primary"], width=2), opacity=0.45,
            showlegend=False, hoverinfo="skip"))
    fig = _layout(fig, "Spanwise load distribution (Schrenk)",
                  f"wing CL {CL:.3f} · height is normalized load, not metres",
                  height=height, aspect=(0.45, 1.0, 0.32))
    fig.update_layout(scene_zaxis=dict(
        title=dict(text="normalized load"), showticklabels=False,
        backgroundcolor=THEME["paper"], gridcolor=THEME["grid"],
        showbackground=True, color=THEME["ink_soft"]))
    return fig


# =========================================================================
# STATIC 3-D (matplotlib) — for report figures and PNG export
# =========================================================================
def fig_wing_3d_static(best, wing, n_span=25, n_airfoil=160, z_exagg=1.0,
                       z_floor=0.22, elev=22, azim=-58):
    """
    Static 3-D view with the axis-label collision fixed.

    Fixes relative to the original: normalized box aspect with a minimum z
    height, tick counts capped, explicit label padding, an offset z-axis
    label, a two-line title, and manual margins instead of tight_layout
    (which mis-handles 3-D axes and was pushing the labels inward).
    """
    X, Y, Z, _ = build_wing_surface(best, wing, n_span=n_span, n_airfoil=n_airfoil)
    aspect, z_factor = _aspect(X, Y, Z, z_exagg, z_floor)

    # A taller box needs a taller canvas and less zoom, otherwise the
    # chordwise axis label is pushed off the bottom edge at high stretch.
    z_frac = aspect[2]
    fig_h = 4.6 + 3.6 * z_frac
    zoom = 1.35 - 0.35 * min(z_frac / 0.5, 1.0)

    fig = plt.figure(figsize=(10.6, fig_h))
    fig.patch.set_facecolor(THEME["paper"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(THEME["paper"])
    ax.view_init(elev=elev, azim=azim)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=max(n_span // 12, 1),
                    linewidth=0.25, antialiased=True,
                    color=THEME["primary"], alpha=0.9,
                    edgecolor=THEME["primary_dk"])

    # zoom fills the canvas: a high-aspect wing otherwise leaves most of the
    # 3-D bounding box empty.
    try:
        ax.set_box_aspect(aspect, zoom=zoom)
    except TypeError:                       # matplotlib < 3.6
        ax.set_box_aspect(aspect)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(THEME["panel"])
        axis.pane.set_edgecolor(THEME["grid"])
        axis.pane.set_alpha(0.6)
        axis._axinfo["grid"]["color"] = THEME["grid"]
    # Tick counts scaled to how much screen length each axis actually gets,
    # so the short chordwise axis does not end up with five crowded labels.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # At true scale a 0.3 m thickness beside a 10 m span leaves no room for
    # several z labels - that geometry is what collided in the first place -
    # so the tick count drops with the available height, and below the point
    # where even two fit, the range moves into the axis title instead.
    z_ticks_hidden = z_frac < 0.07
    if z_ticks_hidden:
        ax.set_zticks([float(np.min(Z)), float(np.max(Z))])
        ax.set_zticklabels([])
    elif z_frac < 0.13:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=2))
    else:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.set_xlabel("x [m] (chordwise)", color=THEME["ink_soft"],
                  fontsize=10, labelpad=12)
    ax.set_ylabel("y [m] (spanwise)", color=THEME["ink_soft"],
                  fontsize=10, labelpad=20)
    if z_ticks_hidden:
        z_lbl = (f"z [m] {float(np.min(Z)):.3f} to {float(np.max(Z)):.3f}, "
                 "true scale")
    elif z_factor < 1.05:
        z_lbl = "z [m]"
    else:
        z_lbl = f"z [m] — true metres, drawn ×{z_factor:.1f}"
    ax.set_zlabel(z_lbl, color=THEME["ink_soft"], fontsize=10, labelpad=10)
    ax.tick_params(axis="x", colors=THEME["ink_soft"], labelsize=8.5, pad=3)
    ax.tick_params(axis="y", colors=THEME["ink_soft"], labelsize=8.5, pad=3)
    ax.tick_params(axis="z", colors=THEME["ink_soft"], labelsize=8.5, pad=6)

    fig.suptitle("3-D wing", color=THEME["title"], fontsize=13,
                 x=0.045, ha="left", y=1.0 - 0.13 / fig_h, va="top")
    fig.text(0.045, 1.0 - 0.42 / fig_h, _wing_head(wing),
             color=THEME["ink_soft"], fontsize=9.5, ha="left", va="top")

    # Manual margins: tight_layout is unreliable for 3-D axes.
    top = 1.0 - 0.62 / fig_h          # constant absolute header height
    fig.subplots_adjust(left=0.00, right=0.95, bottom=0.13, top=top)
    return fig
