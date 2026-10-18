"""Gráficos plotly de la interfaz (sin lógica de negocio)."""

import plotly.graph_objects as go

PALETTE = ["#1f4e79", "#2e86c1", "#85c1e9", "#f5b041", "#e67e22", "#a93226", "#7d3c98", "#117a65"]
INK = "#1b2631"


def _layout(fig, height, title=None):
    fig.update_layout(
        height=height,
        title=title,
        margin={"l": 8, "r": 8, "t": 36 if title else 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"size": 14},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def distribution_bars(distribution, interval=None, title=None):
    """Barras horizontales con el intervalo del 90 % como barra de error (valores en %)."""
    options = list(distribution)[::-1]
    values = [distribution[o] * 100 for o in options]
    error = None
    if interval:
        error = {
            "type": "data",
            "symmetric": False,
            "array": [(interval[o][1] - distribution[o]) * 100 for o in options],
            "arrayminus": [(distribution[o] - interval[o][0]) * 100 for o in options],
            "color": INK,
            "thickness": 1.6,
            "width": 5,
        }
    fig = go.Figure(
        go.Bar(
            x=values,
            y=options,
            orientation="h",
            marker_color=PALETTE[1],
            error_x=error,
            text=[f"{v:.1f} %" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f} %<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, max(100, 1.0) if not values else min(100, max(values) * 1.35 + 8)], title="% de la población adulta")
    return _layout(fig, 90 + 52 * len(options), title)


def stacked_groups(groups, options, title=None):
    """Una barra apilada al 100 % por grupo (edad, sexo…)."""
    names = list(groups)
    fig = go.Figure()
    for i, option in enumerate(options):
        fig.add_bar(
            name=option,
            y=names,
            x=[groups[g]["distribution"][option] * 100 for g in names],
            orientation="h",
            marker_color=PALETTE[i % len(PALETTE)],
            hovertemplate="%{y} · " + option + ": %{x:.1f} %<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 100], title="%")
    return _layout(fig, 120 + 34 * len(names), title)


def heatmap(groups, options, title=None):
    names = list(groups)
    z = [[groups[g]["distribution"][o] * 100 for o in options] for g in names]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=options,
            y=names,
            colorscale="Blues",
            text=[[f"{v:.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="%{y} · %{x}: %{z:.1f} %<extra></extra>",
            colorbar={"title": "%"},
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 120 + 30 * len(names), title)


def grouped_bars(frame, x, ys, title=None, ytitle="MAE"):
    fig = go.Figure()
    for i, y in enumerate(ys):
        fig.add_bar(name=y, x=frame[x], y=frame[y], marker_color=PALETTE[i % len(PALETTE)])
    fig.update_yaxes(title=ytitle)
    return _layout(fig, 360, title)


def scatter_pred_obs(predicted, observed, labels, title=None):
    fig = go.Figure()
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line={"dash": "dot", "color": "#999"}, name="perfecto")
    fig.add_scatter(
        x=observed, y=predicted, mode="markers", text=labels, name="opciones",
        marker={"color": PALETTE[1], "size": 6, "opacity": 0.55},
        hovertemplate="%{text}<br>observado %{x:.1f} %<br>predicho %{y:.1f} %<extra></extra>",
    )
    fig.update_xaxes(title="Observado (CIS) %", range=[0, 100])
    fig.update_yaxes(title="Predicho %", range=[0, 100])
    return _layout(fig, 420, title)
