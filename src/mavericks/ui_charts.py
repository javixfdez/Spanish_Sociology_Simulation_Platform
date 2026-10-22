"""Gráficos plotly de la interfaz (sin lógica de negocio)."""

import plotly.graph_objects as go

PALETTE = ["#581C87", "#A100FF", "#7C3AED", "#C084FC", "#D8B4FE", "#F59E0B", "#3B0764", "#059669"]
INK = "#1E1B4B"
CORE_BAND = "rgba(88, 28, 135, 0.45)"  # banda gruesa del «rango habitual»: morado oscuro translúcido, se lee sobre la barra y sobre el fondo


def _layout(fig, height, title=None):
    fig.update_layout(
        height=height,
        title=title,
        margin={"l": 8, "r": 8, "t": 36 if title else 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"size": 14, "color": "#1E1B4B"},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def distribution_bars(distribution, interval=None, title=None, core=None):
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
            hovertemplate="%{y}: %{x:.1f} %<extra></extra>",
        )
    )
    if core:
        fig.add_trace(
            go.Scatter(
                x=values, y=options, mode="markers", marker={"size": 1, "color": "rgba(0,0,0,0)"}, hoverinfo="skip",
                showlegend=False,
                error_x={
                    "type": "data", "symmetric": False, "color": CORE_BAND, "thickness": 9, "width": 0,
                    "array": [(core[o][1] - distribution[o]) * 100 for o in options],
                    "arrayminus": [(distribution[o] - core[o][0]) * 100 for o in options],
                },
            )
        )
    # la etiqueta va a la derecha de lo que más se extienda (barra, rango del 90 % o banda del rango habitual) para no pisarlos
    reach = []
    for o, v in zip(options, values, strict=True):
        ends = [v]
        if interval:
            ends.append(interval[o][1] * 100)
        if core:
            ends.append(core[o][1] * 100)
        reach.append(max(ends))
    fig.add_trace(
        go.Scatter(
            x=reach, y=options, mode="text", text=[f"{v:.1f} %" for v in values], textposition="middle right",
            hoverinfo="skip", showlegend=False, cliponaxis=False, textfont={"color": INK, "size": 14},
        )
    )
    top = max(reach) if reach else 0
    fig.update_xaxes(range=[0, 100 if not values else min(100, top * 1.2 + 8)], title="% de la población adulta")
    fig = _layout(fig, 90 + 52 * len(options), title)
    fig.update_layout(showlegend=False)   # quita «trace 0»
    return fig


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
            colorscale=[[0, "#F3E8FF"], [1, "#A100FF"]],
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
