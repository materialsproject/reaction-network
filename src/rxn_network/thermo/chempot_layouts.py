default_chempot_layout_3d = dict(
    width=800,
    height=800,
    hovermode="closest",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=True,
    legend=dict(
        orientation="v",
        x=0.1,
        y=0.99,
        traceorder="reversed",
        xanchor="left",
        yanchor="top",
    ),
    scene_camera=dict(projection=dict(type="orthographic")),
    scene=dict(),
)

default_chempot_annotation_layout = {
    "align": "center",
    "opacity": 0.7,
    "showarrow": False,
    "xanchor": "center",
    "yanchor": "auto",
}
