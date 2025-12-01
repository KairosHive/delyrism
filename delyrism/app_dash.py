# pip install dash dash-bootstrap-components
from __future__ import annotations
import io, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL


# ----- bring your code in -----
from delyrism import (
    SymbolSpace, TextEmbedder,
    context_delta_graph, plot_delta_graph,
    plot_contextual_subgraph_colored, plot_ambiguity_metrics
)

# ---------- helpers ----------
def _default_symbols_map():
    return {
        "CLOUDS": ["spirit world","transcendence","wisdom","portal","spiritual connection"],
        "EARTH": ["ground","soil","people","life","physical world"],
        "WATER": ["river","emotion","flow","cleansing","change"],
        "DRAGONFLY": ["messenger","transformation","threshold","quick movement","reflection"],
        "HOUSE": ["home","shelter","family","boundary","gathering"],
    }

def _load_symbols_map(txt: str | None):
    if not txt: return _default_symbols_map()
    data = json.loads(txt); assert isinstance(data, dict)
    return {str(k): [str(x) for x in v] for k, v in data.items()}

def fig_to_datauri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def fig_from_callable(fn, *a, **k):
    fig = plt.figure(); plt.close(fig)
    fn(*a, **k)
    return plt.gcf()

def focus_to_tau(focus, tau_min=0.01, tau_max=0.2):
    return tau_max - focus * (tau_max - tau_min)

# ---------- app ----------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# light-blue headers for all accordions
app.index_string = app.index_string.replace(
    "</head>",
    """
    <style>
      .acc-hdr .accordion-button{
        background:#1E90FF !important; color:#fff !important;
        border:1px solid #4682B4 !important; border-radius:8px !important;
      }
      .acc-hdr .accordion-button:not(.collapsed){
        box-shadow:0 0 0 1px rgba(255,255,255,.08) inset, 0 2px 6px rgba(0,0,0,.18);
      }
    </style>
    </head>
    """
)

# ---- Sidebar controls (accordion sections) ----
symbols_default_txt = json.dumps(_default_symbols_map(), ensure_ascii=False, indent=2)

sidebar = dbc.Accordion([
    dbc.AccordionItem([
        dcc.Upload(
            id="upload-json",
            children=html.Div(["Drag and drop or ", html.A("browse JSON")]),
            style={"width":"100%","height":"80px","lineHeight":"80px",
                   "borderWidth":"1px","borderStyle":"dashed","borderRadius":"8px",
                   "textAlign":"center", "marginBottom":"8px"},
        ),
        dbc.Textarea(id="symbols-text", value=symbols_default_txt, rows=10),
        html.Div(id="load-status", className="mt-1 small")
    ], title="Data", className="acc-hdr"),

    dbc.AccordionItem([
        dbc.Select(id="backend", options=[{"label":x,"value":x} for x in ["qwen3","qwen2","original"]], value="qwen3"),
        dbc.Input(id="model", placeholder="HF model override (optional)", className="mt-2"),
        dbc.Select(id="pooling", options=[{"label":x,"value":x} for x in ["eos","mean","cls","last"]], value="eos", className="mt-2"),
    ], title="Embeddings", className="acc-hdr"),

    dbc.AccordionItem([
        dbc.Textarea(id="ctx-sentence", placeholder="A ceremony by the river…", rows=3),
        dcc.Dropdown(id="ctx-chosen", multi=True),
        html.Div(id="weights-container")  # per-symbol sliders will be inserted here
    ], title="Context", className="acc-hdr"),

    dbc.AccordionItem([
        dcc.Slider(id="tau", min=0.01, max=2.0, step=0.01, value=0.3),
        dcc.Slider(id="alpha", min=0.10, max=0.99, step=0.01, value=0.8),
        dcc.Slider(id="lam", min=0.0, max=1.0, step=0.01, value=0.6),
        dbc.Checklist(options=[{"label":"Use Personalized PageRank","value":"ppr"}],
                      value=["ppr"], id="use-ppr", switch=True, className="mt-2"),
    ], title="Ranking (proposal)", className="acc-hdr"),

    dbc.AccordionItem([
        dcc.Slider(id="beta", min=0.0, max=2.0, step=0.05, value=0.6),
        dbc.Select(id="gate", options=[{"label":x,"value":x} for x in ["relu","cos","softmax","uniform"]], value="relu"),
        dbc.Checklist(options=[{"label":"Softmax within symbol (if gate=softmax)","value":"wsoft"}],
                      value=["wsoft"], id="within-softmax", switch=True, className="mt-2"),
        dcc.Slider(id="memb-alpha", min=0.0, max=1.0, step=0.05, value=0.0),
    ], title="Shift settings", className="acc-hdr"),

    dbc.AccordionItem([
        dcc.Slider(id="ctx-topk-symbols", min=1, max=12, step=1, value=3),
        dcc.Slider(id="ctx-topk-desc", min=1, max=12, step=1, value=3),
        dbc.Select(id="ctx-method", options=[{"label":"ppr","value":"ppr"},{"label":"softmax","value":"softmax"}], value="ppr"),
        dcc.Slider(id="ctx-focus", min=0.0, max=1.0, step=0.01, value=0.6),
        dcc.Slider(id="ctx-alpha", min=0.50, max=0.99, step=0.01, value=0.85),
        dbc.Checklist(options=[{"label":"Normalize by baseline PR","value":"norm"}], value=["norm"], id="ctx-norm", switch=True),
        dcc.Slider(id="desc-thresh", min=0.0, max=0.9, step=0.02, value=0.7),
    ], title="Contextual Subgraph", className="acc-hdr"),

    dbc.AccordionItem([
        dbc.Checklist(options=[{"label":"Within-symbol pairs only","value":"within"}], value=["within"], id="within-symbol", switch=True),
        dcc.Dropdown(id="sym-filter", multi=True),
        dcc.Slider(id="top-abs-edges", min=2, max=100, step=1, value=10),
        dbc.Checklist(options=[{"label":"Connected nodes only","value":"conn"}], value=["conn"], id="connected-only", switch=True),
    ], title="Δ Graph", className="acc-hdr"),
], start_collapsed=False, flush=False, always_open=True)

# ---- Main area ----
main = dbc.Container([
    html.H3("Meaning Space (2D)"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id="reducer", options=[{"label":x,"value":x} for x in ["umap","tsne","pca"]], value="umap"),
            html.Img(id="fig-meaning", style={"width":"100%","marginTop":"8px"}),
            html.Hr(),
            html.H3("Ambiguity Metrics"),
            dcc.Dropdown(id="sort-opt", options=[{"label":x,"value":x} for x in ["dispersion","leakage","entropy","none"]], value="dispersion"),
            html.Img(id="fig-amb", style={"width":"100%","marginTop":"8px"}),
        ], width=6),
        dbc.Col([
            html.H3("Top Symbols for Context"),
            html.Img(id="fig-rank", style={"width":"100%"}),
            html.Hr(),
            html.H3("Descriptor Attention"),
            dcc.Dropdown(id="sym", multi=False),
            html.Img(id="fig-attn", style={"width":"100%","marginTop":"8px"}),
        ], width=6)
    ])
], fluid=True)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col(main, width=9)
    ])
], fluid=True)

# ---------- Callbacks ----------

# Build space when data/model controls change (very simplified)
@app.callback(
    Output("ctx-chosen","options"),
    Output("sym","options"),
    Output("sym-filter","options"),
    Output("load-status","children"),
    State("symbols-text","value"),
    Input("upload-json","contents"),
)
def load_symbols(txt, uploaded):
    if uploaded:
        # uploaded is a base64 data URL
        content = uploaded.split(",",1)[1]
        txt = base64.b64decode(content).decode("utf-8")
        msg = "Loaded JSON from file."
    else:
        msg = "Using text area JSON."
    try:
        symbols_map = _load_symbols_map(txt)
        opts = [{"label":s,"value":s} for s in symbols_map.keys()]
        # store in dcc.Store if you want; for brevity we just set dropdowns now
        app.server.symbols_map = symbols_map
        # create embedder/space once (tweak to include other controls)
        app.server.embedder = TextEmbedder(backend="qwen3", model=None, pooling="eos")
        app.server.space = SymbolSpace(symbols_to_descriptors=symbols_map, embedder=app.server.embedder, descriptor_threshold=0.7)
        return opts, opts, opts, msg
    except Exception as e:
        return [], [], [], f"Failed to parse JSON: {e}"

# Dynamic per-symbol sliders
@app.callback(Output("weights-container","children"), Input("ctx-chosen","value"))
def make_weight_sliders(selected):
    if not selected: return html.Div("Select context symbols above.")
    sliders = []
    for s in selected:
        sliders.append(html.Div([
            html.Div(f"Weight: {s}", className="small"),
            dcc.Slider(id={"type":"w", "sym":s}, min=0, max=1, step=0.05, value=0.5)
        ], style={"marginBottom":"8px"}))
    return sliders

# Render figures (meaning map + ambiguity) — simplified demo
@app.callback(
    Output("fig-meaning","src"),
    Output("fig-amb","src"),
    Input("reducer","value"),
    Input("ctx-chosen","value"),
    Input({"type":"w","sym":ALL}, "value"),  # requires: from dash.dependencies import ALL
    State({"type":"w","sym":ALL}, "id")
)
def update_main(reducer, chosen_syms, weights_vals, ids):
    space = getattr(app.server, "space", None)
    if space is None:
        return None, None
    weights = {}
    if chosen_syms and weights_vals and ids:
        for v, i in zip(weights_vals, ids):
            weights[i["sym"]] = v

    # Meaning map
    fig_ms = fig_from_callable(
        space.plot_map_shift, weights=weights or None, sentence=None,
        method=reducer, with_hulls=True, include_centroids=True,
        normalize_centroids=False, figsize=(6.8,5.4), title="Context shift"
    )
    img1 = fig_to_datauri(fig_ms)

    # Ambiguity
    color_map = getattr(space, "get_symbol_color_dict", lambda: None)()
    fig_amb = plot_ambiguity_metrics(space, sort_by="dispersion", color_dict=color_map, figsize=(7.5,4))
    img2 = fig_to_datauri(fig_amb)
    return img1, img2

if __name__ == "__main__":
    app.run(debug=True)
