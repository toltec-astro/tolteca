"""Shared appearance for KIDs diagnostic figures."""

from html import escape

from tollan.plot.plotly import ColorPalette

FIG_LAYOUT = {
    "xaxis": {
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "linecolor": "black",
        "linewidth": 1,
        "ticks": "outside",
    },
    "yaxis": {
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "linecolor": "black",
        "linewidth": 1,
        "ticks": "outside",
    },
    "plot_bgcolor": "white",
    "autosize": True,
    "margin": {"autoexpand": True, "l": 0, "b": 0, "t": 20},
    "modebar": {"orientation": "v"},
}

COLOR_PALETTE = ColorPalette()


def html_document(title, body, head=""):
    """Wrap scientific artifact content in the shared diagnostic page style."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title>{head}"
        "<style>body{font:16px system-ui,sans-serif;color:#243746;margin:24px}"
        "a{color:#17648f}nav{margin-bottom:8px}"
        ".channels{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));"
        "gap:8px;padding-left:24px}details{margin:16px 0}"
        "table{border-collapse:collapse;width:100%;font-size:14px}"
        "th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #ddd}"
        "th{background:#f5f7f8}label,button,select,input{margin:4px;font:inherit}"
        ".scroll{overflow:auto}caption{text-align:left;margin:8px 0}"
        "</style>"
        f"</head><body>{body}</body></html>"
    )
