"""Offline navigation for retained ModelFit Plotly artifacts."""

from html import escape

from plotly.offline import get_plotlyjs
from tollan.pipeline import ArtifactFile

from ._plot import html_document
from .model_fit_quality import ModelFitBitMask

_FILTER_SCRIPT = """
<script>
const search = document.getElementById('search');
const filter = document.getElementById('flag');
const rows = Array.from(document.querySelectorAll('#groups tbody tr'));
const channelRows = Array.from(document.querySelectorAll('#channels tbody tr'));
let page = 0;
function matches(row) {
  const mask = Number(row.dataset.flag);
  const choice = filter.value;
  return row.dataset.search.toLowerCase().includes(search.value.toLowerCase()) &&
    (choice === 'all' || (choice === 'flagged' ? mask !== 0 :
      choice === 'clear' ? mask === 0 : (mask & Number(choice)) !== 0));
}
function render() {
  const selected = rows.filter(matches);
  const pages = Math.max(1, Math.ceil(selected.length / 40));
  page = Math.max(0, Math.min(page, pages - 1));
  rows.forEach(row => row.hidden = true);
  selected.slice(page * 40, (page + 1) * 40).forEach(row => row.hidden = false);
  channelRows.forEach(row => row.hidden = !matches(row));
  document.getElementById('page').textContent =
    `${selected.length} groups · page ${page+1} / ${pages}`;
  document.getElementById('previous').disabled = page === 0;
  document.getElementById('next').disabled = page + 1 >= pages;
}
search.addEventListener('input', () => {page = 0; render();});
filter.addEventListener('change', () => {page = 0; render();});
document.getElementById('previous').onclick = () => {page--; render();};
document.getElementById('next').onclick = () => {page++; render();};
render();
</script>
"""


def _navigation(name, details):
    nav = '<a href="../model_fit.html">All groups and channels</a>'
    base = name.removesuffix("_correlation")
    if not base.startswith("group_"):
        return nav
    i = next(i for i, row in enumerate(details) if row["stem"] == base)
    for j, label in ((i - 1, "Previous group"), (i + 1, "Next group")):
        if 0 <= j < len(details):
            nav += f' · <a href="{details[j]["stem"]}.html">{label}</a>'
    nav += (
        f' · <a href="{base}.html">Fit panels</a>'
        f' · <a href="{base}_correlation.html">Parameter correlation</a>'
    )
    record = details[i]
    nav += (
        f"<p>Group warnings: {escape(record['flags'])}. "
        "The selector changes display coordinates only; "
        "it does not assign a primary detector.</p>"
    )
    return nav


def write_model_fit_browser(data, *, path):
    """Write summary-to-group navigation with a shared local Plotly library."""
    artifacts = {item.spec.alias: item for item in data.values() if item.spec.save}
    if not artifacts:
        return []
    if "html" not in artifacts["summary"].spec.save_formats:
        return [
            receipt
            for name, item in artifacts.items()
            for receipt in item.spec.write(item.data, path=path / name)
        ]
    directory = path / "model_fit"
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "plotly.min.js"
    library.write_text(get_plotlyjs())
    receipts = [ArtifactFile(library, "js")]
    metadata = artifacts["summary"].data.layout.meta
    details = [row for row in metadata["groups"] if row["stem"]]
    for name, item in artifacts.items():
        figure = item.data
        nav = _navigation(name, details)
        if label := metadata.get("acquisition"):
            nav = f"<p><strong>{escape(label)}</strong></p>" + nav
        target = directory / f"{name}.html"
        body = f"<nav>{nav}</nav>" + figure.to_html(
            full_html=False, include_plotlyjs=False, auto_play=False
        )
        target.write_text(
            html_document(
                f"Model fit · {name}", body, '<script src="plotly.min.js"></script>'
            )
        )
        receipts.append(ArtifactFile(target, "html", name=name))
        other_formats = [fmt for fmt in item.spec.save_formats if fmt != "html"]
        if other_formats:
            spec = item.spec.model_copy(update={"save_formats": other_formats})
            receipts.extend(spec.write(figure, path=directory / name))
    target = path / "model_fit.html"
    body = _index_body(metadata, details, "power" in artifacts)
    target.write_text(html_document("Model fit diagnostics", body + _FILTER_SCRIPT))
    return [ArtifactFile(target, "html", name="model_fit_browser"), *receipts]


def _index_body(metadata, details, has_power):
    options = "".join(
        f'<option value="{int(bit)}">{escape(bit.name)}</option>'
        for bit in ModelFitBitMask
    )
    group_rows = []
    for row in metadata["groups"]:
        group = escape(row["group"])
        label = (
            f'<a href="model_fit/{row["stem"]}.html">{group}</a>'
            if row["stem"]
            else group + " (details not selected)"
        )
        channels = escape(", ".join(str(c) for c in row["channels"]))
        group_rows.append(
            f'<tr data-flag="{row["flag"]}" data-search="{group} {channels}">'
            f"<td>{label}</td><td>{channels}</td>"
            f"<td>{row['residual']:.4g}</td><td>{row['condition']:.4g}</td>"
            f"<td>{escape(row['flags'])}</td></tr>"
        )
    channel_rows = []
    for row in metadata["channels"]:
        links = [
            f'<a href="model_fit/{g["stem"]}.html">{escape(g["group"])}</a>'
            for g in details
            if row["chan"] in g["channels"]
        ]
        channel_rows.append(
            f'<tr data-flag="{row["flag"]}" data-search="{escape(str(row["chan"]))}">'
            f"<td>{escape(str(row['chan']))}</td>"
            f"<td>{', '.join(links) or 'No detailed response'}</td>"
            f"<td>{escape(row['flags'])}</td></tr>"
        )
    body = "<h1>Model fit · sweep diagnostics</h1>"
    if label := metadata.get("acquisition"):
        body += f"<p><strong>{escape(label)}</strong></p>"
    body += '<p><a href="model_fit/summary.html">Network summary</a>'
    if has_power:
        body += ' · <a href="model_fit/power.html">Acquired power diagnostics</a>'
    body += (
        "</p><p>Inspect joint fits through magnitude, raw I/Q, residuals "
        "and derotated I/Q. Each group offers channel-resonator views and "
        "the full local parameter correlation. Warning masks are not a timestream "
        "acceptance decision. Shared gain uncertainty is included within each "
        "joint fit; cross-group uncertainty is not represented.</p>"
        '<label for="search">Find group or channel</label>'
        '<input id="search" type="search">'
    )
    body += (
        '<label for="flag">Warning filter</label><select id="flag">'
        '<option value="all">All</option><option value="flagged">Any warning</option>'
        '<option value="clear">No listed warning</option>' + options + "</select>"
    )
    body += (
        '<p><button id="previous">Previous page</button>'
        '<span id="page" role="status" aria-live="polite"></span>'
        '<button id="next">Next page</button></p>'
    )
    body += (
        '<div class="scroll"><table id="groups"><caption>Joint fitting groups · '
        "thresholds are saved with the ModelFit result</caption><thead><tr>"
        "<th>Group</th><th>Channels</th><th>Residual / IQ norm</th>"
        "<th>Condition</th><th>Group warnings</th></tr></thead><tbody>"
        + "".join(group_rows)
        + "</tbody></table></div>"
    )
    body += (
        f"<details><summary>All acquired channels ({len(channel_rows)}) · "
        'same search and filter</summary><div class="scroll">'
        '<table id="channels"><thead><tr><th>Channel</th><th>Group responses</th>'
        "<th>Channel warnings</th></tr></thead><tbody>"
        + "".join(channel_rows)
        + "</tbody></table></div></details>"
    )
    return body
