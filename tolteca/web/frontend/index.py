import dash_html_components as html

from .utils import get_query_params, get_url
from .common import fa
from . import get_current_dash_app
from tolteca.utils.log import get_logger
from .common import SimplePage
from dash.dependencies import Input, State, Output, ClientsideFunction
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from .utils import odict_from_list


app = get_current_dash_app()
logger = get_logger()


include_pages = odict_from_list([
    SimplePage(p, module_prefix='.pages.', route_prefix=app.config.get("requests_pathname_prefix", "").rstrip('/'))
    for p in (
        'toltecdb',
        'thermometry',
        'kscope',
        )
    ], key=lambda v: v.pathname)


def normalize_pathname(pathname):
    if pathname == '/':
        pathname = next(iter(include_pages.values())).pathname
    return pathname


sidebar_title = html.Header(
        className="brand",
        children=dcc.Link(
            href=get_url(""),
            children=html.H2([fa("far fa-chart-bar"), app.title]),
        ),
    )

sidebar_header = dbc.Row([
        dbc.Col(sidebar_title),
        dbc.Col(
            [
                html.Button(
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    style={
                        "color": "rgba(0,0,0,.5)",
                        "border-color": "rgba(0,0,0,.1)",
                        'outline': 'none',
                    },
                    id="navbar-toggle",
                ),
                html.Button(
                    # use the Bootstrap navbar-toggler classes to style
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    # the navbar-toggler classes don't set color
                    style={
                        "color": "rgba(0,0,0,.5)",
                        "border-color": "rgba(0,0,0,.1)",
                        'outline': 'none',
                    },
                    id="sidebar-toggle",
                ),
            ],
            width="auto",
            align="center",
        ),
    ]
)

sidebar = html.Div(
    [
        sidebar_header,
        dbc.Collapse(
            dbc.Nav(
                [p.nav_link for p in include_pages.values()],
                vertical=True,
                pills=True,
                # className='navbar-dark bg-dark'
            ),
            id="nav-collapse",
        ),
    ],
    id="sidebar",
    className='navbar-dark bg-dark'
)

layout = html.Div([
    dbc.Container(
        fluid=True,
        children=[sidebar, html.Div(id="page-content")], className='px-0'),
    dcc.Location(id="url-location", refresh=False),
    ])


@app.callback(
        Output("page-content", "children"), [
            Input("url-location", "pathname"),
            Input("url-location", "search"),
        ])
def render_page_content(pathname, search):
    if pathname is None:
        raise PreventUpdate(
                "ignoring first Location.pathname callback")
    try:
        rprefix = app.config['requests_pathname_prefix']
        if rprefix is not None:
            import re
            if pathname == rprefix.rstrip("/"):
                pathname = '/'
            elif pathname.startswith(rprefix):
                pathname = re.sub(rprefix, '/', pathname, 1)

        layout = include_pages[normalize_pathname(pathname)].get_layout(
                **get_query_params(search))
        return layout
    except Exception:
        logger.error(f"unable to load page {pathname}", exc_info=True)
        return dbc.Jumbotron([
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"Not a valid pathname {pathname}"),
            ])
    return layout


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(p.nav_link_id, "active") for p in include_pages.values()],
    [Input("url-location", "pathname")],
)
def toggle_active_links(pathname):
    pathname = normalize_pathname(pathname)
    return [pathname == p for p in include_pages.keys()]


app.clientside_callback(
        ClientsideFunction(
            namespace='ui',
            function_name='collapseWithClick',
        ),
        Output("sidebar", 'className'),
        [Input("sidebar-toggle", "n_clicks")],
        [State("sidebar", 'className')],
    )


app.clientside_callback(
        ClientsideFunction(
            namespace='ui',
            function_name='toggleWithClick',
        ),
        Output("nav-collapse", 'is_open'),
        [Input("navbar-toggle", "n_clicks")],
        [State("nav-collapse", 'is_open')],
    )
