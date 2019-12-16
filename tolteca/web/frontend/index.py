import dash_html_components as html

from .utils import DashRouter, DashNavBar
from .pages import (
        character_counter, page2, page3, toltecdb, kidsview, kscope,
        thermometry)
from .components import fa
from . import get_current_dash_app

app = get_current_dash_app()

# Ordered iterable of routes: tuples of (route, layout), where 'route' is a
# string corresponding to path of the route (will be prefixed with Dash's
# 'routes_pathname_prefix' and 'layout' is a Dash Component.
urls = (
    ("", toltecdb.get_layout),
    ("toltecdb", toltecdb.get_layout),
    ("kscope", kscope.get_layout),
    ("thermometry", thermometry.get_layout),
    ("kidsview", kidsview.get_layout),
    ("character-counter", character_counter.get_layout),
    ("page2", page2.layout),
    ("page3", page3.layout),
)

# Ordered iterable of navbar items: tuples of `(route, display)`, where `route`
# is a string corresponding to path of the route (will be prefixed with
# 'routes_pathname_prefix') and 'display' is a valid value for the `children`
# keyword argument for a Dash component (ie a Dash Component or a string).
nav_items = (
    ("toltecdb", html.Div([fa("fas fa-table"), "TolTEC Database"])),
    ("kscope", html.Div(
        [fa("fas fa-keyboard"), "Kids Scope"])),
    ("thermometry", html.Div(
        [fa("fas fa-keyboard"), "Thermometry"])),
    ("kidsview", html.Div(
        [fa("fas fa-keyboard"), "Kids Exam"])),
    ("character-counter", html.Div(
        [fa("fas fa-keyboard"), "Character Counter"])),
    ("page2", html.Div([fa("fas fa-chart-area"), "Page 2"])),
    ("page3", html.Div([fa("fas fa-chart-line"), "Page 3"])),
)

router = DashRouter(app, urls)
navbar = DashNavBar(app, nav_items)
