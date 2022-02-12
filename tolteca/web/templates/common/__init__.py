#! /usr/bin/env python


import dash_html_components as html

from dash_component_template import ComponentTemplate
from schema import Schema, Optional
from .toltec_logo_src import toltec_logo_src


class HeaderWithToltecLogo(ComponentTemplate):

    _component_cls = html.Div
    _component_schema = Schema({
        Optional('logo_colwidth', default=4): int,
        })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        header_container, logo_container = self.grid(1, 2)
        header_container.width = 12
        header_container.md = 12 - self.logo_colwidth
        logo_container.width = self.logo_colwidth
        logo_container.className = 'd-none d-md-block'
        logo_container.child(
                      html.Img(src=toltec_logo_src, height="150px"))
        self._header_container = header_container

    @property
    def header_container(self):
        return self._header_container
