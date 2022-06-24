#! /usr/bin/env python


from dash import html

from dash_component_template import ComponentTemplate
from .toltec_logo_src import toltec_logo_src


class HeaderWithToltecLogo(ComponentTemplate):

    class Meta:
        component_cls = html.Div

    def __init__(self, logo_colwidth=4, **kwargs):
        super().__init__(**kwargs)

        header_container, logo_container = self.grid(1, 2)
        header_container.width = 12
        header_container.md = 12 - logo_colwidth
        logo_container.width = logo_colwidth
        logo_container.className = 'd-none d-md-block'
        logo_container.child(
                      html.Img(src=toltec_logo_src, height="150px"))
        self._header_container = header_container

    @property
    def header_container(self):
        return self._header_container
