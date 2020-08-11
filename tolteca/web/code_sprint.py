#! /usr/bin/env python

from . import env_registry, env_prefix, dasha_config
from . import extensions  # noqa: F401

# register an env var to only include the specified page
# in the list of pages
env_registry.register(
        f"{env_prefix}_CODE_SPRINT_TEMPLATE_NAME",
        "The TolTEC code sprint template name.",
        'data_explorer')


# we modify the dasha_config to only include the specified page module

page_module_name = env_registry.get(
    f"{env_prefix}_CODE_SPRINT_TEMPLATE_NAME")

dasha_config.update(
    title_text='TolTECA',
    pages=[
        {
            'template': f"tolteca.web.templates.{page_module_name}",
            'route_name': page_module_name,
            'title_text': page_module_name,
            'title_icon': 'fa fa-free-code-camp'
            },
        ]
    )
