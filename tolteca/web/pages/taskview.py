#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
import dash_html_components as html
from dasha.web.templates.utils import fa
import dash_core_components as dcc
from dash.dependencies import Output, Input
from dasha.web.extensions.celery import get_celery_app
from redbeat.schedulers import get_redis, RedBeatSchedulerEntry
from tollan.utils.fmt import pformat_dict


class TaskView(ComponentTemplate):

    """This is a view that shows the current scheduled tasks."""

    _component_cls = html.Div

    @property
    def title_text(self):
        return (fa("far fa-chart-bar"), "Tasks")

    def setup_layout(self, app):

        db = get_redis(get_celery_app())

        self.interval = self.child(dcc.Interval, update_interval=1)
        content = self.child(html.Pre)

        super().setup_layout(app)

        redbeat_schedule_key = 'redbeat::schedule'

        @app.callback(
                Output(content.id, 'children'),
                [Input(self.interval.id, 'n_intervals')],
                []
                )
        def update(n_intervals):
            result = {}
            for key in db.zrange(redbeat_schedule_key, 0, -1):
                result[key] = RedBeatSchedulerEntry.from_key(key).__dict__
            return pformat_dict(result)

    @property
    def layout(self):
        return super().layout
