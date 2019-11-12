#! /usr/bin/env python

"""CLI console logging with highlighted words."""

import re
import logging

import click

from .. import log


class ColorFormatter(logging.Formatter):

    colors = {
        logging.ERROR: dict(fg='red'),
        logging.CRITICAL: dict(fg='red'),
        logging.INFO: dict(fg='green'),
        logging.WARNING: dict(fg='yellow'),
        logging.DEBUG: dict(fg='white'),
        "SUCCESS": dict(fg='green', bold=True),
        "FAIL": dict(fg='red', bold=True),
        "IMPORTANT": dict(fg='yellow', bold=False),
        "META": dict(fg='white', bold=False),
        "SPECIAL": dict(fg='green', bold=True)
    }

    def apply_color_style(self, str_, key):
        return ' ' + click.style(str_, **self.colors[key])

    re_level = re.compile(
            r'(^\[?(?:ERROR|CRITICAL|WARNING|DEBUG|INFO)\]?)\s*'
            r'([^ ]+:)(?= )',
            re.IGNORECASE | re.MULTILINE)

    # re_success = re.compile(
    #         r'( (SUCCESS|SUCCESSFUL|SUCCESSFULLY|OK)(?=( |$)))',
    #         re.IGNORECASE)
    re_success = re.compile(
            r'(?: (SUCCESS)(?=( |$)))',
            re.IGNORECASE)
    # re_important = re.compile(
    #         r'( (START|STARTED|STARTING|RUN|RUNNING|'
    #         r'CLOSE|CLOSED|CLOSING|STOP|STOPPED|STOPPING)(?=( |$)))',
    #         re.IGNORECASE)
    re_important = re.compile(
            r'(?: (STARTED|RUNNING|'
            r'CLOSED|STOPPED|FINISHED|SHUTDOWN|DONE)(?=( |$)))',
            re.IGNORECASE)

    re_fail = re.compile(
            r'(?: (FAIL|FAILED|ERROR)(?=( |$)))',
            re.IGNORECASE)

    re_special = re.compile(
            r'(?: (Jobs Done!))',
            )

    def format(self, record):
        msg = super().format(record)
        if not record.exc_info:
            msg = re.sub(
                    self.re_success,
                    lambda s: self.apply_color_style(s.group(1), 'SUCCESS'),
                    msg)
            msg = re.sub(
                    self.re_fail,
                    lambda s: self.apply_color_style(s.group(1), 'FAIL'),
                    msg)
            msg = re.sub(
                    self.re_important,
                    lambda s: self.apply_color_style(s.group(1), 'IMPORTANT'),
                    msg)
            msg = re.sub(
                    self.re_special,
                    lambda s: self.apply_color_style(s.group(1), 'SPECIAL'),
                    msg)
            level = record.levelno
            msg = re.sub(
                    self.re_level,
                    ''.join([
                        self.apply_color_style(r'\1', level),
                        self.apply_color_style(r'\2', 'META'),
                        ]),
                    msg
                    )
        return msg.lstrip()


class ClickHandler(logging.Handler):
    _use_stderr = True

    def emit(self, record):
        try:
            msg = self.format(record)
            click.echo(msg, err=self._use_stderr)
        except Exception:
            self.handleError(record)


def init(level, file_=None):
    config = {
        'formatters': {
            'click': {
                'class': f'{__name__}.ColorFormatter',
                'format': '[%(levelname)s] %(name)s: %(message)s'
                }
            },
        'handlers': {
            'default': {
                'class': f'{__name__}.ClickHandler',
                'formatter': 'click'
                },
            },
        'loggers': {
            '': {'level': level},
            }
        }
    if file_ is not None:
        config['handlers']['logfile'] = {
                'class': 'logging.FileHandler',
                'filename': file_,
                'formatter': "standard",
                }
        config['loggers']['']['handlers'] = ['logfile', ]

    log.init_logging(config)
