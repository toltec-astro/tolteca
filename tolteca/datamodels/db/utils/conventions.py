#! /usr/bin/env python

from datetime import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime


def fk(other):
    return Column(
        f'{other}_pk', Integer,
        ForeignKey(
           f"{other}.pk", onupdate="cascade", ondelete="cascade"),
        nullable=False)


def pfk(other):
    return Column(
        f'{other}_pk', Integer,
        ForeignKey(
           f"{other}.pk", onupdate="cascade", ondelete="cascade"),
        primary_key=True)


def pk(**kwargs):
    return Column('pk', Integer, primary_key=True, **kwargs)


def label(**kwargs):
    return Column('label', String, **kwargs)


def created_at(**kwargs):
    return Column('created_at', DateTime, default=datetime.utcnow(), **kwargs)
