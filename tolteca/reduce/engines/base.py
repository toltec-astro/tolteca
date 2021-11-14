#!/usr/bin/env python


__all__ = ['PipelineEngine', 'PipelineEngineError']


class PipelineEngine(object):
    """Base class for pipeline engine."""
    pass


class PipelineEngineError(RuntimeError):
    """Base class for exceptions related to pipeline engine."""
    pass
