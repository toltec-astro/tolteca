from collections import namedtuple
from enum import Enum, Flag, auto
import abc
from collections import defaultdict
from functools import cached_property
import functools
from loguru import logger
from .types import ToltecDataKind


class KidsDataAxis(Enum):
    Block = "block"
    Chan = "chan"
    Sample = "sample"
    Sweep = "sweep"
    Time = "time"


class KidsDataAxisSlicerMeta(abc.ABCMeta):
    """A mixin class to provide helpers to slice along axis.

    This enables a set of properties ``*_loc`` to installed class.
    These properties are of type `_slicer_cls` (if not provided,
    the installed class is used as the slicer class).

    The installed class is expected to have property `io_obj`
    that provide hints of the data kind.
    """

    # the below makes available the get_axis_types method based on
    # the data kind
    _axis_types_map = {
        ToltecDataKind.KidsData: {
            KidsDataAxis.Block,
            KidsDataAxis.Chan,
            KidsDataAxis.Sample,
        },
        ToltecDataKind.Sweep: {
            KidsDataAxis.Sweep,
        },
        ToltecDataKind.TimeStream: {KidsDataAxis.Time},
    }

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        def _get_axis_types(inst):
            slicer_cls = getattr(inst, "_slicer_cls", cls)
            if not isinstance(inst, slicer_cls):
                inst = slicer_cls(inst)
            data_kind = inst.io_obj.data_kind
            result = set()
            for k, types in cls._axis_types_map.items():
                if data_kind & k:
                    result = result.union(types)
            return result

        cls.axis_types = cached_property(_get_axis_types)

        # the below makes available the *_loc methods so
        # that is returns an instance of slicer_cls
        def _axis_loc(inst, axis_type=None):
            slicer_cls = getattr(inst, "_slicer_cls", cls)
            if isinstance(inst, slicer_cls):
                slicer = inst
                io_obj = inst.io_obj
            else:
                # inst is the underlying obj
                slicer = slicer_cls(inst)
                io_obj = inst.io_obj
            if axis_type not in io_obj.axis_types:
                raise ValueError(
                    f"{axis_type} axis is not available"
                    f" for data kind {io_obj.data_kind}"
                )
            slicer._axis_type = axis_type
            return slicer

        for t in functools.reduce(
            lambda a, b: a.union(b), cls._axis_types_map.values()
        ):
            func = functools.partial(_axis_loc, axis_type=t)
            name = f"{t}_loc"
            prop = property(func)
            prop.__doc__ = f"The axis slicer for locating {t}."
            setattr(cls, name, prop)


class KidsDataAxisSlicer(object, metaclass=KidsDataAxisSlicerMeta):
    """A helper class to provide sliced view of `DataIO` for KIDs data.

    The class follows the builder pattern to collect arguments
    and keyword arguments that are later packed and send to
    the :meth:`read` function to load the sliced data.

    Parameters
    ----------
    io_obj : `DataIO`
        The data IO object.
    """

    def __init__(self, arg):
        if isinstance(arg, self.__class__):
            # arg is a slicer, get the underlying
            self._io_obj = arg.io_obj
        else:
            # arg is io_obj
            self._io_obj = arg
        self._args = defaultdict(lambda: [(), {}])

    def __getitem__(self, *args):
        # store the positional args
        self._args[self._axis_type][0] = args  # type: ignore
        return self

    def __call__(self, *args, **kwargs):
        # store all args
        self._args[self._axis_type][0] = args  # type: ignore
        self._args[self._axis_type][1] = kwargs  # type: ignore
        return self

    def io_obj(self):
        return self._io_obj

    def read(self):
        # call the data reader
        return self.io_obj._read_sliced(self)

    def get_slice_op(self, axis_type):
        # this returns the actual slice operator to be applied to the data
        dispatch_validator = {
            KidsDataAxis.Block: self._validate_slicer_index_only,
            KidsDataAxis.Time: self._validate_slicer_slice_only,
        }
        result = dispatch_validator.get(axis_type, self._validate_slicer_args_only)(
            self, axis_type
        )
        logger.debug(f"resolved {axis_type} slice obj: {result}")
        return result

    @staticmethod
    def _validate_slicer_index_only(slicer, axis_type):
        args, kwargs = slicer._args[axis_type]
        if len(kwargs) > 0:
            raise ValueError(f"{axis_type} loc does not accept keyword arguments.")
        if len(args) == 0:
            s = None
        elif len(args) == 1:
            s = args[0]
        else:
            raise ValueError(f"{axis_type} loc expects one argument.")
        if s is not None and not isinstance(s, int):
            raise ValueError(f"{axis_type} loc can only be integer or None.")
        return s

    @staticmethod
    def _validate_slicer_slice_only(slicer, axis_type):
        args, kwargs = slicer._args[axis_type]
        if len(kwargs) > 0:
            raise ValueError(f"{axis_type} loc does not accept keyword arguments.")
        if len(args) == 0:
            s = None
        elif len(args) == 1:
            s = args[0]
        else:
            raise ValueError(f"{axis_type} loc expects one argument.")
        if s is not None and not isinstance(s, slice):
            raise ValueError(f"{axis_type} loc can only be slice.")
        return s

    @staticmethod
    def _validate_slicer_args_only(slicer, axis_type):
        args, kwargs = slicer._args[axis_type]
        if len(kwargs) > 0:
            raise ValueError(f"{axis_type} loc does not accept keyword arguments.")
        if len(args) > 1:
            raise ValueError(f"{axis_type} loc expects one argument.")
        if len(args) == 0:
            return None
        if len(args) == 1:
            return args[0]
