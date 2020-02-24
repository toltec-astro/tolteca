#! /usr/bin/env python


"""This recipe shows how to work with the `kidsdata.wcs` Module."""

from gwcs import coordinate_frames as cf
from gwcs import selector, wcs, LabelMapper
from kidsproc.kidsdata.wcs.frames import NDDataFrame
import astropy.units as u
from astropy.time import Time
from astropy.modeling import models
import numpy as np


def main():
    tone_axis = NDDataFrame(
            axes_order=(0, ),
            name='tone_axis'
            )

    time_axis = cf.TemporalFrame(
            reference_frame=Time,
            axes_order=(1, ),
            unit=('s', ),
            axes_names=("time", ),
            name='time_axis'
            )
    sweep_axis = cf.SpectralFrame(
            axes_order=(1, ),
            unit=('GHz', ),
            axes_names=("flo", ),
            name='sweep_axis'
            )
    data_frame = NDDataFrame(axes_order=(0, 1), name='data_axis')
    sweep_frame = cf.CompositeFrame(
            [tone_axis, sweep_axis], name='sweep')

    # the transformation
    sel = {
            i: models.Shift(i) | models.Multiply(1e3 * u.GHz) |
            models.Shift(1e8 * u.GHz)
            for i in range(1, 10)
            }

    regions_transform = selector.RegionsSelector(
            inputs=['i', 'j'],
            outputs=['tone', 'f', 'iq'],
            selector=sel,
            label_mapper=LabelMapper(('i', 'j'), models.Mapping([0, ])),
            undefined_transform_value=np.nan)
    print(
            tone_axis.__repr__(),
            time_axis.__repr__(),
            sweep_axis.__repr__(),
            data_frame.__repr__())
    kids_wcs = wcs.WCS(
            forward_transform=regions_transform,
            output_frame=sweep_frame, input_frame=data_frame)

    from asdf import AsdfFile
    tree = {"wcs": kids_wcs}
    wcs_file = AsdfFile(tree)
    wcs_file.write_to('test_kids_wcs.asdf')


if __name__ == "__main__":
    main()
