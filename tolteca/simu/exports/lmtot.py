#!/usr/bin/env python

import astropy.units as u
import numpy as np
from tollan.utils.log import get_logger

from ..mapping.raster import SkyRasterScanModel
from ..mapping.lissajous import (
    SkyLissajousModel, SkyDoubleLissajousModel,
    SkyRastajousModel)


class LmtOTComposer(object):
    """A class to create LMT OT scripts."""

    logger = get_logger()

    def __init__(self):
        pass

    def make_pointing_steps(self):
        pass

    def make_mapping_steps(
            self, mapping_model, target_name=None, t_exp=None):
        steps = list()
        target = mapping_model.target
        ref_frame = mapping_model.ref_frame
        if t_exp is None:
            t_exp = mapping_model.t_pattern
        if target_name is None:
            target_name = 'unknown_source'
        if target.frame.name == 'icrs':
            # Eq
            source_line = (
                "Source Source;  Source  -BaselineList [] -CoordSys Eq"
                " -DecProperMotionCor 0"
                " -Dec[0] {Dec} -Dec[1] {Dec}"
                " -El[0] 0.000000 -El[1] 0.000000 -EphemerisTrackOn 0"
                " -Epoch 2000.0"
                " -GoToZenith 0 -L[0] 0.0 -L[1] 0.0 -LineList [] -Planet None"
                " -RaProperMotionCor 0 -Ra[0] {RA} -Ra[1] {RA}"
                " -SourceName {Name}"
                " -VelSys Lsr -Velocity 0.000000 -Vmag 0.0").format(
                    RA=target.ra.to_string(
                        unit=u.hour, pad=True,
                        decimal=False, fields=3, sep=':'),
                    Dec=target.dec.to_string(
                        unit=u.degree, pad=True, alwayssign=True,
                        decimal=False, fields=3, sep=':'),
                    Name=target_name,
                    )
            steps.append(source_line)
        else:
            raise ValueError(f"Invalid target frame {target.frame.name}")
        # mapping line
        # TODO fix this for non-offset models
        m = mapping_model.offset_mapping_model
        if isinstance(m, (SkyLissajousModel)):
            mapping_line = (
                "Lissajous -ExecMode 0 -RotateWithElevation 0 -TunePeriod 0"
                " -TScan {TScan} -ScanRate {ScanRate}"
                " -XLength {XLength} -YLength {YLength} -XOmega {XOmega}"
                " -YOmega {YOmega} -XDelta {XDelta}"
                " -XLengthMinor 0.0 -YLengthMinor 0.0 -XDeltaMinor 0.0"
                ).format(**self._sky_lissajous_params_to_mapping_step_params(
                    x_length=m.x_length,
                    y_length=m.y_length,
                    x_omega=m.x_omega,
                    y_omega=m.y_omega,
                    delta=m.delta,
                    t_exp=t_exp
                    ))
        elif isinstance(m, (SkyDoubleLissajousModel)):
            major_params = self._sky_lissajous_params_to_mapping_step_params(
                    x_length=m.x_length_0,
                    y_length=m.y_length_0,
                    x_omega=m.x_omega_0,
                    y_omega=m.y_omega_0,
                    delta=m.delta_0,
                    t_exp=t_exp
                    )
            minor_params = self._sky_lissajous_params_to_mapping_step_params(
                    x_length=m.x_length_1,
                    y_length=m.y_length_1,
                    x_omega=m.x_omega_1,
                    y_omega=m.y_omega_1,
                    delta=m.delta_1,
                    t_exp=0 << u.s
                    )
            self.logger.warning(
                    "some parameters will be ignored during the exporting "
                    "and the result will be different.")
            mapping_line = (
                "Lissajous -ExecMode 0 -RotateWithElevation 0 -TunePeriod 0"
                " -TScan {TScan} -ScanRate {ScanRate}"
                " -XLength {XLength} -YLength {YLength} -XOmega {XOmega}"
                " -YOmega {YOmega} -XDelta {XDelta}"
                " -XLengthMinor {XLengthMinor} -YLengthMinor {YLengthMinor}"
                " -XDeltaMinor {XDeltaMinor}"
                ).format(
                    XLengthMinor=minor_params['XLength'],
                    YLengthMinor=minor_params['YLength'],
                    XDeltaMinor=minor_params['XDelta'],
                    **major_params)
        elif isinstance(m, (SkyRasterScanModel)):
            mapping_line = (
                "RasterMap Map; Map -ExecMode 0 -HPBW 1 -HoldDuringTurns 0"
                " -MapMotion Continuous -NumPass 1 -NumRepeats 1 -NumScans 0"
                " -RowsPerScan 1000000 -ScansPerCal 0 -ScansToSkip 0"
                " -TCal 0 -TRef 0 -TSamp 1"
                " -MapCoord {MapCoord} -ScanAngle {ScanAngle}"
                " -XLength {XLength} -XOffset 0 -XRamp 0 -XStep {XStep}"
                " -YLength {YLength} -YOffset 0 -YRamp 0 -YStep {YStep}"
                ).format(**self._sky_raster_params_to_mapping_step_params(
                    length=m.length,
                    space=m.space,
                    n_scans=m.n_scans,
                    rot=m.rot,
                    speed=m.speed,
                    ref_frame=ref_frame
                    ))
        elif isinstance(m, (SkyRastajousModel)):
            raster_params = self._sky_raster_params_to_mapping_step_params(
                    length=m.length,
                    space=m.space,
                    n_scans=m.n_scans,
                    rot=m.rot,
                    speed=m.speed,
                    ref_frame=ref_frame
                    )
            major_params = self._sky_lissajous_params_to_mapping_step_params(
                    x_length=m.x_length_0,
                    y_length=m.y_length_0,
                    x_omega=m.x_omega_0,
                    y_omega=m.y_omega_0,
                    delta=m.delta_0,
                    t_exp=t_exp
                    )
            minor_params = self._sky_lissajous_params_to_mapping_step_params(
                    x_length=m.x_length_1,
                    y_length=m.y_length_1,
                    x_omega=m.x_omega_1,
                    y_omega=m.y_omega_1,
                    delta=m.delta_1,
                    t_exp=0 << u.s
                    )
            self.logger.warning(
                    "some parameters will be ignored during the exporting "
                    "and the result will be different.")
            mapping_line_lissajous = (
                "Lissajous -ExecMode 1 -RotateWithElevation 0 -TunePeriod 0"
                " -TScan {TScan} -ScanRate {ScanRate}"
                " -XLength {XLength} -YLength {YLength} -XOmega {XOmega}"
                " -YOmega {YOmega} -XDelta {XDelta}"
                " -XLengthMinor {XLengthMinor} -YLengthMinor {YLengthMinor}"
                " -XDeltaMinor {XDeltaMinor}"
                ).format(
                    XLengthMinor=minor_params['XLength'],
                    YLengthMinor=minor_params['YLength'],
                    XDeltaMinor=minor_params['XDelta'],
                    **major_params)
            mapping_line_raster = (
                "RasterMap Map; Map -ExecMode 1 -HPBW 1 -HoldDuringTurns 0"
                " -MapMotion Continuous -NumPass 1 -NumRepeats 1 -NumScans 0"
                " -RowsPerScan 1000000 -ScansPerCal 0 -ScansToSkip 0"
                " -TCal 0 -TRef 0 -TSamp 1"
                " -MapCoord {MapCoord} -ScanAngle {ScanAngle}"
                " -XLength {XLength} -XOffset 0 -XRamp 0 -XStep {XStep}"
                " -YLength {YLength} -YOffset 0 -YRamp 0 -YStep {YStep}"
                ).format(**raster_params)
            mapping_line = f"{mapping_line_lissajous}\n{mapping_line_raster}"
        else:
            raise NotImplementedError
        steps.append(mapping_line)
        return steps

    def make_setup_steps(self, instru_name):
        if instru_name == 'toltec':
            return self._toltec_setup_steps()

        raise NotImplementedError(
            f"setup steps not implemented for {instru_name}")

    def _toltec_setup_steps(self):
        steps = list()
        steps.append('ObsGoal Dcs; Dcs -ObsGoal Science')
        return steps

    @classmethod
    def _sky_lissajous_params_to_mapping_step_params(
            cls,
            x_length, y_length, x_omega, y_omega,
            delta, t_exp
            ):
        v_x = x_length * x_omega.quantity.to(
                u.Hz, equivalencies=[(u.cy/u.s, u.Hz)])
        v_y = y_length * y_omega.quantity.to(
                u.Hz, equivalencies=[(u.cy/u.s, u.Hz)])
        return dict(
            XLength=x_length.quantity.to_value(u.arcmin),
            YLength=y_length.quantity.to_value(u.arcmin),
            XOmega=x_omega.quantity.to_value(u.rad/u.s),
            YOmega=y_omega.quantity.to_value(u.rad/u.s),
            XDelta=delta.quantity.to_value(u.rad),
            TScan=t_exp,
            ScanRate=np.hypot(v_x, v_y).to_value(u.arcsec / u.s),
            )

    @classmethod
    def _sky_raster_params_to_mapping_step_params(
            cls,
            length, space, n_scans, rot, speed, ref_frame
            ):
        if ref_frame.name == 'icrs':
            MapCoord = 'Ra'
        elif ref_frame.name == 'altaz':
            MapCoord = 'Az'
        else:
            raise NotImplementedError(f"invalid ref_frame {ref_frame}")
        return dict(
            MapCoord=MapCoord,
            ScanAngle=rot.quantity.to_value(u.deg),
            XLength=length.quantity.to_value(u.arcsec),
            XStep=speed.quantity.to_value(u.arcsec / u.s),
            YLength=(n_scans * space.quantity).to_value(u.arcsec),
            YStep=space.quantity.to_value(u.arcsec),
            )
