"""Timestream projection analysis step.

For every raw data sample (nw, detector, time-index) this step records which
source segment, if any, is being observed.  The raw data files are **read but
never modified**.

The segmentation image is rendered from the input point-source catalog using
filled circles (one image per TolTEC array because each array has its own beam
FWHM).  The rendering mirrors
:meth:`~tolteca.simu.sources.models.CatalogSourceModel.make_image_model` as
closely as possible – the WCS geometry, pixel scale, image size and padding are
identical.  The only difference is that instead of rendering Gaussian flux
profiles we paint filled circles whose radius is
``fwhm * segment_radius_to_fwhm_ratio`` and whose pixel value is
``source_index + 1`` (0 = background).

Output (written to a numbered ``<jobkey>NNN/`` directory):

``tolteca.yaml``
    Copy of the step configuration.
``seg_model.fits``
    FITS file with one image extension per TolTEC array, each containing the
    integer segmentation map.
``seg_sources.ecsv``
    Source table with columns ``seg_id``, ``source_name``, ``ra``, ``dec``.
``toltecXX_seg_ids.npz``
    Compressed numpy archive per network (NW) with keys:

    ``seg_ids``
        ``int16`` array of shape ``(n_good_det, n_nc_times)``.  Value 0 means
        background; value *k* means the detector/sample visited source
        *k – 1* (0-based) in the catalog / ``seg_sources.ecsv``.
    ``det_uids``
        Detector UIDs (``int``) corresponding to rows of ``seg_ids``.
    ``t_i0``
        Absolute time index in the raw nc file of ``seg_ids[:, 0]``.
    ``fsmp``
        Sampling frequency in Hz of the raw nc file.
    ``t0_grid``
        Unix timestamp of sample index 0 in the raw nc file.
Output layout inside ``<jobkey>NNN/``:

``seg_model.fits``
    Shared across all obsnums (written once).
``seg_sources.ecsv``
    Shared across all obsnums (written once).
``tolteca.yaml``
    Step configuration.
``<obsnum_subobsnum_scannum>/``
    Per-obsnum subdirectory containing:

    ``toltecXX_seg_ids.npz``
        Per-network arrays (see above).
    ``seg_visits.ecsv``
        Table of every consecutive visit (≥ ``n_consec_sample_min`` samples)
        for that obsnum.  Columns:

        ``nw``         – network index
        ``kids_tone``  – tone column index in the raw nc file
        ``det_uid``    – detector UID
        ``seg_id``     – segment visited (same encoding as ``seg_ids``)
        ``t_start``    – start sample index (absolute, in the nc file)
        ``t_stop``     – stop  sample index (exclusive)
        ``n_samples``  – run length (= t_stop − t_start)
        ``t_start_s``  – start time in seconds relative to ``t_i0``
        ``duration_s`` – duration in seconds
"""
from ... import steps_registry
from pathlib import Path
import numpy as np
from astropy.table import Table, QTable
from astropy.time import Time
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from scipy.interpolate import interp1d

from ....datamodels.toltec import BasicObsDataset
from tollan.utils.log import get_logger, timeit
from tollan.utils.dataclass_schema import add_schema
from dataclasses import dataclass, field

from ....simu import PerfParamsConfig, sources_registry, RuntimeBase
from ....simu.mapping.lmt_tcs import LmtTcsTrajMappingModel
from ...engines.citlali import CitlaliConfig, CitlaliProc
from ....simu.sources.models import CatalogSourceModel
from ....simu.toltec.toltec_info import toltec_info
from ....simu.toltec.simulator import ToltecObsSimulator
from ....simu.utils import make_time_grid, SkyBoundingBox

from .simu import SimuExecutor, _load_time_grid, _load_kids_model


# ---------------------------------------------------------------------------
# Consecutive-visit extraction
# ---------------------------------------------------------------------------

def _find_all_seg_visits(
    seg_ids: np.ndarray,
    min_samples: int,
    det_uids: np.ndarray | None = None,
    *,
    t_i0: int = 0,
    fsmp: float = 1.0,
) -> QTable:
    """Find every consecutive run of a non-background segment.

    A *visit* is a maximal contiguous block of samples where a single
    detector's ``seg_id`` holds the same non-zero value.  Only visits
    whose length is ≥ *min_samples* are returned.

    Parameters
    ----------
    seg_ids : ndarray, shape (n_det, n_times), int16
        Per-detector, per-sample segment assignments.
    min_samples : int
        Minimum run length to keep.
    det_uids : 1-D array, optional
        Detector UID for each row.  Defaults to 0, 1, 2, …
    t_i0 : int
        Index of the first filled sample (samples before this are skipped).
    fsmp : float
        Sampling rate in Hz (used to convert to seconds).

    Returns
    -------
    QTable
        Columns: ``kids_tone, det_uid, seg_id, t_start, t_stop,
        n_samples, t_start_s, duration_s``.
    """
    n_det, n_times = seg_ids.shape
    if det_uids is None:
        det_uids = np.arange(n_det, dtype=np.int64)
    else:
        det_uids = np.asarray(det_uids, dtype=np.int64)

    # Only process the filled time range
    seg_slice = seg_ids[:, t_i0:]   # (n_det, n_filled)
    n_filled  = seg_slice.shape[1]

    tones_out   = []
    uids_out    = []
    segids_out  = []
    starts_out  = []
    stops_out   = []
    lens_out    = []

    # Sentinel value that is never a valid seg_id (which is ≥ 0)
    _SENT = np.int32(-1)

    padded = np.empty(n_filled + 2, dtype=np.int32)
    padded[0]  = _SENT
    padded[-1] = _SENT

    for row_idx in range(n_det):
        padded[1:-1] = seg_slice[row_idx].astype(np.int32)

        # Indices in diff where a transition occurs:
        # diff[j] != 0  ⟺  padded[j+1] != padded[j]
        # → new run starts at padded[j+1], i.e. ts_idx = j
        boundaries = np.flatnonzero(np.diff(padded))

        if len(boundaries) < 2:
            # Entire row is a single constant value – no interesting runs
            continue

        run_starts = boundaries[:-1]              # ts_idx (0-based in seg_slice)
        run_stops  = boundaries[1:]               # exclusive
        run_lens   = run_stops - run_starts
        run_vals   = padded[boundaries[:-1] + 1].astype(np.int16)

        # Keep non-background runs that meet the length threshold
        keep = (run_vals > 0) & (run_lens >= min_samples)
        if not keep.any():
            continue

        for rs, re, rv, rl in zip(
                run_starts[keep], run_stops[keep],
                run_vals[keep],   run_lens[keep]):
            tones_out.append(row_idx)
            uids_out.append(int(det_uids[row_idx]))
            segids_out.append(int(rv))
            starts_out.append(int(rs) + t_i0)
            stops_out.append(int(re) + t_i0)
            lens_out.append(int(rl))

    n_visits = len(tones_out)
    tbl = QTable()
    tbl['kids_tone']  = np.array(tones_out,  dtype=np.int32)
    tbl['det_uid']    = np.array(uids_out,   dtype=np.int64)
    tbl['seg_id']     = np.array(segids_out, dtype=np.int16)
    tbl['t_start']    = np.array(starts_out, dtype=np.int64)
    tbl['t_stop']     = np.array(stops_out,  dtype=np.int64)
    tbl['n_samples']  = np.array(lens_out,   dtype=np.int32)
    if n_visits:
        tbl['t_start_s']  = (tbl['t_start'] - t_i0) / fsmp * u.s
        tbl['duration_s'] = tbl['n_samples']         / fsmp * u.s
    else:
        tbl['t_start_s']  = np.array([], dtype=float) * u.s
        tbl['duration_s'] = np.array([], dtype=float) * u.s
    return tbl


# ---------------------------------------------------------------------------
# Segmentation image helpers
# ---------------------------------------------------------------------------

def _render_circle(seg_img, cx, cy, radius_pix, seg_value):
    """Paint a filled circle at pixel centre ``(cx, cy)`` onto *seg_img*.

    The image is modified in-place.  All pixels whose centre-to-centre
    distance from ``(cx, cy)`` is ≤ *radius_pix* are set to *seg_value*.

    Parameters
    ----------
    seg_img : 2-D ``int`` ndarray  (modified in place)
    cx, cy : float  pixel coordinates (column, row) of the circle centre
    radius_pix : float  circle radius in pixels
    seg_value : int  value to paint
    """
    s = seg_img.shape[0]  # image is square
    i_lo = max(0, int(np.floor(cy - radius_pix)))
    i_hi = min(s, int(np.ceil(cy + radius_pix)) + 1)
    j_lo = max(0, int(np.floor(cx - radius_pix)))
    j_hi = min(s, int(np.ceil(cx + radius_pix)) + 1)
    ii, jj = np.mgrid[i_lo:i_hi, j_lo:j_hi]
    mask = (ii - cy) ** 2 + (jj - cx) ** 2 <= radius_pix ** 2
    seg_img[i_lo:i_hi, j_lo:j_hi][mask] = seg_value


# ---------------------------------------------------------------------------
# SegmentationImageModel
# ---------------------------------------------------------------------------

class SegmentationImageModel:
    """Sky segmentation map built from a point-source catalog.

    One FITS ``ImageHDU`` per TolTEC array is maintained, each containing a
    2-D ``int16`` image.  Pixel value 0 is background; pixel value ``i + 1``
    corresponds to source index *i* (0-based) in the input catalog.

    Construct via :meth:`from_catalog_source_model` rather than directly.

    Parameters
    ----------
    hdus_by_array : dict  ``{array_name: ImageHDU}``
    source_names : list of str
        Source names in catalog order (index 0 → seg_id 1, …).
    wcsobj : `~astropy.wcs.WCS`
        The common WCS shared by all HDUs (stored for inspection).
    """

    logger = get_logger()

    def __init__(self, hdus_by_array, source_names, wcsobj):
        self._hdus_by_array = hdus_by_array
        self.source_names = list(source_names)
        self.wcsobj = wcsobj

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_catalog_source_model(
            cls, catalog_model, fwhms, pixscale,
            segment_radius_to_fwhm_ratio=1.0):
        """Build a segmentation image from a :class:`CatalogSourceModel`.

        The WCS layout intentionally mirrors
        :meth:`CatalogSourceModel.make_image_model
        <tolteca.simu.sources.models.CatalogSourceModel.make_image_model>`
        so that both images share the same sky footprint and pixel scale.
        The only difference is that instead of rendering Gaussian flux
        profiles we paint filled circles whose radius equals
        ``fwhm * segment_radius_to_fwhm_ratio`` and whose pixel value is
        ``source_index + 1``.

        Parameters
        ----------
        catalog_model : `CatalogSourceModel`
        fwhms : dict  ``{array_name: Quantity}``  beam FWHMs
        pixscale : `~astropy.units.Quantity`
            Pixel scale as ``arcsec / pix`` (the value only; unit is
            ``u.arcsec / u.pix`` or an equivalent).
        segment_radius_to_fwhm_ratio : float
            Multiplier applied to the per-array FWHM to obtain the circle
            radius.  Default 1.0.
        """
        logger = get_logger()
        pixscale_equiv = u.pixel_scale(pixscale)
        delta_pix = (1.0 << u.pix).to(u.arcsec, equivalencies=pixscale_equiv)

        # ---- WCS setup identical to CatalogSourceModel.make_image_model ----
        ref_coord = catalog_model.pos[0]
        wcsobj = WCS(naxis=2)
        wcsobj.wcs.crpix = [1.5, 1.5]
        wcsobj.wcs.cdelt = np.array([
            -delta_pix.to_value(u.deg),
             delta_pix.to_value(u.deg),
        ])
        wcsobj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcsobj.wcs.crval = [ref_coord.ra.degree, ref_coord.dec.degree]

        x, y = wcsobj.wcs_world2pix(
            catalog_model.pos.ra.degree,
            catalog_model.pos.dec.degree, 0)
        l_x, r_x = float(np.min(x)), float(np.max(x))
        b_y, t_y = float(np.min(y)), float(np.max(y))

        fwhm_max = np.max(u.Quantity(list(fwhms.values())))
        padding = 10.0 * fwhm_max.to_value(u.pix, equivalencies=pixscale_equiv)
        s = int(np.ceil(np.max([r_x - l_x, t_y - b_y]) + padding))
        logger.debug(f"segmentation image size: {s}×{s} px")

        c_ra, c_dec = wcsobj.wcs_pix2world(
            (l_x + r_x) / 2.0, (b_y + t_y) / 2.0, 0)
        wcsobj.wcs.crpix = [s / 2 + 1, s / 2 + 1]
        wcsobj.wcs.crval = [float(c_ra), float(c_dec)]
        header = wcsobj.to_header()

        # recompute pixel positions with the final (re-centred) WCS
        x, y = wcsobj.wcs_world2pix(
            catalog_model.pos.ra.degree,
            catalog_model.pos.dec.degree, 0)
        assert ((x < 0) | (x > s)).sum() == 0, \
            "sources outside segmentation image footprint (x)"
        assert ((y < 0) | (y > s)).sum() == 0, \
            "sources outside segmentation image footprint (y)"

        source_names = list(catalog_model._prop_tbl['name'])
        n_src = len(source_names)

        # ---- render one segmentation image per TolTEC array ----
        hdus_by_array = {}
        for array_name, fwhm in fwhms.items():
            radius_pix = (fwhm * segment_radius_to_fwhm_ratio).to_value(
                u.pix, equivalencies=pixscale_equiv)
            seg_img = np.zeros((s, s), dtype=np.int16)
            with timeit(f"render {n_src} circles for {array_name}"):
                for src_idx, (xx, yy) in enumerate(zip(x, y)):
                    _render_circle(
                        seg_img, xx, yy, radius_pix, src_idx + 1)
            hdu = fits.ImageHDU(seg_img, header=header)
            hdu.header['EXTNAME'] = array_name
            hdu.header['RATIO'] = (
                float(segment_radius_to_fwhm_ratio),
                'segment_radius_to_fwhm_ratio')
            hdu.header['FWHM'] = (
                float(fwhm.to_value(u.arcsec)),
                'array beam FWHM [arcsec]')
            hdu.header['RPIX'] = (float(radius_pix), 'circle radius [pix]')
            hdus_by_array[array_name] = hdu
            logger.info(
                f"segmentation {array_name}: "
                f"FWHM={fwhm:.2f}, radius={radius_pix:.1f} px, "
                f"n_src={n_src}")

        return cls(
            hdus_by_array=hdus_by_array,
            source_names=source_names,
            wcsobj=wcsobj,
        )

    # ------------------------------------------------------------------
    # Sky-trajectory look-up
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup_seg_for_hdu(hdu, det_ra, det_dec, out_mask, seg_out):
        """Look up integer seg_id values from *hdu* for one array's detectors.

        This is the integer-valued counterpart of
        :meth:`ImageSourceModel._set_data_for_hdu
        <tolteca.simu.sources.models.ImageSourceModel._set_data_for_hdu>`,
        using the same WCS bounding-box filter and nearest-pixel lookup.

        Parameters
        ----------
        hdu : `~astropy.io.fits.ImageHDU`
            Segmentation image for one array (integer pixel values).
        det_ra, det_dec : `~astropy.coordinates.Angle`
            Detector sky positions, shape ``(n_m, n_times)`` where *n_m* is
            the number of detectors in the current array subset.
        out_mask : bool array, shape ``(n_det_total,)``
            Selects the rows of *seg_out* that correspond to the passed
            detectors (one entry per detector in the full good-detector array).
        seg_out : ``int16`` array, shape ``(n_det_total, n_times)``
            Output array, modified in-place.
        """
        logger = get_logger()
        wcsobj = WCS(hdu.header).sub(2)
        hdu_data = hdu.data.reshape(wcsobj.array_shape)
        ny, nx = hdu_data.shape
        sky_bbox = SkyBoundingBox.from_wcs(wcsobj, (ny, nx))

        det_ra_deg = det_ra.wrap_at(sky_bbox.lon_wrap_angle).degree
        det_dec_deg = det_dec.degree

        # spatial bounding-box filter
        g = (
            (det_ra_deg > sky_bbox.w.degree)
            & (det_ra_deg < sky_bbox.e.degree)
            & (det_dec_deg > sky_bbox.s.degree)
            & (det_dec_deg < sky_bbox.n.degree)
        )
        logger.debug(f"seg lookup mask {g.sum()}/{det_dec.size}")
        if g.sum() == 0:
            return

        x_g, y_g = wcsobj.wcs_world2pix(det_ra_deg[g], det_dec_deg[g], 0)
        ii = np.rint(y_g).astype(int)
        jj = np.rint(x_g).astype(int)

        # pixel validity filter
        gp = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
        g[g] = gp
        x_g, y_g = wcsobj.wcs_world2pix(det_ra_deg[g], det_dec_deg[g], 0)
        ii = np.rint(y_g).astype(int)
        jj = np.rint(x_g).astype(int)

        # ig → detector axis (0..n_m-1), jg → time axis
        ig, jg = np.where(g)
        seg_out[np.flatnonzero(out_mask)[ig], jg] = hdu_data[ii, jj]

    def evaluate_seg_tod_icrs(self, det_array_name, det_ra, det_dec):
        """Return the per-sample seg_id array.

        Parameters
        ----------
        det_array_name : array-like of str, shape ``(n_det,)``
        det_ra, det_dec : `~astropy.coordinates.Angle`,
            shape ``(n_det, n_times)``

        Returns
        -------
        seg_ids : ``int16`` ndarray, shape ``(n_det, n_times)``
            0 = background; ``i+1`` = source index *i* (0-based) in the
            catalog / ``seg_sources.ecsv``.
        """
        seg_out = np.zeros(det_ra.shape, dtype=np.int16)
        for array_name, hdu in self._hdus_by_array.items():
            m = (det_array_name == array_name)
            if m.sum() == 0:
                continue
            self._lookup_seg_for_hdu(
                hdu=hdu,
                det_ra=det_ra[m],
                det_dec=det_dec[m],
                out_mask=m,
                seg_out=seg_out,
            )
        return seg_out

    def to_fits(self):
        """Return an :class:`~astropy.io.fits.HDUList` suitable for saving."""
        phdu = fits.PrimaryHDU()
        phdu.header['NSRC'] = (len(self.source_names), 'number of sources')
        for i, name in enumerate(self.source_names):
            phdu.header[f'SRC{i:04d}'] = (
                str(name)[:68], f'source name for seg_id={i + 1}')
        return fits.HDUList([phdu] + list(self._hdus_by_array.values()))


# ---------------------------------------------------------------------------
# Step configuration
# ---------------------------------------------------------------------------

@steps_registry.register('timestream_project')
@add_schema
@dataclass
class TimestreamProjectStepConfig():
    '''
    Project detector sky trajectories onto a source segmentation image.

    For every raw data sample the step records which source segment (if any) is
    being visited.  The raw data files are **not** modified; only output files
    are written to the job output directory.

    The segmentation image is rendered from the single
    ``point_source_catalog`` source supplied via *sources*.  Each source
    occupies a filled circle of radius ``fwhm * segment_radius_to_fwhm_ratio``
    (one image per TolTEC array because each array has its own beam FWHM).
    '''

    enabled: bool = field(
        default=True,
        metadata={'description': 'Enable/disable this pipeline step.'})

    jobkey: str = field(
        default='timeproj',
        metadata={'description': 'Unique identifier for this job.'})

    citlali_config: CitlaliConfig = field(
        default_factory=CitlaliConfig,
        metadata={"description": "Related citlali settings (e.g. APT)."})

    sources: list = field(
        default_factory=list,
        metadata={
            'description': (
                'Exactly one ``point_source_catalog`` source must be '
                'provided.  All other source types are rejected.'),
            'schema': list(sources_registry.item_schemas),
            'pformat_schema_type': f"[<{sources_registry.name}>, ...]",
        })

    perf_params: PerfParamsConfig = field(
        default_factory=PerfParamsConfig,
        metadata={'description': 'Performance-related parameters.'})

    segment_radius_to_fwhm_ratio: float = field(
        default=1.0,
        metadata={
            'description': (
                'Ratio of the segmentation-circle radius to the array beam '
                'FWHM.  A value of 1.0 (default) uses exactly one FWHM as '
                'the radius.  The ratio is applied independently per TolTEC '
                'array using that array\'s nominal FWHM.'
            )
        })

    debug_max_chunks: int = field(
        default=0,
        metadata={
            'description': (
                'Debug option: stop after this many time chunks and write '
                'partial results.  0 (default) means no limit.'
            )
        })

    n_consec_sample_min: int = field(
        default=122,
        metadata={
            'description': (
                'Minimum number of consecutive samples a detector must '
                'continuously observe a segment for the visit to be '
                'recorded in ``seg_visits.ecsv``.  At the default sample '
                'rate of ~122 Hz this corresponds to approximately 1 second.'
            )
        })

    def __post_init__(self):
        self.logger = get_logger()

    def __call__(self, cfg):
        return TimestreamProjectExecutor(
            jobkey=self.jobkey,
            citlali_config=self.citlali_config,
            sources=self.sources,
            perf_params=self.perf_params,
            segment_radius_to_fwhm_ratio=self.segment_radius_to_fwhm_ratio,
            debug_max_chunks=self.debug_max_chunks,
            n_consec_sample_min=self.n_consec_sample_min,
        )

    def run(self, cfg, inputs=None):
        """Run this reduction step."""
        if inputs is None:
            inputs = cfg.load_input_data()
        bods = [i for i in inputs if isinstance(i, BasicObsDataset)]
        if len(bods) == 0:
            self.logger.debug("no valid input for this step, skip")
            return None
        assert len(bods) == 1
        bods = bods[0]
        output_dir = cfg.get_or_create_output_dir()
        executor = self(cfg)
        return executor(dataset=bods, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class TimestreamProjectExecutor(SimuExecutor):
    """Execute the timestream-projection analysis.

    Inherits the following helper methods from
    :class:`~.simu.SimuExecutor` without modification:

    * ``_get_source_models``  – validate and instantiate sources
    * ``_get_mapping_model`` – load :class:`LmtTcsTrajMappingModel`
    * ``_make_time_grids``   – compute aligned simulation time grid

    The ``__call__`` entry point is overridden to skip all file-copying and
    IQ-demodulation logic; only the sky-trajectory projection is performed.
    """

    logger = get_logger()

    def __init__(
            self, jobkey, citlali_config, sources, perf_params,
            segment_radius_to_fwhm_ratio=1.0, debug_max_chunks=0,
            n_consec_sample_min=122):
        super().__init__(
            jobkey=jobkey,
            citlali_config=citlali_config,
            sources=sources,
            perf_params=perf_params,
        )
        self._segment_radius_to_fwhm_ratio = float(segment_radius_to_fwhm_ratio)
        self._debug_max_chunks = int(debug_max_chunks)
        self._n_consec_sample_min = int(n_consec_sample_min)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(self, dataset, output_dir):
        citlali_proc = CitlaliProc(citlali=None, config=self._citlali_config)
        citlali_cfg = citlali_proc._prepare_citlali_config(dataset, output_dir)
        input_items = citlali_cfg['inputs']
        proj_output_dir = self._get_or_create_proj_output_dir(
            rootpath=output_dir)

        # ---- Build segmentation model once (same for all obsnums) ----
        catalog_model = self._get_catalog_source_model()
        perf_params = self._perf_params
        fwhms = {
            array_name: toltec_info[array_name]['a_fwhm']
            for array_name in toltec_info['array_names']
        }
        seg_model = SegmentationImageModel.from_catalog_source_model(
            catalog_model=catalog_model,
            fwhms=fwhms,
            pixscale=perf_params.catalog_model_render_pixel_size / u.pix,
            segment_radius_to_fwhm_ratio=self._segment_radius_to_fwhm_ratio,
        )
        self.logger.info(
            f"segmentation model: {len(seg_model.source_names)} sources, "
            f"segment_radius_to_fwhm_ratio={self._segment_radius_to_fwhm_ratio}")

        # ---- Save static outputs (shared across all obsnums) ----
        seg_fits_path = proj_output_dir / 'seg_model.fits'
        seg_model.to_fits().writeto(str(seg_fits_path), overwrite=True)
        self.logger.info(f"saved segmentation image → {seg_fits_path}")

        seg_sources_tbl = QTable()
        seg_sources_tbl['seg_id'] = np.arange(
            1, len(seg_model.source_names) + 1, dtype=np.int16)
        seg_sources_tbl['source_name'] = seg_model.source_names
        seg_sources_tbl['ra'] = catalog_model.pos.ra.to(u.deg)
        seg_sources_tbl['dec'] = catalog_model.pos.dec.to(u.deg)
        seg_sources_path = proj_output_dir / 'seg_sources.ecsv'
        seg_sources_tbl.write(
            str(seg_sources_path), format='ascii.ecsv', overwrite=True)
        self.logger.info(f"saved source table → {seg_sources_path}")

        # ---- Save config ----
        config_filepath = proj_output_dir / 'tolteca.yaml'
        with open(config_filepath, 'w') as fo:
            config = {
                'timestream_project': {
                    'sources': [
                        {k: v for k, v in s.to_dict().items()
                         if isinstance(v, (Path, str, int, float, dict, list))}
                        for s in self._sources
                    ],
                    'perf_params': self._perf_params.to_dict(),
                    'segment_radius_to_fwhm_ratio': (
                        self._segment_radius_to_fwhm_ratio),
                    'debug_max_chunks': self._debug_max_chunks,
                    'n_consec_sample_min': self._n_consec_sample_min,
                }
            }
            RuntimeBase.yaml_dump(config, fo)

        # ---- Per-obsnum projection ----
        self.logger.info(
            f"processing {len(input_items)} obsnum(s) → {proj_output_dir}")
        for item in input_items:
            name = item['meta']['name']
            obs_output_dir = proj_output_dir / name
            obs_output_dir.mkdir(parents=True, exist_ok=True)
            self._run_timestream_project(
                item, obs_output_dir, seg_model)

        self.logger.info(
            f"timestream_project complete → {proj_output_dir}\n"
            f"  seg_model.fits       segmentation image (one ext per array)\n"
            f"  seg_sources.ecsv     source catalog with seg_id mapping\n"
            f"  tolteca.yaml         step configuration\n"
            f"  <obsnum>/            per-obsnum subdirectory with:\n"
            f"    toltecXX_seg_ids.npz  (int16, n_det × n_nc_times)\n"
            f"    seg_visits.ecsv       (n_consec_sample_min="
            f"{self._n_consec_sample_min})")
        return proj_output_dir

    # ------------------------------------------------------------------
    # Output directory (parallel to SimuExecutor.get_or_create_simu_output_dir)
    # ------------------------------------------------------------------

    def _get_or_create_proj_output_dir(self, rootpath):
        logger = get_logger()
        jobkey = self._jobkey
        existing = list(rootpath.glob(jobkey + '[0-9][0-9][0-9]'))
        if existing:
            index = max(
                int(p.name.replace(jobkey, '')) for p in existing) + 1
        else:
            index = 1
        output_dir = rootpath / f'{jobkey}{index:03d}'
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"timestream_project output dir: {output_dir}")
        return output_dir

    # ------------------------------------------------------------------
    # Source model validation
    # ------------------------------------------------------------------

    def _get_catalog_source_model(self):
        """Return the single :class:`CatalogSourceModel`; raise otherwise."""
        source_models = self._get_source_models()  # also checks power loading
        catalog_models = [
            m for m in source_models if isinstance(m, CatalogSourceModel)
        ]
        non_catalog = [
            m for m in source_models if not isinstance(m, CatalogSourceModel)
        ]
        if len(catalog_models) == 0:
            raise ValueError(
                "timestream_project requires exactly one "
                "'point_source_catalog' source; none found.")
        if len(catalog_models) > 1:
            raise ValueError(
                "timestream_project requires exactly one "
                f"'point_source_catalog' source; "
                f"{len(catalog_models)} found.")
        if non_catalog:
            raise ValueError(
                "timestream_project only supports a single "
                "'point_source_catalog' source; "
                f"unexpected models: {non_catalog}")
        return catalog_models[0]

    # ------------------------------------------------------------------
    # NW info loading – read-only, no file copying
    # ------------------------------------------------------------------

    def _load_nw_info(self, input):
        """Load per-network time-grid and kids-model info without copying files.

        Returns a dict of the same structure as the *output_by_nw* dict used
        in :meth:`~.simu.SimuExecutor._make_time_grids`, so that the
        inherited ``_make_time_grids`` class method works unchanged.
        """
        nw_info = {}
        for item in input['data_items']:
            meta = item.get('meta', None)
            if meta is None:
                continue
            interface = meta.get('interface', '')
            if not interface.startswith('toltec'):
                continue
            nw = int(interface.replace('toltec', ''))
            filepath = Path(item['filepath'])
            t_info = _load_time_grid(filepath)
            kids_model = _load_kids_model(filepath)
            nw_info[nw] = {
                'interface': interface,
                'filepath': filepath,
                'time': t_info,
                'kids_model': kids_model,
            }
        return nw_info

    # ------------------------------------------------------------------
    # Core projection loop
    # ------------------------------------------------------------------

    def _run_timestream_project(
            self, input, obs_output_dir, seg_model):
        name = input['meta']['name']
        self.logger.info(
            f"run timestream_project for {name}, "
            f"output_dir={obs_output_dir}")

        tel_filepath = Path([
            d['filepath'] for d in input['data_items']
            if d['meta']['interface'] == 'lmt'
        ][0])
        perf_params = self._perf_params

        # ---- mapping model ----
        mapping_model = self._get_mapping_model(tel_filepath)

        # ---- NW info (read-only) ----
        nw_info = self._load_nw_info(input)
        if not nw_info:
            raise ValueError("no toltec network data found in input")

        # ---- array property table ----
        apt_filepath = Path([
            d['filepath'] for d in input['cal_items']
            if d['type'] == 'array_prop_table'
        ][0])
        apt_in = Table.read(apt_filepath, format='ascii.ecsv')

        apt = Table()
        for c in ['uid', 'array', 'nw', 'fg', 'pg', 'ori', 'loc', 'flag']:
            apt[c] = apt_in[c].astype(int)

        # positional / beam columns needed for sky-projection
        apt_dispatch = [
            ('x_t',        'x_t',        u.arcsec),
            ('x_t_err',    'x_t_err',    u.arcsec),
            ('y_t',        'y_t',        u.arcsec),
            ('y_t_err',    'y_t_err',    u.arcsec),
            ('a_fwhm',     'a_fwhm',     u.arcsec),
            ('b_fwhm',     'b_fwhm',     u.arcsec),
            ('a_fwhm_err', 'a_fwhm_err', u.arcsec),
            ('b_fwhm_err', 'b_fwhm_err', u.arcsec),
            ('x_t_raw',    'x_t_raw',    u.arcsec),
            ('y_t_raw',    'y_t_raw',    u.arcsec),
            ('x_t_derot',  'x_t_derot',  u.arcsec),
            ('y_t_derot',  'y_t_derot',  u.arcsec),
            ('tone_freq',    'tone_freq',    u.Hz),
            ('kids_fr',      'kids_fr',      u.Hz),
            ('responsivity', 'responsivity', None),
            ('flxscale',     'flxscale',     None),
        ]
        for k, kk, ku in apt_dispatch:
            if kk not in apt_in.colnames:
                continue
            apt[k] = apt_in[kk] << ku if ku is not None else apt_in[kk]

        apt['array_name'] = [
            toltec_info['array_names'][a] for a in apt['array']
        ]

        # assign kids_tone: maps each apt detector row to its column index in
        # the raw nc file (tone order).  mirrors the assignment in simu.py.
        apt['kids_tone'] = np.full(len(apt), -1, dtype=int)
        for nw, d in nw_info.items():
            m = apt['nw'] == nw
            apt['kids_tone'][m] = range(len(d['kids_model'].table))

        apt_full = apt.copy()
        gmask = apt['flag'] == 0
        simu = ToltecObsSimulator(array_prop_table=apt_full[gmask])
        apt = simu.array_prop_table   # good detectors only
        self.logger.debug(
            f"using {gmask.sum()} / {len(gmask)} good detectors")

        # ---- time grids (reuse SimuExecutor class method) ----
        t_info = self._make_time_grids(
            mapping_model, nw_info, chunk_len=perf_params.chunk_len)
        t_chunks = t_info['t_chunks']

        # ---- pointing model ----
        po_az_arcsec = [0]
        po_alt_arcsec = [0]
        cal_astrometry = [
            c for c in input['cal_items'] if c['type'] == 'astrometry'
        ]
        if cal_astrometry:
            offsets = cal_astrometry[0].get('pointing_offsets', None)
            if offsets is not None:
                for cc in offsets:
                    if cc['axes_name'] == 'az':
                        po_az_arcsec = cc['value_arcsec']
                    elif cc['axes_name'] == 'alt':
                        po_alt_arcsec = cc['value_arcsec']

        def _make_po_interp(po):
            if len(po) == 1:
                return lambda x: np.full(x.shape, po[0])
            return interp1d(
                [t_chunks[0][0].to_value(u.s),
                 t_chunks[-1][-1].to_value(u.s)],
                po, kind='linear')

        po_az_interp  = _make_po_interp(po_az_arcsec)
        po_alt_interp = _make_po_interp(po_alt_arcsec)

        def pointing_model_altaz(t):
            return (
                po_az_interp(t.to_value(u.s)) << u.arcsec,
                po_alt_interp(t.to_value(u.s)) << u.arcsec,
            )

        # Pass empty source list: surface-brightness evaluation is not needed;
        # we only call the evaluator with mapping_only=True.
        mapping_evaluator, _ = simu.mapping_evaluator(
            mapping=mapping_model,
            sources=[],
            pointing_model_altaz=pointing_model_altaz,
            erfa_interp_len=perf_params.mapping_erfa_interp_len,
            eval_interp_len=perf_params.mapping_eval_interp_len,
            catalog_model_render_pixel_size=(
                perf_params.catalog_model_render_pixel_size),
        )

        # pre-evaluation pass to establish sky bounding box for lon wrap angles
        t_grid_pre_eval = np.linspace(
            t_chunks[0][0].to_value(u.s),
            t_chunks[-1][-1].to_value(u.s),
            perf_params.pre_eval_t_grid_size,
        ) << u.s
        mapping_info_pre = mapping_evaluator(t_grid_pre_eval, mapping_only=True)
        det_sky_traj_pre = mapping_info_pre['det_sky_traj']

        bbox_padding = (
            perf_params.pre_eval_sky_bbox_padding_size,
            perf_params.pre_eval_sky_bbox_padding_size,
        )
        det_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
            det_sky_traj_pre['ra'], det_sky_traj_pre['dec'],
        ).pad_with(*bbox_padding)
        det_sky_bbox_altaz = SkyBoundingBox.from_lonlat(
            det_sky_traj_pre['az'], det_sky_traj_pre['alt'],
        ).pad_with(*bbox_padding)
        self.logger.info(
            f"pre-eval sky bbox:\n"
            f"  ra:  {det_sky_bbox_icrs.w!s} – {det_sky_bbox_icrs.e!s}\n"
            f"  dec: {det_sky_bbox_icrs.s!s} – {det_sky_bbox_icrs.n!s}\n"
            f"  size: {det_sky_bbox_icrs.width}, {det_sky_bbox_icrs.height}")

        # ---- pre-allocate per-nw output arrays ----
        # Each array has shape (n_tones_in_nc, n_samples_in_nc_file) so the
        # row order matches the raw nc file column order exactly.
        # Rows for flagged-bad detectors remain 0 (background) throughout.
        seg_id_arrays  = {}
        det_uid_arrays = {}
        t_i0_by_nw     = {}
        for nw, d in nw_info.items():
            n_tones    = len(d['kids_model'].table)
            n_times_nc = len(d['time']['t_grid'])
            seg_id_arrays[nw] = np.zeros((n_tones, n_times_nc), dtype=np.int16)
            # uid array indexed by nc tone position; -1 for unmapped tones
            uids = np.full(n_tones, -1, dtype=np.int64)
            m_full = apt_full['nw'] == nw
            for ti, uid in zip(
                    apt_full['kids_tone'][m_full],
                    apt_full['uid'][m_full]):
                if ti >= 0:
                    uids[int(ti)] = int(uid)
            det_uid_arrays[nw] = uids

        # ---- main projection loop ----
        n_chunks = len(t_chunks)
        if self._debug_max_chunks > 0:
            self.logger.warning(
                f"debug_max_chunks={self._debug_max_chunks}: "
                f"only processing first {self._debug_max_chunks} of "
                f"{n_chunks} chunks")
        with timeit("timestream projection"):
            for ci, t in enumerate(t_chunks):
                self.logger.info(
                    f"project chunk {ci + 1}/{n_chunks} "
                    f"t_min={t.min()} t_max={t.max()}")

                mapping_info = mapping_evaluator(
                    t,
                    mapping_only=True,
                    lon_wrap_angle_altaz=det_sky_bbox_altaz.lon_wrap_angle,
                    lon_wrap_angle_icrs=det_sky_bbox_icrs.lon_wrap_angle,
                )
                det_sky_traj = mapping_info['det_sky_traj']
                det_ra  = det_sky_traj['ra']    # (n_good_det, n_t)
                det_dec = det_sky_traj['dec']   # (n_good_det, n_t)

                # segmentation look-up for all detectors × time samples
                det_seg_ids = seg_model.evaluate_seg_tod_icrs(
                    apt['array_name'], det_ra, det_dec)
                # shape: (n_good_det, n_t), dtype int16

                for nw, d in nw_info.items():
                    fsmp  = d['time']['fsmp'] << u.Hz
                    nw_t0 = Time(d['time']['t0_grid'], format='unix')
                    nw_i0 = int(np.round(
                        ((t[0] + mapping_model.t0 - nw_t0) * fsmp
                         ).to_value(u.dimensionless_unscaled)))
                    nw_i1 = nw_i0 + len(t)

                    # record the first i0 seen (chunks are contiguous)
                    if nw not in t_i0_by_nw:
                        t_i0_by_nw[nw] = nw_i0

                    m_apt = apt['nw'] == nw   # mask within good-detector apt
                    # scatter into the nc-file tone positions
                    tone_indices = apt['kids_tone'][m_apt]
                    seg_id_arrays[nw][tone_indices, nw_i0:nw_i1] = (
                        det_seg_ids[m_apt, :])

                    n_nonbg = int((det_seg_ids[m_apt, :] > 0).sum())
                    self.logger.debug(
                        f"nw={nw} chunk [{nw_i0}:{nw_i1}] "
                        f"non-background samples: {n_nonbg}")

                if self._debug_max_chunks > 0 and ci + 1 >= self._debug_max_chunks:
                    self.logger.warning(
                        f"debug_max_chunks={self._debug_max_chunks} reached, "
                        f"stopping early after chunk {ci + 1}/{n_chunks}")
                    break

        # ---- save per-nw npz files ----
        for nw, seg_ids in seg_id_arrays.items():
            npz_path = obs_output_dir / f'toltec{nw:02d}_seg_ids.npz'
            np.savez_compressed(
                str(npz_path),
                seg_ids=seg_ids,
                det_uids=det_uid_arrays[nw],
                t_i0=np.int64(t_i0_by_nw.get(nw, 0)),
                fsmp=np.float64(nw_info[nw]['time']['fsmp']),
                t0_grid=np.float64(nw_info[nw]['time']['t0_grid']),
            )
            n_nonbg = int((seg_ids > 0).sum())
            self.logger.info(
                f"saved {npz_path.name}: "
                f"shape={seg_ids.shape}  non-background samples={n_nonbg}")

        # ---- build and save seg_visits table ----
        visit_tables = []
        for nw, seg_ids in seg_id_arrays.items():
            t_i0_nw = int(t_i0_by_nw.get(nw, 0))
            fsmp_nw = float(nw_info[nw]['time']['fsmp'])
            tbl = _find_all_seg_visits(
                seg_ids=seg_ids,
                min_samples=self._n_consec_sample_min,
                det_uids=det_uid_arrays[nw],
                t_i0=t_i0_nw,
                fsmp=fsmp_nw,
            )
            tbl.add_column(
                np.full(len(tbl), nw, dtype=np.int16),
                name='nw', index=0)
            visit_tables.append(tbl)
            self.logger.info(
                f"nw={nw}: {len(tbl)} visits "
                f"(n_consec_sample_min={self._n_consec_sample_min})")

        from astropy.table import vstack as _vstack
        all_visits = (_vstack(visit_tables)
                      if any(len(t) for t in visit_tables)
                      else visit_tables[0] if visit_tables else QTable())
        visits_path = obs_output_dir / 'seg_visits.ecsv'
        all_visits.write(str(visits_path), format='ascii.ecsv', overwrite=True)
        self.logger.info(
            f"[{name}] saved seg_visits.ecsv: "
            f"{len(all_visits)} visits across {len(seg_id_arrays)} NWs "
            f"(n_consec_sample_min={self._n_consec_sample_min})")
