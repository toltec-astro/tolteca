#! /usr/bin/env python


"""This module helps setup and run the reduction pipeline."""


from ..utils import touch_file
from ..utils.log import get_logger, timeit
from datetime import datetime
import os


class PipelineRuntime(object):

    _workdir_contents = {
            'logdir': 'logs',
            'externdir': 'extern',
            'configfile': 'config_base.yaml'
            }
    _backup_time_fmt = "%Y_%m_%d_%H_%M_%S"

    def __init__(self, workdir):
        self.workdir = workdir

    def __getattr__(self, name, *args):
        if name in self._workdir_contents:
            return self._get_content_path(self.workdir, name)
        return super().__getattribute__(name, *args)

    @classmethod
    def _get_content_path(cls, root, item):
        return root.joinpath(cls._workdir_contents[item])

    @classmethod
    def from_dir(cls, path, backup=True, create=True):
        logger = get_logger()
        items_to_backup = ['configfile', ]

        for item in items_to_backup:
            content_path = cls._get_content_path(path, item)
            if content_path.exists():
                if backup:
                    timestamp = datetime.fromtimestamp(
                        content_path.lstat().st_mtime).strftime(
                            cls._backup_time_fmt)
                    backup_path = content_path.with_name(
                            f"{content_path.stem}_{timestamp}"
                            f"{content_path.suffix}"
                            )
                    logger.warning(f"backup existing {item} to {backup_path}")
                    os.rename(content_path, backup_path)
                else:
                    logger.warning(
                            f"overwrite existing {item} {content_path}")

        def create_item_path(item, path):
            if item.endswith('dir'):
                path.mkdir(parents=True, exist_ok=True)
            elif item.endswith('file'):
                touch_file(path)
            else:
                pass

        for item in cls._workdir_contents.keys():
            content_path = cls._get_content_path(path, item)
            if not create and not content_path.exists():
                raise RuntimeError(
                        f"unable to create pipeline runtime from {dir}:"
                        f" missing item {item} {content_path.name}")
            if create:
                if not content_path.exists():
                    logger.info(f"create {item} {content_path}")
                create_item_path(item, content_path)

        return cls(path)


@timeit
def setup_workdir(dirpath, empty_only=True, backup=True, **kwargs):
    """
    Setup execution environment and dump default configuration file
    Parameters
    ----------
    dirpath: pathlib.Path
        The dir to store the created dirs and files.
    empty_only: bool
        When set to True, raise RuntimeError if the workdir is not empty
    backup: bool
        When not empty_only, this controls whether or not to create
        backups for existing files.
    kwargs: dict
        Entries that are passed directly into the created configuration file.
    """

    logger = get_logger()

    logger.info(f"setup {dirpath} as workdir")

    pipeline_runtime = None

    # validate workdir
    if dirpath.exists():
        try:
            next(dirpath.iterdir())
        except StopIteration:
            pass
        else:
            # dirpath is not empty
            if empty_only:
                raise RuntimeError(
                        f"{dirpath} is not empty "
                        f"but empty_only is set to {empty_only}")
            logger.warning(
                    f"use non-empty workdir {dirpath}")
            try:
                pipeline_runtime = PipelineRuntime.from_dir(
                        dirpath, backup=backup)
            except Exception as e:
                raise RuntimeError(
                    f"unable to create pipeline runtime from {dirpath}: {e}")

    return
    # external dependencies
    externdir = os.path.join(workdir, externdir)
    which_path = os.path.abspath(externdir) + ':' + os.environ['PATH']
    if os.path.isdir(externdir):
        logger.warning("use existing extern dir {}".format(externdir))
    else:
        os.makedirs(externdir)
        logger.info("create extern dir {}".format(externdir))
    logger.info("check external dependencies")
    astromatic_prefix = []
    for name, cmds, datadir in [
            ("SExtractor", ("sex", 'ldactoasc'), "sextractor"),
            ("SCAMP", ("scamp", ), "scamp"),
            ("SWarp", ("swarp", ), "swarp")]:
        cmds = [which(cmd, path=which_path) for cmd in cmds]
        if any(c is None for c in cmds):
            raise RuntimeError("not able to locate {}"
                               .format(name))
        prefix = os.path.normpath(
                os.path.join(os.path.dirname(cmds[0]), '../'))
        datadir = os.path.join(prefix, "share", datadir)
        if not os.path.exists(datadir):
            raise RuntimeError(
                "not able to locate data files for {0}. It is likely that {0} "
                "is compiled within the source directory but without proper "
                "installation. To resolve the issue, either run `make install`"
                " in the source directory, or manually link the data "
                "directory to {1}.".format(name, datadir))
        logger.info("{0:10s} ... OK".format(name))
        astromatic_prefix.append(prefix)
    if len(set(astromatic_prefix)) > 1:
        raise RuntimeError(
            "it seems that the SExtractor, SCAMP and SWarp are not installed "
            "into the same prefix. {app_name} for now does not deal with this "
            "situation. Try to re-configure SExtractor, SCAMP and SWarp with "
            "--prefix=<prefixpath>".format(app_name=APP_NAME))
    astromatic_prefix = os.path.normpath(astromatic_prefix[0])
    logger.info("use shared astromatic prefix {}".format(
        astromatic_prefix))
    stilts_cmd = which("stilts", path=which_path)
    if stilts_cmd is None:
        logger.warning("not able to find stilts. Get from internet")
        # retrieve stilts
        from astropy.utils.data import download_file
        stilts_jar_tmp = download_file(
                "http://www.star.bris.ac.uk/%7Embt/stilts/stilts.jar",
                cache=True)
        stilts_jar = os.path.join(externdir, 'stilts.jar')
        shutil.copyfile(stilts_jar_tmp, stilts_jar)
        stilts_cmd = os.path.join(externdir, 'stilts')
        with open(stilts_cmd, 'w') as fo:
            fo.write("""#!/bin/sh
java -Xmx4000M -classpath "{0}:$CLASSPATH" uk.ac.starlink.ttools.Stilts "$@"
""".format(os.path.abspath(stilts_jar)))
        os.chmod(stilts_cmd, os.stat(stilts_cmd).st_mode | stat.S_IEXEC)
    logger.info("{0:10s} ... OK".format("stilts"))
    logger.info("use stilts {}".format(stilts_cmd))
    funpack_cmd = which("funpack", path=which_path)
    if funpack_cmd is None:
        logger.warning("not able to find funpack. Get from internet")
        # retrieve stilts
        from astropy.utils.data import download_file
        funpack_tmp = download_file(
                "http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/"
                "cfitsio_latest.tar.gz",
                cache=True)
        funpack_src = os.path.join(externdir, 'cfitsio')
        import tarfile
        with tarfile.open(funpack_tmp) as tar:
            tar.extractall(path=externdir)
        logger.warning("try compiling funpack")
        import subprocess
        try:
            for command in [
                    './configure', 'make', 'make fpack', 'make funpack']:
                subprocess.check_call(command.split(), cwd=funpack_src)
        except subprocess.CalledProcessError:
            raise RuntimeError("unable to compile funpack")
        funpack_cmd = os.path.join(externdir, 'funpack')
        shutil.copy(
                os.path.join(externdir, 'cfitsio', 'funpack'),
                funpack_cmd)
    logger.info("{0:10s} ... OK".format("funpack"))
    logger.info("use funpack {}".format(funpack_cmd))

    # create logging directory
    logdir = os.path.join(workdir, logdir)
    if os.path.isdir(logdir):
        logger.warning("use existing log dir {}".format(logdir))
    else:
        os.makedirs(logdir)
        logger.info("create log dir {}".format(logdir))

    # setup scratch space
    tmpdir = os.path.join(workdir, tmpdir)
    # def freespace_GiB(path):
    #     stat = os.statvfs(path)
    #     return stat.f_bfree * stat.f_frsize / 1024 ** 3

    # logger.info("{:.2f} GiB free in {}".format(
    #     freespace_GiB(scratch_dir), scratch_dir))
    if os.path.isdir(tmpdir):
        logger.warning("use existing tmp dir {}".format(tmpdir))
    else:
        os.makedirs(tmpdir)
        logger.info("create tmp dir {}".format(tmpdir))

    # dump default config
    time_fmt = "%b-%d-%Y_%H-%M-%S"
    base_fmt = ("{obsid}_{object}_{instru}_{band}")
    config = """## config file of {app_name:s}
## {time:s}
# calib setting
phot_model_flags: 'color,chip,expo'
# qa inputs
qa_headers:
    odi: [
        'OBSID', 'CRVAL1', 'CRVAL2', 'OBJECT', 'EXPMEAS', 'AIRMASS',
        'SEEING', 'SKY_MEDI', 'SKY_STD',
        'FILTER', 'INSTRUME', 'MJD-OBS'
        ]
    decam: [
        'OBSID', 'CRVAL1', 'CRVAL2', 'OBJECT', 'EXPTIME', 'AIRMASS',
        'FWHM',  'AVSKY', 'SKYSIGMA',
        'FILTER', 'INSTRUME', 'MJD-OBS'
    ]
# naming
reg_arch: '{reg_arch:s}'  # regex to parse the filenames from data archive
fmt_orig: '{{imflag}}{base_fmt}.{{ext}}'
reg_orig: '{reg_orig:s}'  # regex to parse the filenames in job-in dir
reg_inputs: '{reg_inputs:s}'  # regex to parse the filenames in jobdir
fmt_inputs: 'orig_{base_fmt}.{{ext}}'  # format string for image in jobdir
sel_inputs: ['orig_*_?.fits', 'orig_*_?.fits.fz']
fmt_masked: 'masked_{base_fmt:s}.fits'  # format string of masked images
sel_masked: 'masked_*_?.fits'   # selection string of masked images
fmt_selected: '{{ppflag}}{{imflag}}_{base_fmt:s}.fits'
sel_fcomb: 'fcomb[0-9]*_masked_*.fits'
fmt_objcat: '{{ppflag}}objcat_{base_fmt:s}.cat'  # format of object catalog
fmt_objmask: '{{ppflag}}objmask_{base_fmt:s}.fits'  # format of object masks
sel_objmask: 'fcomb[0-9]*_objmask_*.fits'   # selection string of object masks
fmt_sky: '{{ppflag}}sky_{base_fmt:s}.fits'  # format string of sky images
fmt_fcomb: '{{ppflag}}combined.fits'
reg_fcomb: '{reg_fcomb:s}'  # regex to parse grouped master
fmt_fsmooth: '{{ppflag}}smoothed.fits'
sel_fsub: 'fsub[0-9]*_masked_*.fits'
fmt_fsub_fsmooth: 'fcomb{{grpid}}_smoothed.fits'  # fmt for subtract fcomb
fmt_fsub: 'fsub_{base_fmt:s}.fits'
sel_fsubed: 'fsub_*.fits'
sel_phot: 'phot[0-9]*_*_*.fits'
fmt_photcat: '{{ppflag}}{{imflag}}_{base_fmt:s}.cat'
fmt_photcat_cleaned: '{{ppflag}}{{imflag}}_{base_fmt:s}.cln.cat'
fmt_photcat_matched: '{{ppflag}}{{imflag}}_{base_fmt:s}.zp.cat'
fmt_phot: '{{ppflag}}{{instru}}_{{band}}.cat'
reg_phot: '{reg_phot:s}'  # regex to parse grouped phot master
phot_hdr_suffix: 'hdr_phot'
phot_hdr_glob: 'phot[0-9]*_*{{obsid}}*{{instru}}_{{band}}.hdr_phot'
sel_mosaic: 'mosaic[0-9]*_*_*.fits'
fmt_mosaic_orig: 'swarp{{grpid}}_{{imflag}}_{{instru}}_{{band}}.fits'
fmt_mosaic_hdr: 'coadd{{grpid}}.mosaic'
fmt_mosaic: 'coadd{{grpid}}_{{imflag}}_{{instru}}_{{band}}.fits'
fmt_mosaic_wht: 'coadd{{grpid}}_{{imflag}}_{{instru}}_{{band}}.wht.fits'
reg_mosaic: {reg_mosaic:s}
reg_mosaic_fits: {reg_mosaic_fits:s}
fmt_msccat_matched: 'coadd{{grpid}}{{imflag}}_{{instru}}_{{band}}.zp.cat'
reg_grp: '{reg_grp:s}'  # regex to parse grouped images
fmt_grp: '{{ppflag}}{{grpid}}_{{imflag}}_{base_fmt:s}.fits'
# environ
workdir: {workdir}
tmpdir: {tmpdir}
logdir: {logdir}
astromatic_prefix: {astromatic_prefix}
stilts_cmd: {stilts_cmd}
funpack_cmd: {funpack_cmd}
""".format(app_name=APP_NAME,
           time=datetime.now().strftime(time_fmt),
           version="0.0",
           reg_arch=r'(?P<imflag>[^_/]+_)?'
                    r'(?P<obsid>[^_/]*20\d{6}[Tt]\d{6}(?:\.\d)?)'
                    r'_(?P<object>.+?)'
                    r'_(?P<instru>odi|decam)_(?P<band>[ugrizY])(?:_.+)?\.'
                    r'(?P<ext>fits|fits\.fz)$',
           reg_orig=r'(?P<ppflag>[^_/]+_)?(?P<imflag>[^_/]+)_'
                    r'(?P<obsid>[^_/]*20\d{6}[Tt]\d{6}(?:\.\d)?)'
                    r'_(?P<object>.+?)'
                    r'_(?P<instru>odi|decam)_(?P<band>[ugrizY])'
                    r'\.(?P<ext>fits|fits\.fz)$',
           reg_inputs=r'(?P<ppflag>[^_/]+_)?(?P<imflag>[^_/]+)_'
                    r'(?P<obsid>[^_/]*20\d{6}[Tt]\d{6}(?:\.\d)?)'
                    r'_(?P<object>.+?)'
                    r'_(?P<instru>odi|decam)_(?P<band>[ugrizY])'
                    r'\.(?P<ext>[^/]+)$',
           reg_fcomb=r'(?P<ppflag>[^_/]+_)(?P<imflag>[^/]+)\.fits',
           reg_phot=r'(?P<ppflag>[^_/]+)'
                    r'_(?P<instru>odi|decam)_(?P<band>[ugrizY])'
                    r'\.(?P<ext>[^/]+)',
           reg_mosaic=r'(?P<ppflag>[a-z]+)(?P<grpid>\d+)_(?P<imflag>[^_/]+)'
                    r'_(?P<instru>odi|decam)_(?P<band>[ugrizY])'
                    r'\.(?P<ext>[^/]+)$',
           reg_mosaic_fits=r'(?P<ppflag>[a-z]+)(?P<grpid>\d+)'
                           r'_(?P<imflag>[^_/]+)'
                           r'_(?P<instru>odi|decam)_(?P<band>[ugrizY])'
                           r'\.fits$',
           reg_grp=r'(?P<ppflag>[a-z]+)(?P<grpid>\d+)_(?P<imflag>[^_/]+)_'
                    r'(?P<obsid>[^_/]*20\d{6}[Tt]\d{6}(?:\.\d)?)'
                    r'_(?P<object>.+?)'
                    r'_(?P<instru>odi|decam)_(?P<band>[ugrizY])'
                    r'\.(?P<ext>[^/]+)$',
           # reg_inputs=r'(?P<ppflag>[^_/]+_)?(?P<imflag>[^_/]+)_'
           #            r'(?P<obsid>20\d{6}T\d{6}\.\d)_(?P<object>.+?)'
           #            r'_odi_(?P<band>[ugriz])'
           #            r'_(?P<featgrp>\d+)_(?P<photgrp>\d+)_(?P<mscgrp>\d+)'
           #            r'\.(?P<ext>fits|fits\.fz)$',
           **locals())
    configfile = os.path.join(workdir, configfile)
    if os.path.exists(configfile):
        if backup_config:
            timestamp = datetime.fromtimestamp(
                os.path.getmtime(configfile)).strftime(time_fmt)
            bakfile = os.path.join(
                workdir,
                "{1}_{0}{2}".format(timestamp, *os.path.splitext(
                    os.path.basename(configfile)))
                )
            logger.warning("backup existing config file to {}".format(
                bakfile))
            os.rename(configfile, bakfile)
        else:
            logger.warning(
                    "overwrite existing config file {}".format(configfile))
    with open(configfile, 'w') as fo:
        fo.write(config)
