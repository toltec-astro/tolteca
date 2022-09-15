import numpy as np
import pandas as pd


class OverheadCalculation:
    def __init__(
        self, map_type, science_time, n_steps=None, add_nights=0, t_pattern=None
    ):
        """
        map_type (string): raster, lissajous, rastajous, double_lissajous

        science_time (float): Scieence observation time to achieve desired depth
            units = s

        n_steps (int): Must be specified for raster and rastajous

        add_nights (int): number of additional (> 1) night observation will be
            split into

        t_pattern (float): how long is takes to complete 1 pass of the pattern
            units = s

        The map_type determines if there is turnaround time in the map that
        should be accounted for. The science time determines how many pointings
        are necessary. For raster and rastajous maps, the number of steps is
        used to determine what fraction of the time is spent turning around.
        """
        self._total_time = None
        self._overhead_time = None
        self._overhead_percent = None
        self._science_overhead = None
        self._integration_time = None
        self._tot_pointing_time = None
        self._n_pointings = None
        self._add_nights = add_nights
        # TODO: add beammap to track separately - 1 at the start of several observations?
        # part of instrumental setup?
        # These only become relevant if map type is raster or rastajous
        self._n_steps = n_steps
        self._t_pattern = t_pattern

        # Estimate based on (tune = 30s) + (pointing obs = 60s) + (slew max = 20s)
        self._pointing_time = 110  # s
        # TODO: check number with BL Lac observation
        self._t_turnaround = 5  # s
        # Placeholder in case steps require observer actions to proceed
        self._observer_time = 0  # s
        # Used to get tune time at the start of each science observation
        self._tune_time = 30  # s

        self._map_type = map_type
        # Science observation time from obsplanner (includes turnaround overhead)
        self._science_time = science_time

        # Calculate all class variables
        self._update(self._add_nights)

    def make_output_dict(self):
        """
        Returns dictionary with variables calculated by class.
        """
        output_dict = {
            "total_time": self._total_time,
            "overhead_time": self._overhead_time,
            "overhead_percent": self._overhead_percent,
            "science_overhead": self._science_overhead,
            "integration_time": self._integration_time,
            "tot_pointing_time": self._tot_pointing_time,
            "n_pointings": self._n_pointings,
            "pointing_time": self._pointing_time,
        }

        return output_dict

    def set_pointing_time(self, new_time):
        """
        The pointing time is initialized to 110s based on estimates from past
        observations. This is meant to include the pointing observation time,
        tune time, and slew time needed for every pointing observations.
        The slew assumes the point source is the max recommended distance
        (10 deg) from the map, taking 20s round trip to reach moving 1 deg/s.
        A new time can be set to more accurately reflect the observational
        setup.
        """
        self._pointing_time = new_time
        self._update(self._add_nights)

    def add_nights(self, n_nights):
        """
        Each night the observation is split into adds 1 pointing. n_nights is
        the number of nights beyond the first, which is included in the
        calcNPointings() calculation.

        ex. 3 night observation, n_nights = 2
        """
        self._n_pointings += n_nights
        self._update()

    def _update(self, add_nights=0):
        # Get the number of pointings and optionally add more for additional nights
        # calc_n_pointings also calculates the number of science observations between
        # pointing observations
        if self._n_pointings is None:
            self._calc_n_pointings()
        if add_nights > 0:
            self.add_nights(add_nights)
        self._calc_integration_time()
        self._calc_overhead_time()
        self._calc_tot_time()
        self._calc_overhead_percent()

    def _calc_overhead_percent(self):
        """
        Calculates the total overhead time from the science observation, pointing
        observation, tune, and slew and divides it by the total observation time.
        """
        if self._overhead_time is None:
            self._calc_overhead_time()
        if self._total_time is None:
            self._calc_tot_time()
        if self._total_time > 0:
            self._overhead_percent = self._overhead_time / self._total_time

    def _calc_overhead_time(self):
        """
        Calculates the science overhead based on the map type and adding in the
        time spent on pointings, tunes, slew (wrapped up in pointing_time), and
        observer time.
        """
        if self._science_overhead is None:
            self._calc_science_overhead()

        # also includes the tune at the start of each science observation
        self._tot_pointing_time = (
            self._n_pointings * self._pointing_time
            + self.n_science_obs * self._tune_time
        )
        self._overhead_time = (
            self._science_overhead + self._tot_pointing_time + self._observer_time
        )

    def _calc_integration_time(self):
        """
        Subtracts the calculated overhead time from the science observation from
        the total science obervation time to get the integration time spent on the
        source.
        """
        if self._science_overhead is None:
            self._calc_science_overhead()
        self._integration_time = self._science_time - self._science_overhead

    def _calc_tot_time(self):
        """
        Adds the pointing observations, tunes, slew time, and observer time to
        the science time.
        """
        self._total_time = (
            self._science_time + self._tot_pointing_time + self._observer_time
        )

    def _calc_science_overhead(self):
        """
        Lissajous and double lissajous maps have a 100% observing effieciency,
        so their science observation overhead is 0. Raster and rastajous maps
        include a turnaround for each step. Their science overhead is the
        number of steps times the amount of time it takes to turn around. Since
        the pattern may be repeated multiple times, the fraction of overhead
        per pass is multiplied by the total map time.
        """
        if self._map_type == "lissajous" or self._map_type == "double_lissajous":
            self._science_overhead = 0
        else:
            if self._t_pattern is None or self._n_steps is None:
                raise ValueError(
                    "t_pattern/n_steps is required to calcualte science overhead"
                )
            if self._t_pattern > 0:
                overhead_per_pass = self._t_turnaround * self._n_steps
                percent_overhead = overhead_per_pass / self._t_pattern
                self._science_overhead = percent_overhead * self._science_time
            else:
                self._science_overhead = 0

    def _calc_n_pointings(self):
        """
        Convert science observation time to hours, round to next largest whole
        hour, add 1 to get number of pointing observations (assuming 1 night
        of observing).

        Want pointing observations at the start, end, and every hour of observing.
        """
        self.n_science_obs = np.ceil(self._science_time / 3600)
        self._n_pointings = self.n_science_obs + 1
