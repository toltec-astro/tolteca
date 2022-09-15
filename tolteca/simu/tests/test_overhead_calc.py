import pytest
from toltec_overhead import OverheadCalculation 

class TestOverheadCalc:
    def test_overhead_no_science(self):
        # science map time = 0, would just do pointing and be done
        calc = OverheadCalculation(map_type='raster', science_time=0, n_steps=0, add_nights=0, t_pattern=0)
        output = calc.make_output_dict()
        assert output['total_time'] == output['pointing_time'] 
        assert output['overhead_time'] == output['pointing_time'] 
        assert output['overhead_percent'] == 1.
        assert output['science_overhead'] == 0
        assert output['integration_time'] == 0
        assert output['tot_pointing_time'] == output['pointing_time']
        assert output['n_pointings'] == 1
    
    def test_liss_one_hour(self):
        # science overhead hsould be 0, 2 pointings (1 before, 1 after)
        calc = OverheadCalculation(map_type='lissajous', science_time=3600)
        output = calc.make_output_dict()
        assert output['science_overhead'] == 0
        assert output['integration_time'] == 3600.
        assert output['n_pointings'] == 2
    
    def test_double_liss_two_hour(self):
        # science overhead should be 0, 3 pointings 
        calc = OverheadCalculation(map_type='double_lissajous', science_time=2*3600)
        output = calc.make_output_dict()
        assert output['science_overhead'] == 0
        assert output['integration_time'] == 2*3600.
        assert output['n_pointings'] == 3
    
    def test_raster_two_nights(self):
        #  1 deg, 8 arcmin/s, step = 0.5 arcmin, 120 steps
        calc = OverheadCalculation(map_type='raster', science_time=2*3600, add_nights=1, 
                                t_pattern = 1495, n_steps = 120)
        output = calc.make_output_dict()
        assert output['science_overhead'] != 0
        assert output['integration_time'] < 2*3600.
        assert output['n_pointings'] == 4
    
    def test_set_pointing_add_nights(self):
        # science map time = 0, would just do pointing and be done

        # just the build-in pointing time (110) 
        calc = OverheadCalculation(map_type='raster', science_time=0, n_steps=0, add_nights=0, t_pattern=0)
        output = calc.make_output_dict()
        assert output['tot_pointing_time'] == 110.

        # set pointing time (120) 
        calc.set_pointing_time(120)
        output = calc.make_output_dict()
        assert output['tot_pointing_time'] == 120.

        # new pointing time (120) for 3 nights
        calc.add_nights(2)
        output = calc.make_output_dict()
        assert output['tot_pointing_time'] == 120.*3

        # make sure new calc object resets everything
        calc = OverheadCalculation(map_type='raster', science_time=0, n_steps=0, add_nights=0, t_pattern=0)
        output = calc.make_output_dict()
        assert output['tot_pointing_time'] == 110.

