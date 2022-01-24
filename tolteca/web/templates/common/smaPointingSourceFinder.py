from astropy.coordinates import SkyCoord
from bs4 import BeautifulSoup as bs
from datetime import datetime, timezone
from datetime import date
import astropy.units as u
import numpy as np
import requests
import json
import csv


class SMAPointingCatalog(object):

    def __init__(self, filepath):
        self._filepath = filepath

    @property
    def filepath(self):
        return self._filepath

    # This function scrapes the SMA calibrator website and returns a list
    # of potential pointing sources.  In order to avoid hitting the SMA
    # site too often, this code writes the output list of dictionaries to
    # an ascii cvs file which is hard-coded to be smaPointingSources.csv.
    # Inputs - none
    # Returns - a list of pointing source dictionaries
    # Outputs - a csv file containing the pointing source data
    def updateSMAPointingList(self):
        # The file to be updated
        psList = self.filepath
        # The SMA pointing source website
        http = "http://sma1.sma.hawaii.edu/callist/callist.html"
        date_pulled = str(date.today())

        # Use requests for the scraping
        page = requests.get(http).content

        # And use Beautiful Soup for the parsing
        s = bs(page, 'html.parser')
        rr = s.find_all('tr', align="center")

        # loop through the <tr> tags of rr and find the names of all source
        pointingSources = []
        for r in rr:
            # Here's the issue. Sometimes, r will be the beginning of a source
            # and sometimes r will be the continuation of the flux
            # measurements of a source.  We can differentiate these by the
            # first tag in the element.
            tds = r.find_all('td')
            if(tds[0].has_attr('rowspan')):
                # This is a new source
                ra = str(tds[2].text.strip())
                dec = str(tds[3].text.strip())
                source = {
                    'commonName': str(tds[0].text.replace('\xa0','').strip()),
                    'name': str(tds[1].text.strip()),
                    'ra':   ra,
                    'dec':  dec,
                    'freq': [str(tds[4].text.strip())],
                    'date': [str(tds[5].text.strip())],
                    'obs':  [str(tds[6].text.strip())],
                    'flux': [str(tds[7].text.strip())],
                }
                pointingSources.append(source)
            else:
                # This is a source with multiple flux measurements
                pointingSources[-1]['freq'].append(str(tds[0].text.strip()))
                pointingSources[-1]['date'].append(str(tds[1].text.strip()))
                pointingSources[-1]['obs'].append(str(tds[2].text.strip()))
                pointingSources[-1]['flux'].append(str(tds[3].text.strip()))

        # write the list of pointing sources to a file
        oF = open(psList, "w")
        oF.write("# SMA Pointing Sources\n")
        oF.write("# from "+http+"\n")
        oF.write("# "+date_pulled+"\n")
        oF.write("#")
        dict_writer = csv.DictWriter(oF, pointingSources[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(pointingSources)
        oF.close()

        return pointingSources


    # This function reads the csv file created by updateSMAPointingList,
    # checks the date since the SMA site was last scraped, and regenerates
    # the csv file if it's more than 30 days old.
    # Inputs - none
    # Returns - a list of pointing source dictionaries
    def readSMAPointingList(self):
        psList = self.filepath
        pointingSources = []
        psFile = open(psList, "r")
        l1 = psFile.readline().strip()
        l2 = psFile.readline().strip()
        l3 = psFile.readline().replace("# ", "").strip()
        d = datetime.strptime(l3, "%Y-%m-%d").date()
        t = date.today()
        delta = t-d
        # check if we need an update
        if(delta.days > 30):
            print("Pointing source list last scraped on "+str(d))
            print("This is more than 30 days.  Updating data.")
            ps = self.updateSMAPointingList()
            psFile.close()
            return ps
        
        # otherwise, continue reading what we've got
        dr = csv.DictReader(psFile)
        for row in dr:
            pointingSources.append(row)
        psFile.close()
        return pointingSources


    # This function returns the brightest point source from the SMA
    # pointing source catalog within radius degrees.
    # Inputs:
    #   target - a SkyCoord object
    #   radius - the initial search radius
    # Returns: a pointing source dictionary for the closest/brightest source
    def findPointingSource(self, target, radius=1.):
        pointingSources = self.readSMAPointingList()
        txtcoords = []
        # convert pointing source coords to SkyCoord
        for ps in pointingSources:
            txtcoords.append(ps['ra']+ps['dec'])
        catalog = SkyCoord(txtcoords,
                           unit=(u.hourangle, u.deg),
                           frame='icrs')

        # find the sources within radius degrees, widen the search if
        # nothing is found
        w = []
        sep = catalog.separation(target)
        while(len(w)==0):
            w = np.where(sep < radius*u.deg)[0]
            radius *= 2.
        print("{} sources found within {} deg.".format(len(w), radius/2.))

        # if only one pointing source is found, we're done
        if(len(w)==1):
            return pointingSources[w[0]]

        # otherwise, find the one with the highest 1mm flux
        plusminus = u"\u00B1"
        flux = []
        for i in np.arange(len(w)):
            ps = pointingSources[w[i]]
            psf = ps['flux']
            psf = psf.strip('][').split(', ')[0]
            psf = psf.replace("'", "")
            flux.append(float(psf.split(plusminus)[0]))
        flux = np.array(flux)
        wm = np.where(flux == flux.max())[0]
        return pointingSources[w[wm[0]]]



    def generateOTScript(self, pointSource, outFile="pointing.scr"):
        ra = pointSource['ra']
        dec = pointSource['dec']
        sname = pointSource['name']
        otxt = "ObsGoal DCS; Dcs -ObsGoal Pointing\n"
        stxt = "Source Source;  Source  -BaselineList [] -CoordSys Eq -DecProperMotionCor 0 -Dec[0] {0:} -Dec[1] {0:} -El[0] 0.000000 -El[1] 0.000000 -EphemerisTrackOn 0 -Epoch 2000.0 -GoToZenith 1 -L[0] 0.0 -L[1] 0.0 -LineList [] -Planet None -RaProperMotionCor 0 -Ra[0] {1:} -Ra[1] {1:} -SourceName \"{2:}\" -VelSys Lsr -Velocity 0.000000 -Vmag 0.0\n".format(dec, ra, sname)
        ltxt = "Lissajous -ExecMode 0 -RotateWithElevation 0 -TunePeriod 0 -TScan 120 -ScanRate 58.21155955421865 -XLength 0.5 -YLength 0.5 -XOmega 9.2 -YOmega 8.0 -XDelta 0.7853981633974483 -XLengthMinor 0.0 -YLengthMinor 0.0 -XDeltaMinor 0.0"
        # head = "# LMT OT script created by smaPointingSourceFinder.py\n" + 'ObsGoal pointing; Dcs -ObsGoal "example_simu"\n'
        timestamp = datetime.now(timezone.utc).strftime('%a %b %d %H:%M:%S %Z %Y')
        head = '#ObservingScript -Name "pointing_{sname}.txt" -Author "obs_planner" -Date "{timestamp}"\n'.format(**locals())
        mctxt = head + otxt, stxt + ltxt + "\n"
        
        if(0):
            oF = open(outFile, "w")
            oF.write("# LMT OT script created by smaPointingSourceFinder.py\n")
            oF.write('ObsGoal DCS; Dcs -ObsGoal Pointing\n')
            oF.write(stxt)
            oF.write(ltxt)
            oF.close()
            
        return mctxt


     
