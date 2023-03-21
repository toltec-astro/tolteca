#!/bin/bash

brcdir=$HOME/toltec_astro/run/tolteca/beammap
prcdir=$HOME/toltec_astro/run/tolteca/pointing
srcdir=$HOME/toltec_astro/run/tolteca/science

if [[ ! $1 ]]; then
    echo enter obsnum
else
    obsnum=$1

    aptdir=${HOME}/toltec_astro/run/apt

    ln -sf ${aptdir}/apt_${obsnum}_cleaned.ecsv ${brcdir}/apt.ecsv
    ln -sf ${aptdir}/apt_${obsnum}_cleaned.ecsv ${prcdir}/apt.ecsv
    ln -sf ${aptdir}/apt_${obsnum}_cleaned.ecsv ${srcdir}/apt.ecsv
fi
