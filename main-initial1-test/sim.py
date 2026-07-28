# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license

###------------ import XCIST-CatSim
import gecatsim as xc
from gecatsim.reconstruction.pyfiles import recon

##--------- Initialize
ct = xc.CatSim("./cfg/Phantom",
               "./cfg/Protocol",
               "./cfg/Scanner",
               "./cfg/Physics",
               "./cfg/Recon",

        )  # initialization

ct.resultsName = "test_xcist_executable"

##--------- Run simulation
ct.run_all()  # run the scans defined by protocol.scanTypes


if ct.physics.monochromatic>0:
        ct.recon.mu = xc.GetMu('water', ct.physics.monochromatic)[0]/10

cfg = ct.get_current_cfg();
cfg.do_Recon = 1
cfg.waitForKeypress = 0
recon.recon(cfg)
