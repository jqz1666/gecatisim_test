# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license

import time

import gecatsim as xc
from gecatsim.pyfiles.OneScan import one_scan


CFG_FILES = (
    "../examples/cfg/Phantom_Sample",
    "../examples/cfg/Scanner_Sample_generic",
    "../examples/cfg/Protocol_Sample_axial",
)


def make_cfg():
    ct = xc.CatSim(*CFG_FILES)
    cfg = ct.get_current_cfg()
    cfg.physics.enableQuantumNoise = 0
    cfg.physics.enableElectronicNoise = 0
    cfg.protocol.startViewId = 0
    cfg.protocol.stopViewId = 9
    return cfg


def run_once(backend):
    run_cfg = make_cfg()
    run_cfg.physics.projectorBackend = backend
    run_cfg.sim.thisScanType = [0, 0, 1, 0]
    start = time.perf_counter()
    one_scan(run_cfg)
    return time.perf_counter() - start


def main():
    cpu_time = run_once("cpu")
    print(f"CPU projector: {cpu_time:.3f} s")

    cuda_time = run_once("cuda")
    print(f"Native DD3-compatible CUDA interface: {cuda_time:.3f} s")

    numba_time = run_once("numba")
    print(f"Experimental Numba CUDA projector: {numba_time:.3f} s")


if __name__ == "__main__":
    main()
