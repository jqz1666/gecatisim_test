# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license


import os
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
from gecatsim.pyfiles.CommonTools import my_path


_CUDA_LIB = None
_CUDA_LIB_PATH = None


def is_cuda_projector_available(cfg=None):
    if cfg is not None and not hasattr(cfg.phantom, "gpuVoxelData"):
        return False
    try:
        _load_cuda_lib(cfg)
        return True
    except Exception:
        return False


def C_Projector_Voxelized_CUDA(cfg, viewId, subViewId):
    lib = _load_cuda_lib(cfg)
    host = _get_or_create_host_cache(cfg)

    det = cfg.detNew
    src = cfg.srcNew
    n_pixels = int(det.totalNumCells)
    n_energy = int(cfg.spec.nEbin)
    n_modules = int(det.nMod)
    n_materials = int(cfg.phantom.numberOfMaterials)
    det_cells_per_mod = int(det.nCells)
    device_id = int(getattr(cfg.physics, "gpuDeviceId", 0))

    det_cell_coords = np.ascontiguousarray(det.cellCoords, dtype=np.float32)
    det_mod_coords = np.ascontiguousarray(det.modCoords, dtype=np.float32)
    det_uvecs = np.ascontiguousarray(det.uvecs, dtype=np.float32)
    det_vvecs = np.ascontiguousarray(det.vvecs, dtype=np.float32)
    det_start_indices = np.ascontiguousarray(det.startIndices, dtype=np.int32)
    mod_type_inds = np.ascontiguousarray(det.modTypes.reshape(-1), dtype=np.int32)
    source_points = np.ascontiguousarray(src.samples[: src.nSamples, :], dtype=np.float32)
    source_weights = np.ascontiguousarray(src.weights[0, : src.nSamples], dtype=np.float32)

    trans = np.zeros((n_pixels, n_energy), dtype=np.float32)
    batch_projector = getattr(lib, "voxelized_projector_cuda_batch", None)

    if batch_projector is not None:
        status = batch_projector(
            host["volume_data"],
            host["volume_offsets"],
            host["dims"],
            host["volume_offsets_xyz"],
            host["voxel_size"],
            host["mu"],
            host["xy_mask"],
            host["xy_mask_offsets"],
            source_points,
            source_weights,
            int(src.nSamples),
            det_cell_coords,
            mod_type_inds,
            det_mod_coords,
            det_uvecs,
            det_vvecs,
            det_start_indices,
            det_cells_per_mod,
            n_modules,
            n_materials,
            n_energy,
            n_pixels,
            int(host["volume_data"].size),
            trans,
            device_id,
        )
        if status != 0:
            raise RuntimeError(f"Native CUDA voxelized batch projector failed with status {status}.")
    else:
        matPVS = np.zeros((n_pixels, n_energy), dtype=np.float32)
        for srcId in range(src.nSamples):
            theSourcePoint = np.ascontiguousarray(src.samples[srcId, :], dtype=np.float32)
            pValueSpectrum = np.zeros((n_pixels, n_energy), dtype=np.float32)

            for matId in range(n_materials):
                matPVS[:] = 0
                MaterialIndex = matId + 1
                MaterialIndexInMemory = MaterialIndex

                status = lib.voxelized_projector_cuda(
                    host["volume_data"],
                    host["volume_offsets"],
                    host["dims"],
                    host["volume_offsets_xyz"],
                    host["voxel_size"],
                    host["mu"],
                    host["xy_mask"],
                    host["xy_mask_offsets"],
                    theSourcePoint,
                    det_cell_coords,
                    mod_type_inds,
                    det_mod_coords,
                    det_uvecs,
                    det_vvecs,
                    det_start_indices,
                    det_cells_per_mod,
                    n_modules,
                    n_materials,
                    n_energy,
                    n_pixels,
                    MaterialIndex,
                    MaterialIndexInMemory,
                    int(host["volume_data"].size),
                    matPVS,
                    device_id,
                )
                if status != 0:
                    raise RuntimeError(f"Native CUDA voxelized projector failed with status {status}.")

                pValueSpectrum += matPVS

            trans += source_weights[srcId] * np.exp(-pValueSpectrum)

    cfg.thisSubView *= trans

    if viewId == cfg.sim.stopViewId and subViewId == cfg.sim.subViewCount - 1:
        clear_cache = getattr(lib, "voxelized_projector_cuda_clear_cache", None)
        if clear_cache is not None:
            clear_cache()

    return cfg


def _load_cuda_lib(cfg=None):
    global _CUDA_LIB, _CUDA_LIB_PATH

    lib_path = _resolve_cuda_lib_path(cfg)
    if _CUDA_LIB is not None and _CUDA_LIB_PATH == lib_path:
        return _CUDA_LIB

    if os.name == "nt":
        my_path.add_dir_to_path(os.path.dirname(lib_path))

    lib = cdll.LoadLibrary(lib_path)
    lib.voxelized_projector_cuda.argtypes = [
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_int64, flags="C_CONTIGUOUS"),
        ndpointer(c_int, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_ubyte, flags="C_CONTIGUOUS"),
        ndpointer(c_int64, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_int, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_int, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int64,
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        c_int,
    ]
    lib.voxelized_projector_cuda.restype = c_int

    try:
        lib.voxelized_projector_cuda_batch.argtypes = [
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            ndpointer(c_int64, flags="C_CONTIGUOUS"),
            ndpointer(c_int, flags="C_CONTIGUOUS"),
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            ndpointer(c_ubyte, flags="C_CONTIGUOUS"),
            ndpointer(c_int64, flags="C_CONTIGUOUS"),
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            c_int,
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            ndpointer(c_int, flags="C_CONTIGUOUS"),
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            ndpointer(c_int, flags="C_CONTIGUOUS"),
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int64,
            ndpointer(c_float, flags="C_CONTIGUOUS"),
            c_int,
        ]
        lib.voxelized_projector_cuda_batch.restype = c_int
    except AttributeError:
        pass

    _CUDA_LIB = lib
    _CUDA_LIB_PATH = lib_path

    try:
        lib.voxelized_projector_cuda_clear_cache.argtypes = []
        lib.voxelized_projector_cuda_clear_cache.restype = None
    except AttributeError:
        pass

    return lib


def _resolve_cuda_lib_path(cfg=None):
    if cfg is not None and hasattr(cfg.physics, "cudaProjectorLib"):
        return cfg.physics.cudaProjectorLib

    lib_dir = my_path.paths["lib"]
    if os.name == "nt":
        name = "catsim_voxel_projector_cuda.dll"
    elif hasattr(os, "uname") and os.uname()[0] == "Darwin":
        name = "libcatsim_voxel_projector_cuda.dylib"
    else:
        name = "libcatsim_voxel_projector_cuda.so"
    return os.path.join(lib_dir, name)


def _get_or_create_host_cache(cfg):
    gpu_data = cfg.phantom.gpuVoxelData
    signature = (
        id(gpu_data),
        tuple(tuple(v["dims"].tolist()) for v in gpu_data["volumes"]),
        tuple(v["data"].shape[0] for v in gpu_data["volumes"]),
        tuple(gpu_data["mus"].shape),
    )

    cache = getattr(cfg.phantom, "_cudaProjectorHostCache", None)
    if cache is not None and cache.get("phantom_signature") == signature:
        return cache

    volumes = gpu_data["volumes"]
    volume_offsets = np.zeros(len(volumes) + 1, dtype=np.int64)
    for idx, volume in enumerate(volumes):
        volume_offsets[idx + 1] = volume_offsets[idx] + volume["data"].size

    volume_data = np.empty(volume_offsets[-1], dtype=np.float32)
    xy_mask_offsets = np.zeros(len(volumes) + 1, dtype=np.int64)
    for idx, volume in enumerate(volumes):
        dims = np.asarray(volume["dims"], dtype=np.int32)
        xy_mask_offsets[idx + 1] = xy_mask_offsets[idx] + int(dims[0] * dims[1] * 2)

    xy_mask = np.empty(xy_mask_offsets[-1], dtype=np.uint8)
    dims = np.empty((len(volumes), 3), dtype=np.int32)
    volume_offsets_xyz = np.empty((len(volumes), 3), dtype=np.float32)
    voxel_size = np.empty((len(volumes), 3), dtype=np.float32)

    for idx, volume in enumerate(volumes):
        start = volume_offsets[idx]
        stop = volume_offsets[idx + 1]
        volume_data[start:stop] = volume["data"]
        mask_start = xy_mask_offsets[idx]
        mask_stop = xy_mask_offsets[idx + 1]
        dims[idx, :] = volume["dims"]
        voxel_size[idx, :] = volume["voxelsize"]
        volume_offsets_xyz[idx, :] = volume["offsets"]
        nx = int(dims[idx, 0])
        ny = int(dims[idx, 1])
        base_mask = np.ascontiguousarray(volume["xyMask"].reshape(-1)[: nx * ny], dtype=np.uint8)
        xy_mask[mask_start:mask_start + nx * ny] = base_mask
        xy_mask[mask_start + nx * ny:mask_stop] = base_mask.reshape(ny, nx).T.reshape(-1)

    cache = {
        "phantom_signature": signature,
        "volume_data": np.ascontiguousarray(volume_data, dtype=np.float32),
        "volume_offsets": np.ascontiguousarray(volume_offsets, dtype=np.int64),
        "dims": np.ascontiguousarray(dims, dtype=np.int32),
        "volume_offsets_xyz": np.ascontiguousarray(volume_offsets_xyz, dtype=np.float32),
        "voxel_size": np.ascontiguousarray(voxel_size, dtype=np.float32),
        "mu": np.ascontiguousarray(gpu_data["mus"], dtype=np.float32),
        "xy_mask": np.ascontiguousarray(xy_mask, dtype=np.uint8),
        "xy_mask_offsets": np.ascontiguousarray(xy_mask_offsets, dtype=np.int64),
    }
    cfg.phantom._cudaProjectorHostCache = cache
    return cache
