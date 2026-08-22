# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license


import os
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
from gecatsim.pyfiles.CommonTools import my_path


_CUDA_LIB = None
_CUDA_LIB_PATH = None


def is_cuda_projector_available(cfg=None):
    try:
        _load_cuda_lib(cfg)
        if cfg is not None:
            host = _get_cpu_phantom_data(cfg)
            if host["volume_data_type"] not in (0, 1):
                return False
        return True
    except Exception:
        return False


def C_Projector_Voxelized_CUDA(cfg, viewId, subViewId):
    lib = _load_cuda_lib(cfg)
    host = _get_cpu_phantom_data(cfg)

    if host["volume_data_type"] not in (0, 1):
        raise RuntimeError("Native CUDA voxelized projector received an unsupported CPU phantom data type.")

    det = cfg.detNew
    src = cfg.srcNew
    n_pixels = int(det.totalNumCells)
    n_energy = int(cfg.spec.nEbin)
    n_sources = int(src.nSamples)
    n_modules = int(det.nMod)
    n_materials = int(cfg.phantom.numberOfMaterials)
    det_cells_per_mod = int(det.nCells)
    device_id = int(getattr(cfg.physics, "gpuDeviceId", 0))

    src_samples = np.ascontiguousarray(src.samples, dtype=np.float32)
    src_weights = np.ascontiguousarray(src.weights.reshape(-1), dtype=np.float32)
    det_cell_coords = np.ascontiguousarray(det.cellCoords, dtype=np.float32)
    det_mod_coords = np.ascontiguousarray(det.modCoords, dtype=np.float32)
    det_uvecs = np.ascontiguousarray(det.uvecs, dtype=np.float32)
    det_vvecs = np.ascontiguousarray(det.vvecs, dtype=np.float32)
    det_start_indices = np.ascontiguousarray(det.startIndices, dtype=np.int32)
    trans = np.empty((n_pixels, n_energy), dtype=np.float32)

    status = lib.voxelized_projector_cuda(
        host["volume_pointers"],
        host["dims"],
        host["volume_offsets_xyz"],
        host["voxel_size"],
        host["mu"],
        host["xy_mask_pointers"],
        host["phantom_signature"],
        host["volume_data_type"],
        src_samples,
        src_weights,
        det_cell_coords,
        np.ascontiguousarray(det.modTypes.reshape(-1), dtype=np.int32),
        det_mod_coords,
        det_uvecs,
        det_vvecs,
        det_start_indices,
        det_cells_per_mod,
        n_modules,
        n_materials,
        n_energy,
        n_pixels,
        n_sources,
        trans,
        device_id,
    )
    if status != 0:
        clear_cache = getattr(lib, "voxelized_projector_cuda_clear_cache", None)
        if clear_cache is not None:
            clear_cache()
        raise RuntimeError(f"Native CUDA voxelized projector failed with status {status}.")

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
        POINTER(c_void_p),
        ndpointer(c_int, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        POINTER(c_float),
        POINTER(POINTER(c_ubyte)),
        c_ulonglong,
        c_int,
        ndpointer(c_float, flags="C_CONTIGUOUS"),
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
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        c_int,
    ]
    lib.voxelized_projector_cuda.restype = c_int

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


def _get_cpu_phantom_data(cfg):
    getter = cfg.clib.get_phantom_info_vox_for_cuda
    getter.argtypes = [
        c_int,
        POINTER(c_void_p),
        ndpointer(c_int, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        ndpointer(c_float, flags="C_CONTIGUOUS"),
        POINTER(POINTER(c_ubyte)),
        POINTER(POINTER(c_float)),
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_int),
    ]
    getter.restype = c_int

    n_materials = int(cfg.phantom.numberOfMaterials)
    dims = np.empty((n_materials, 3), dtype=np.int32)
    volume_offsets_xyz = np.empty((n_materials, 3), dtype=np.float32)
    voxel_size = np.empty((n_materials, 3), dtype=np.float32)
    volume_pointers = (c_void_p * n_materials)()
    xy_mask_pointers = (POINTER(c_ubyte) * n_materials)()
    mu_pointer = POINTER(c_float)()
    material_count = c_int()
    energy_count = c_int()
    volume_data_type = c_int()
    pointer_signature_values = []

    for material_index in range(n_materials):
        volume_pointer = c_void_p()
        xy_mask_pointer = POINTER(c_ubyte)()
        this_mu_pointer = POINTER(c_float)()
        status = getter(
            material_index + 1,
            byref(volume_pointer),
            dims[material_index],
            volume_offsets_xyz[material_index],
            voxel_size[material_index],
            byref(xy_mask_pointer),
            byref(this_mu_pointer),
            byref(material_count),
            byref(energy_count),
            byref(volume_data_type),
        )
        if status != 0:
            raise RuntimeError(f"Unable to access CPU voxelized phantom data for material {material_index + 1} (status {status}).")
        if material_count.value != n_materials:
            raise RuntimeError("CPU voxelized phantom material count does not match cfg.phantom.numberOfMaterials.")
        if not volume_pointer.value or not bool(xy_mask_pointer) or not bool(this_mu_pointer):
            raise RuntimeError("CPU voxelized phantom returned a null data pointer.")

        volume_pointers[material_index] = volume_pointer.value
        xy_mask_pointers[material_index] = xy_mask_pointer
        if material_index == 0:
            mu_pointer = this_mu_pointer
        elif addressof(this_mu_pointer.contents) != addressof(mu_pointer.contents):
            raise RuntimeError("CPU voxelized phantom returned inconsistent material attenuation data.")
        pointer_signature_values.extend((
            volume_pointer.value,
            cast(xy_mask_pointer, c_void_p).value,
            int(dims[material_index, 0]),
            int(dims[material_index, 1]),
            int(dims[material_index, 2]),
        ))

    pointer_signature_values.extend((
        cast(mu_pointer, c_void_p).value,
        n_materials,
        energy_count.value,
        volume_data_type.value,
    ))

    return {
        "volume_pointers": volume_pointers,
        "dims": np.ascontiguousarray(dims, dtype=np.int32),
        "volume_offsets_xyz": np.ascontiguousarray(volume_offsets_xyz, dtype=np.float32),
        "voxel_size": np.ascontiguousarray(voxel_size, dtype=np.float32),
        "mu": mu_pointer,
        "xy_mask_pointers": xy_mask_pointers,
        "volume_data_type": volume_data_type.value,
        "phantom_signature": _pointer_signature(pointer_signature_values),
    }


def _pointer_signature(values):
    signature = 1469598103934665603
    for value in values:
        signature ^= int(value)
        signature = (signature * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return signature
