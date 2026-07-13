# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license

import math
import numpy as np

try:
    from numba import cuda, float32, int32
except Exception:
    cuda = None
    float32 = None
    int32 = None


def is_gpu_projector_available(cfg=None):
    if cuda is None:
        return False
    try:
        if cfg is not None and not hasattr(cfg.phantom, "gpuVoxelData"):
            return False
        return cuda.is_available()
    except Exception:
        return False


def C_Projector_Voxelized_GPU(cfg, viewId, subViewId):
    if not is_gpu_projector_available(cfg):
        raise RuntimeError("Numba CUDA voxelized projector is not available.")

    cuda.select_device(getattr(cfg.physics, "gpuDeviceId", 0))

    cache = _get_or_create_cache(cfg)
    _refresh_geometry_cache(cfg, cache, viewId, subViewId)

    det = cfg.detNew
    n_pix = int(det.totalNumCells)
    n_ebin = int(cfg.spec.nEbin)
    trans = cuda.device_array((n_pix, n_ebin), dtype=np.float32)

    threads = 128
    blocks = (n_pix + threads - 1) // threads
    _voxelized_projector_kernel[blocks, threads](
        cache["volume_data"],
        cache["volume_offsets"],
        cache["dims"],
        cache["volume_origin"],
        cache["voxel_size"],
        cache["mu"],
        cache["src_samples"],
        cache["src_weights"],
        cache["det_cell_coords"],
        cache["det_mod_coords"],
        cache["det_uvecs"],
        cache["det_vvecs"],
        cache["det_start_indices"],
        cache["det_cells_per_mod"],
        np.int32(det.nMod),
        np.int32(cfg.phantom.numberOfMaterials),
        np.int32(n_ebin),
        trans,
    )

    cfg.thisSubView *= trans.copy_to_host()
    return cfg


def _get_or_create_cache(cfg):
    gpu_data = cfg.phantom.gpuVoxelData
    signature = (
        id(gpu_data),
        tuple(tuple(v["dims"].tolist()) for v in gpu_data["volumes"]),
        tuple(v["data"].shape[0] for v in gpu_data["volumes"]),
        tuple(gpu_data["mus"].shape),
    )

    cache = getattr(cfg.phantom, "_gpuProjectorCache", None)
    if cache is not None and cache.get("phantom_signature") == signature:
        return cache

    volumes = gpu_data["volumes"]
    volume_offsets = np.zeros(len(volumes) + 1, dtype=np.int32)
    for idx, volume in enumerate(volumes):
        volume_offsets[idx + 1] = volume_offsets[idx] + volume["data"].size

    volume_data = np.empty(volume_offsets[-1], dtype=np.float32)
    dims = np.empty((len(volumes), 3), dtype=np.int32)
    volume_origin = np.empty((len(volumes), 3), dtype=np.float32)
    voxel_size = np.empty((len(volumes), 3), dtype=np.float32)

    for idx, volume in enumerate(volumes):
        start = volume_offsets[idx]
        stop = volume_offsets[idx + 1]
        volume_data[start:stop] = volume["data"]
        dims[idx, :] = volume["dims"]
        voxel_size[idx, :] = volume["voxelsize"]
        volume_origin[idx, :] = volume["offsets"] - 0.5 * volume["dims"] * volume["voxelsize"]

    cache = {
        "phantom_signature": signature,
        "volume_data": cuda.to_device(volume_data),
        "volume_offsets": cuda.to_device(volume_offsets),
        "dims": cuda.to_device(dims),
        "volume_origin": cuda.to_device(volume_origin),
        "voxel_size": cuda.to_device(voxel_size),
        "mu": cuda.to_device(np.ascontiguousarray(gpu_data["mus"], dtype=np.float32)),
        "geometry_signature": None,
    }
    cfg.phantom._gpuProjectorCache = cache
    return cache


def _refresh_geometry_cache(cfg, cache, viewId, subViewId):
    det = cfg.detNew
    src = cfg.srcNew
    signature = (
        viewId,
        subViewId,
        id(det),
        id(src),
        np.asarray(det.modCoords).shape,
        np.asarray(src.samples).shape,
    )
    if cache.get("geometry_signature") == signature:
        return

    cache["src_samples"] = cuda.to_device(np.ascontiguousarray(src.samples, dtype=np.float32))
    cache["src_weights"] = cuda.to_device(np.ascontiguousarray(src.weights.reshape(-1), dtype=np.float32))
    cache["det_cell_coords"] = cuda.to_device(np.ascontiguousarray(det.cellCoords, dtype=np.float32))
    cache["det_mod_coords"] = cuda.to_device(np.ascontiguousarray(det.modCoords, dtype=np.float32))
    cache["det_uvecs"] = cuda.to_device(np.ascontiguousarray(det.uvecs, dtype=np.float32))
    cache["det_vvecs"] = cuda.to_device(np.ascontiguousarray(det.vvecs, dtype=np.float32))
    cache["det_start_indices"] = cuda.to_device(np.ascontiguousarray(det.startIndices, dtype=np.int32))
    cache["det_cells_per_mod"] = np.int32(det.nCells)
    cache["geometry_signature"] = signature


if cuda is not None:
    @cuda.jit
    def _voxelized_projector_kernel(
        volume_data,
        volume_offsets,
        dims,
        volume_origin,
        voxel_size,
        mu,
        src_samples,
        src_weights,
        det_cell_coords,
        det_mod_coords,
        det_uvecs,
        det_vvecs,
        det_start_indices,
        det_cells_per_mod,
        n_mod,
        n_materials,
        n_ebin,
        trans,
    ):
        pix = cuda.grid(1)
        if pix >= trans.shape[0]:
            return

        mod_id = int32(0)
        for m in range(n_mod):
            start = det_start_indices[m]
            if pix >= start and pix < start + det_cells_per_mod:
                mod_id = m
                break

        local_pix = pix - det_start_indices[mod_id]
        det_x = det_mod_coords[mod_id, 0] + det_cell_coords[local_pix, 0] * det_uvecs[mod_id, 0] + det_cell_coords[local_pix, 1] * det_vvecs[mod_id, 0]
        det_y = det_mod_coords[mod_id, 1] + det_cell_coords[local_pix, 0] * det_uvecs[mod_id, 1] + det_cell_coords[local_pix, 1] * det_vvecs[mod_id, 1]
        det_z = det_mod_coords[mod_id, 2] + det_cell_coords[local_pix, 0] * det_uvecs[mod_id, 2] + det_cell_coords[local_pix, 1] * det_vvecs[mod_id, 2]

        for e in range(n_ebin):
            weighted_trans = float32(0.0)
            for src_id in range(src_samples.shape[0]):
                src_x = src_samples[src_id, 0]
                src_y = src_samples[src_id, 1]
                src_z = src_samples[src_id, 2]
                ray_x = det_x - src_x
                ray_y = det_y - src_y
                ray_z = det_z - src_z
                ray_len = math.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
                p_value = float32(0.0)

                for mat_id in range(n_materials):
                    path = _trace_volume(
                        volume_data,
                        volume_offsets,
                        dims,
                        volume_origin,
                        voxel_size,
                        mat_id,
                        src_x,
                        src_y,
                        src_z,
                        ray_x,
                        ray_y,
                        ray_z,
                        ray_len,
                    )
                    p_value += path * mu[e, mat_id]

                weighted_trans += src_weights[src_id] * math.exp(-p_value)
            trans[pix, e] = weighted_trans


    @cuda.jit(device=True)
    def _trace_volume(
        volume_data,
        volume_offsets,
        dims,
        volume_origin,
        voxel_size,
        mat_id,
        src_x,
        src_y,
        src_z,
        ray_x,
        ray_y,
        ray_z,
        ray_len,
    ):
        nx = dims[mat_id, 0]
        ny = dims[mat_id, 1]
        nz = dims[mat_id, 2]
        ox = volume_origin[mat_id, 0]
        oy = volume_origin[mat_id, 1]
        oz = volume_origin[mat_id, 2]
        vx = voxel_size[mat_id, 0]
        vy = voxel_size[mat_id, 1]
        vz = voxel_size[mat_id, 2]

        t0 = float32(0.0)
        t1 = float32(1.0)
        t0, t1 = _clip_axis(src_x, ray_x, ox, ox + nx * vx, t0, t1)
        if t0 >= t1:
            return float32(0.0)
        t0, t1 = _clip_axis(src_y, ray_y, oy, oy + ny * vy, t0, t1)
        if t0 >= t1:
            return float32(0.0)
        t0, t1 = _clip_axis(src_z, ray_z, oz, oz + nz * vz, t0, t1)
        if t0 >= t1:
            return float32(0.0)

        t = t0
        x = src_x + t * ray_x
        y = src_y + t * ray_y
        z = src_z + t * ray_z
        ix = _clamp_index(int32(math.floor((x - ox) / vx)), nx)
        iy = _clamp_index(int32(math.floor((y - oy) / vy)), ny)
        iz = _clamp_index(int32(math.floor((z - oz) / vz)), nz)

        step_x = int32(0)
        step_y = int32(0)
        step_z = int32(0)
        next_tx = float32(2.0)
        next_ty = float32(2.0)
        next_tz = float32(2.0)
        delta_tx = float32(2.0)
        delta_ty = float32(2.0)
        delta_tz = float32(2.0)

        if ray_x > 0:
            step_x = 1
            next_tx = ((ox + (ix + 1) * vx) - src_x) / ray_x
            delta_tx = vx / ray_x
        elif ray_x < 0:
            step_x = -1
            next_tx = ((ox + ix * vx) - src_x) / ray_x
            delta_tx = -vx / ray_x

        if ray_y > 0:
            step_y = 1
            next_ty = ((oy + (iy + 1) * vy) - src_y) / ray_y
            delta_ty = vy / ray_y
        elif ray_y < 0:
            step_y = -1
            next_ty = ((oy + iy * vy) - src_y) / ray_y
            delta_ty = -vy / ray_y

        if ray_z > 0:
            step_z = 1
            next_tz = ((oz + (iz + 1) * vz) - src_z) / ray_z
            delta_tz = vz / ray_z
        elif ray_z < 0:
            step_z = -1
            next_tz = ((oz + iz * vz) - src_z) / ray_z
            delta_tz = -vz / ray_z

        path = float32(0.0)
        base = volume_offsets[mat_id]
        while t < t1 and ix >= 0 and ix < nx and iy >= 0 and iy < ny and iz >= 0 and iz < nz:
            t_next = t1
            axis = int32(2)
            if next_tx < t_next:
                t_next = next_tx
                axis = 0
            if next_ty < t_next:
                t_next = next_ty
                axis = 1
            if next_tz < t_next:
                t_next = next_tz
                axis = 2

            if t_next > t:
                idx = base + (iy * nx + ix) * nz + iz
                path += volume_data[idx] * (t_next - t) * ray_len

            t = t_next
            if axis == 0:
                ix += step_x
                next_tx += delta_tx
            elif axis == 1:
                iy += step_y
                next_ty += delta_ty
            else:
                iz += step_z
                next_tz += delta_tz

        return path


    @cuda.jit(device=True)
    def _clip_axis(src, ray, lower, upper, t0, t1):
        if ray == 0:
            if src < lower or src > upper:
                return float32(1.0), float32(0.0)
            return t0, t1

        inv = float32(1.0) / ray
        ta = (lower - src) * inv
        tb = (upper - src) * inv
        if ta > tb:
            tmp = ta
            ta = tb
            tb = tmp
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        return t0, t1


    @cuda.jit(device=True)
    def _clamp_index(index, size):
        if index < 0:
            return int32(0)
        if index >= size:
            return size - 1
        return index
