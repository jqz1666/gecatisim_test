// Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

static const int CUDA_PROJECTOR_SUCCESS = 0;
static const int CUDA_PROJECTOR_BAD_ARGUMENT = -1;
static const int CUDA_PROJECTOR_DETECTOR_ERROR = -2;
static const int CUDA_PROJECTOR_MEMORY_ERROR = -3;
static const int CUDA_PROJECTOR_RUNTIME_ERROR = -4;

#define MIN_CUDA(a, b) (((a) < (b)) ? (a) : (b))

static int convert_modular_detector_compatible(
    float **xds,
    float **yds,
    float **zds,
    int *nrdetcols,
    int *nrdetrows,
    int nModulesIn,
    const int *modTypeInds,
    const float *Up,
    const float *Right,
    const float *Center,
    const float *coords,
    int pixPerModule)
{
    const float tiny = 1e-8f;
    const float tol = 1e-6f;
    int thisNumRows = 0;

    *nrdetcols = 0;
    *nrdetrows = 0;

    for (int modIndex = 0; modIndex < nModulesIn; ++modIndex)
    {
        const float *up = Up + 3 * modIndex;
        const float *right = Right + 3 * modIndex;

        float inPlane = up[0] * up[0] + up[1] * up[1];
        float inZ = up[2] * up[2];
        if ((inPlane > inZ) || (inZ != 0.0f && inPlane / inZ > tiny))
        {
            return CUDA_PROJECTOR_DETECTOR_ERROR;
        }

        inPlane = right[0] * right[0] + right[1] * right[1];
        inZ = right[2] * right[2];
        if ((inZ > inPlane) || (inPlane != 0.0f && inZ / inPlane > tiny))
        {
            return CUDA_PROJECTOR_DETECTOR_ERROR;
        }
    }

    for (int colIndex = 0; colIndex < pixPerModule; ++colIndex)
    {
        if (colIndex == 0 || fabsf(coords[2 * colIndex] - coords[0]) < tol)
        {
            thisNumRows++;
        }
        else
        {
            break;
        }
    }

    if (thisNumRows <= 0 || pixPerModule % thisNumRows != 0)
    {
        return CUDA_PROJECTOR_DETECTOR_ERROR;
    }
    *nrdetrows = thisNumRows;

    for (int colIndex = 0; colIndex < pixPerModule / thisNumRows; ++colIndex)
    {
        for (int rowIndex = 0; rowIndex < thisNumRows; ++rowIndex)
        {
            int idx = rowIndex + colIndex * thisNumRows;
            if (fabsf(coords[2 * idx + 1] - coords[2 * rowIndex + 1]) > tol ||
                fabsf(coords[2 * idx] - coords[2 * colIndex * thisNumRows]) > tol)
            {
                return CUDA_PROJECTOR_DETECTOR_ERROR;
            }
        }
    }

    for (int modIndex = 0; modIndex < nModulesIn; ++modIndex)
    {
        (void)modTypeInds[modIndex];
        *nrdetcols += pixPerModule / (*nrdetrows);
    }

    *xds = new float[*nrdetcols];
    *yds = new float[*nrdetcols];
    *zds = new float[*nrdetrows];
    if (!*xds || !*yds || !*zds)
    {
        return CUDA_PROJECTOR_MEMORY_ERROR;
    }

    *nrdetcols = 0;
    for (int modIndex = 0; modIndex < nModulesIn; ++modIndex)
    {
        const float *up = Up + 3 * modIndex;
        const float *right = Right + 3 * modIndex;
        const float *center = Center + 3 * modIndex;
        int thisNumCols = pixPerModule / (*nrdetrows);

        if (modIndex == 0)
        {
            for (int rowIndex = 0; rowIndex < *nrdetrows; ++rowIndex)
            {
                (*zds)[rowIndex] = center[2] + coords[1 + rowIndex * 2] * up[2];
            }
        }
        else
        {
            for (int rowIndex = 0; rowIndex < *nrdetrows; ++rowIndex)
            {
                float z = center[2] + coords[1 + rowIndex * 2] * up[2];
                if (fabsf((*zds)[rowIndex] - z) > tol)
                {
                    return CUDA_PROJECTOR_DETECTOR_ERROR;
                }
            }
        }

        for (int colIndex = 0; colIndex < thisNumCols; ++colIndex)
        {
            int idx = colIndex * 2 * (*nrdetrows);
            (*xds)[colIndex + *nrdetcols] = center[0] + coords[idx] * right[0];
            (*yds)[colIndex + *nrdetcols] = center[1] + coords[idx] * right[1];
        }
        *nrdetcols += thisNumCols;
    }

    return CUDA_PROJECTOR_SUCCESS;
}

static void dd3_boundaries(int nrBoundaries, const float *centers, float *boundaries)
{
    if (nrBoundaries >= 3)
    {
        boundaries[0] = 1.5f * centers[0] - 0.5f * centers[1];
        for (int i = 1; i <= nrBoundaries - 2; ++i)
        {
            boundaries[i] = 0.5f * centers[i - 1] + 0.5f * centers[i];
        }
        boundaries[nrBoundaries - 1] = 1.5f * centers[nrBoundaries - 2] - 0.5f * centers[nrBoundaries - 3];
    }
    else
    {
        boundaries[0] = centers[0] - 0.5f;
        boundaries[1] = centers[0] + 0.5f;
    }
}

__global__ void prepare_dd3_geometry_kernel(
    const float *src_samples,
    const float *volume_offsets_xyz,
    const float *xdi,
    const float *ydi,
    int nrdetcols,
    int n_materials,
    int n_sources,
    float *detX,
    float *scales,
    float *x0_local,
    float *y0_local,
    float *z0_local,
    int *vertical_flags)
{
    int task = blockIdx.x * blockDim.x + threadIdx.x;
    int task_count = n_sources * n_materials;
    if (task >= task_count)
    {
        return;
    }

    int mat_id = task % n_materials;
    int src_id = task / n_materials;
    float xoff = volume_offsets_xyz[mat_id * 3 + 0];
    float yoff = volume_offsets_xyz[mat_id * 3 + 1];
    float zoff = volume_offsets_xyz[mat_id * 3 + 2];
    float x0 = src_samples[src_id * 3 + 0] - xoff;
    float y0 = src_samples[src_id * 3 + 1] - yoff;
    float z0 = src_samples[src_id * 3 + 2] - zoff;
    int vertical = fabsf(y0) >= fabsf(x0);

    float *task_detX = detX + (size_t)task * (size_t)(nrdetcols + 3);
    float *task_scales = scales + (size_t)task * (size_t)(nrdetcols + 2);
    task_detX[0] = 1.0e12f;

    if (vertical)
    {
        for (int i = 0; i < nrdetcols + 1; ++i)
        {
            float xr = xdi[i] - xoff;
            float yr = ydi[i] - yoff;
            task_detX[i + 1] = (x0 * yr - xr * y0) / (yr - y0);
            task_scales[i + 1] = y0 / (y0 - yr);
        }
    }
    else
    {
        for (int i = 0; i < nrdetcols + 1; ++i)
        {
            float xr = xdi[i] - xoff;
            float yr = ydi[i] - yoff;
            task_detX[i + 1] = -(y0 * xr - yr * x0) / (xr - x0);
            task_scales[i + 1] = x0 / (x0 - xr);
        }
        float old_x0 = x0;
        x0 = -y0;
        y0 = -old_x0;
    }

    task_detX[nrdetcols + 2] = 1.0e12f;
    task_scales[0] = task_scales[1];
    for (int i = 1; i <= nrdetcols; ++i)
    {
        task_scales[i] = 0.5f * (task_scales[i] + task_scales[i + 1]);
    }

    x0_local[task] = x0;
    y0_local[task] = y0;
    z0_local[task] = z0;
    vertical_flags[task] = vertical;
}

__device__ void dd3_project_row_device(
    float imgX,
    float imgXstep,
    int nrcols,
    float imgZstart,
    float imgZstep,
    int nrplanes,
    const float *pImg,
    int pImgBase,
    int trans_rowstep,
    const float *detX,
    int increment,
    const float *detZabs,
    const float *scales,
    float src_z_abs,
    float z0,
    float *view,
    int nrdetrows,
    int nrdetcols,
    const unsigned char *xy_mask_row)
{
    int colnr = 0;
    int planenr = 0;
    int jumpcount = 0;
    float previousX = imgX;
    float imgXStart = imgX;
    float imgZ_end = imgZstart + nrplanes * imgZstep;
    float inv_imgZstep = 1.0f / imgZstep;
    float inv_imgXstep = 1.0f / imgXstep;
    float imgZ = imgZstart + imgZstep;
    imgX += imgXstep;

    int detX_ind = 0;
    int detZ_ind = 0;
    float detX_0 = detX[0];
    float detX_end = detX[nrdetcols * increment];
    int start_imgX_ind = 0;
    int end_imgX_ind = nrcols;

    if (detX_end < (imgXStart + (nrcols - 1) * imgXstep))
    {
        end_imgX_ind = (int)((detX_end - imgXStart) * inv_imgXstep) + 2;
        end_imgX_ind = MIN_CUDA(end_imgX_ind, nrcols);
    }

    int pImgOffset = pImgBase;
    int view_col = 0;
    if (detX_0 > previousX)
    {
        start_imgX_ind = (int)((detX_0 - imgXStart) * inv_imgXstep - 1.0f);
        if (start_imgX_ind > 0)
        {
            pImgOffset += start_imgX_ind * nrplanes * trans_rowstep;
            colnr = start_imgX_ind;
            previousX = imgXStart + imgXstep * start_imgX_ind;
            imgX = previousX + imgXstep;
        }
    }
    else
    {
        while (detX[detX_ind] <= previousX)
        {
            detX_ind += increment;
            jumpcount += increment;
        }
        view_col += jumpcount;
    }

    while (colnr < end_imgX_ind)
    {
        float scale = scales[detX_ind];
        int view_col_copy = view_col;
        int mask_ind = colnr;
        float dx;

        if (imgX <= detX[detX_ind])
        {
            dx = imgX - previousX;
            previousX = imgX;
            imgX += imgXstep;
            colnr++;
            jumpcount = nrplanes * trans_rowstep;
        }
        else
        {
            dx = detX[detX_ind] - previousX;
            previousX = detX[detX_ind];
            detX_ind += increment;
            view_col += increment;
            jumpcount = 0;
        }

        if (xy_mask_row[mask_ind])
        {
            planenr = 0;
            float previousZ = imgZstart;
            imgZ = imgZstart + imgZstep;
            detZ_ind = 0;
            float nextdetZ = z0 + scale * (detZabs[0] - src_z_abs);
            float enddetZ = z0 + scale * (detZabs[nrdetrows] - src_z_abs);
            int start_imgZ_ind = 0;
            int end_imgZ_ind = nrplanes;
            int pImgCopyOffset = pImgOffset;

            if (enddetZ < imgZ_end)
            {
                end_imgZ_ind = (int)((enddetZ - imgZstart) * inv_imgZstep) + 2;
                end_imgZ_ind = MIN_CUDA(end_imgZ_ind, nrplanes);
            }

            if (nextdetZ > previousZ)
            {
                start_imgZ_ind = (int)((nextdetZ - imgZstart) * inv_imgZstep) - 1;
                if (start_imgZ_ind > 0)
                {
                    pImgCopyOffset += start_imgZ_ind;
                    planenr = start_imgZ_ind;
                    previousZ = imgZstart + imgZstep * start_imgZ_ind;
                    imgZ = previousZ + imgZstep;
                }
            }
            else
            {
                while (nextdetZ <= previousZ)
                {
                    detZ_ind++;
                    nextdetZ = z0 + scale * (detZabs[detZ_ind] - src_z_abs);
                }
            }

            while (planenr < end_imgZ_ind)
            {
                int view_idx = view_col_copy * (nrdetrows + 2) + detZ_ind;
                if (imgZ <= nextdetZ)
                {
                    atomicAdd(view + view_idx, dx * (imgZ - previousZ) * pImg[pImgCopyOffset]);
                    pImgCopyOffset++;
                    planenr++;
                    previousZ = imgZ;
                    imgZ += imgZstep;
                }
                else
                {
                    atomicAdd(view + view_idx, dx * (nextdetZ - previousZ) * pImg[pImgCopyOffset]);
                    detZ_ind++;
                    previousZ = nextdetZ;
                    nextdetZ = z0 + scale * (detZabs[detZ_ind] - src_z_abs);
                }
            }
        }

        pImgOffset += jumpcount;
    }
}

__global__ void dd3_row_kernel(
    const float *volume_data,
    const int *volume_offsets,
    const int *dims,
    const float *volume_offsets_xyz,
    const float *voxel_size,
    const unsigned char *xy_mask,
    const int *xy_mask_offsets,
    const float *src_samples,
    const float *detZabs,
    const float *detX,
    const float *scales,
    const float *x0_local,
    const float *y0_local,
    const float *z0_local,
    const int *vertical_flags,
    int nrdetcols,
    int nrdetrows,
    int n_materials,
    int n_sources,
    int max_rows,
    float *views)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int task_count = n_sources * n_materials;
    int total = task_count * max_rows;
    if (idx >= total)
    {
        return;
    }

    int task = idx / max_rows;
    int rownr = idx - task * max_rows;
    int mat_id = task % n_materials;
    int src_id = task / n_materials;
    int nx = dims[mat_id * 3 + 0];
    int ny = dims[mat_id * 3 + 1];
    int nz = dims[mat_id * 3 + 2];
    float vox_xy = voxel_size[mat_id * 3 + 0];
    float vox_z = voxel_size[mat_id * 3 + 2];
    int vertical = vertical_flags[task];
    int nrcols = vertical ? nx : ny;
    int nrrows = vertical ? ny : nx;
    if (rownr >= nrrows)
    {
        return;
    }

    float x0 = x0_local[task];
    float y0 = y0_local[task];
    float z0 = z0_local[task];
    float mag_fac = y0 / (y0 - ((nrrows - 1) * 0.5f - rownr) * vox_xy);
    float imgXstep = mag_fac * vox_xy;
    float imgX = -x0 / (y0 - ((nrrows - 1) * 0.5f - rownr) * vox_xy) *
                     ((nrrows - 1) * 0.5f - rownr) * vox_xy -
                 (nrcols * 0.5f * vox_xy) * mag_fac;
    float imgZstep = mag_fac * vox_z;
    float imgZ = z0 - (nz * 0.5f * vox_z + z0) * mag_fac;

    int pImgBase;
    int trans_rowstep;
    const unsigned char *mask_row;
    if (vertical)
    {
        pImgBase = volume_offsets[mat_id] + rownr * nx * nz;
        trans_rowstep = 1;
        mask_row = xy_mask + xy_mask_offsets[mat_id] + rownr * nx;
    }
    else
    {
        pImgBase = volume_offsets[mat_id] + rownr * nz;
        trans_rowstep = nx;
        mask_row = xy_mask + xy_mask_offsets[mat_id] + ny * nx + rownr * ny;
    }

    const float *task_detX = detX + (size_t)task * (size_t)(nrdetcols + 3);
    const float *task_scales = scales + (size_t)task * (size_t)(nrdetcols + 2);
    float *task_view = views + (size_t)task * (size_t)(nrdetcols + 2) * (size_t)(nrdetrows + 2);
    int increment;
    const float *detXcopy;
    const float *scalesCopy;
    int view_start_col;
    if (task_detX[2] > task_detX[1])
    {
        increment = 1;
        detXcopy = task_detX + 1;
        scalesCopy = task_scales;
        view_start_col = 0;
    }
    else
    {
        increment = -1;
        detXcopy = task_detX + nrdetcols + 1;
        scalesCopy = task_scales + nrdetcols + 1;
        view_start_col = nrdetcols + 1;
    }

    dd3_project_row_device(
        imgX,
        imgXstep,
        nrcols,
        imgZ,
        imgZstep,
        nz,
        volume_data,
        pImgBase,
        trans_rowstep,
        detXcopy,
        increment,
        detZabs,
        scalesCopy,
        src_samples[src_id * 3 + 2],
        z0,
        task_view + view_start_col * (nrdetrows + 2),
        nrdetrows,
        nrdetcols,
        mask_row);
}

__global__ void dd3_scale_kernel(
    const float *detZabs,
    const float *src_samples,
    const float *voxel_size,
    const float *detX,
    const float *scales,
    const float *x0_local,
    const float *y0_local,
    const float *views,
    int nrdetcols,
    int nrdetrows,
    int n_materials,
    int n_pixels,
    int n_sources,
    float *paths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int task_count = n_sources * n_materials;
    int total = task_count * n_pixels;
    if (idx >= total)
    {
        return;
    }

    int pixel = idx % n_pixels;
    int task = idx / n_pixels;
    int mat_id = task % n_materials;
    int src_id = task / n_materials;
    int det_col = pixel / nrdetrows;
    int det_row = pixel - det_col * nrdetrows;
    const float *task_detX = detX + (size_t)task * (size_t)(nrdetcols + 3);
    const float *task_scales = scales + (size_t)task * (size_t)(nrdetcols + 2);
    const float *task_view = views + (size_t)task * (size_t)(nrdetcols + 2) * (size_t)(nrdetrows + 2);
    int view_idx = (det_col + 1) * (nrdetrows + 2) + det_row + 1;
    float view_value = task_view[view_idx];
    if (view_value == 0.0f)
    {
        paths[idx] = 0.0f;
        return;
    }

    float x0 = x0_local[task];
    float y0 = y0_local[task];
    float deltaX = 0.5f * (task_detX[det_col + 1] + task_detX[det_col + 2]) - x0;
    float detXstep = fabsf(task_detX[det_col + 2] - task_detX[det_col + 1]);
    float scale = task_scales[det_col + 1];
    float src_z = src_samples[src_id * 3 + 2];
    float detZ0 = detZabs[det_row] - src_z;
    float detZ1 = detZabs[det_row + 1] - src_z;
    float deltaZ = 0.5f * (detZ0 + detZ1) * scale;
    float detZstep = fabsf(detZ1 - detZ0) * scale;
    float invCos = sqrtf(y0 * y0 + deltaX * deltaX + deltaZ * deltaZ) / fabsf(y0) *
                   voxel_size[mat_id * 3 + 0];
    paths[idx] = invCos / (detXstep * detZstep) * view_value;
}

__global__ void combine_trans_kernel(
    const float *paths,
    const float *mu,
    const float *src_weights,
    int n_materials,
    int n_energy,
    int n_pixels,
    int n_sources,
    float *trans_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_pixels * n_energy;
    if (idx >= total)
    {
        return;
    }

    int energy = idx % n_energy;
    int pixel = idx / n_energy;
    float trans = 0.0f;

    for (int src_id = 0; src_id < n_sources; ++src_id)
    {
        float p_value = 0.0f;
        for (int mat_id = 0; mat_id < n_materials; ++mat_id)
        {
            int path_idx = (src_id * n_materials + mat_id) * n_pixels + pixel;
            p_value += paths[path_idx] * mu[energy * n_materials + mat_id];
        }
        trans += src_weights[src_id] * expf(-p_value);
    }

    trans_out[idx] = trans;
}

static void cuda_cleanup(void **ptrs, int count)
{
    for (int i = 0; i < count; ++i)
    {
        cudaFree(ptrs[i]);
    }
}

extern "C"
{
    DLLEXPORT
    int voxelized_projector_cuda(
        const float *volume_data,
        const int *volume_offsets,
        const int *dims,
        const float *volume_offsets_xyz,
        const float *voxel_size,
        const float *mu,
        const unsigned char *xy_mask,
        const int *xy_mask_offsets,
        const float *src_samples,
        const float *src_weights,
        const float *det_cell_coords,
        const int *det_mod_types,
        const float *det_mod_coords,
        const float *det_uvecs,
        const float *det_vvecs,
        const int *det_start_indices,
        int det_cells_per_mod,
        int n_modules,
        int n_materials,
        int n_energy,
        int n_pixels,
        int n_sources,
        int volume_data_count,
        float *trans_out,
        int device_id)
    {
        (void)det_start_indices;

        if (!volume_data || !volume_offsets || !dims || !volume_offsets_xyz || !voxel_size || !mu ||
            !xy_mask || !xy_mask_offsets || !src_samples || !src_weights || !det_cell_coords ||
            !det_mod_types || !det_mod_coords || !det_uvecs || !det_vvecs || !trans_out)
        {
            return CUDA_PROJECTOR_BAD_ARGUMENT;
        }
        if (det_cells_per_mod <= 0 || n_modules <= 0 || n_materials <= 0 || n_energy <= 0 ||
            n_pixels <= 0 || n_sources <= 0 || volume_data_count <= 0)
        {
            return CUDA_PROJECTOR_BAD_ARGUMENT;
        }

        float *xds = 0;
        float *yds = 0;
        float *zds = 0;
        int nrdetcols = 0;
        int nrdetrows = 0;
        int status = convert_modular_detector_compatible(
            &xds, &yds, &zds, &nrdetcols, &nrdetrows, n_modules, det_mod_types,
            det_vvecs, det_uvecs, det_mod_coords, det_cell_coords, det_cells_per_mod);
        if (status != CUDA_PROJECTOR_SUCCESS)
        {
            delete[] xds;
            delete[] yds;
            delete[] zds;
            return status;
        }
        if (nrdetcols * nrdetrows != n_pixels)
        {
            delete[] xds;
            delete[] yds;
            delete[] zds;
            return CUDA_PROJECTOR_DETECTOR_ERROR;
        }

        float *xdi = new float[nrdetcols + 1];
        float *ydi = new float[nrdetcols + 1];
        float *detZ = new float[nrdetrows + 1];
        if (!xdi || !ydi || !detZ)
        {
            delete[] xds;
            delete[] yds;
            delete[] zds;
            delete[] xdi;
            delete[] ydi;
            delete[] detZ;
            return CUDA_PROJECTOR_MEMORY_ERROR;
        }
        dd3_boundaries(nrdetcols + 1, xds, xdi);
        dd3_boundaries(nrdetcols + 1, yds, ydi);
        dd3_boundaries(nrdetrows + 1, zds, detZ);

        if (cudaSetDevice(device_id) != cudaSuccess)
        {
            delete[] xds;
            delete[] yds;
            delete[] zds;
            delete[] xdi;
            delete[] ydi;
            delete[] detZ;
            return CUDA_PROJECTOR_RUNTIME_ERROR;
        }

        int max_rows = 0;
        int xy_mask_count = 0;
        for (int mat = 0; mat < n_materials; ++mat)
        {
            int nx = dims[mat * 3 + 0];
            int ny = dims[mat * 3 + 1];
            if (nx > max_rows)
                max_rows = nx;
            if (ny > max_rows)
                max_rows = ny;
        }
        xy_mask_count = xy_mask_offsets[n_materials];

        int task_count = n_sources * n_materials;
        size_t volume_bytes = (size_t)volume_data_count * sizeof(float);
        size_t volume_offsets_bytes = (size_t)(n_materials + 1) * sizeof(int);
        size_t dims_bytes = (size_t)n_materials * 3 * sizeof(int);
        size_t vector3_bytes = (size_t)n_materials * 3 * sizeof(float);
        size_t mu_bytes = (size_t)n_energy * (size_t)n_materials * sizeof(float);
        size_t xy_mask_bytes = (size_t)xy_mask_count * sizeof(unsigned char);
        size_t src_samples_bytes = (size_t)n_sources * 3 * sizeof(float);
        size_t src_weights_bytes = (size_t)n_sources * sizeof(float);
        size_t xdi_bytes = (size_t)(nrdetcols + 1) * sizeof(float);
        size_t detZ_bytes = (size_t)(nrdetrows + 1) * sizeof(float);
        size_t task_detX_bytes = (size_t)task_count * (size_t)(nrdetcols + 3) * sizeof(float);
        size_t task_scales_bytes = (size_t)task_count * (size_t)(nrdetcols + 2) * sizeof(float);
        size_t task_scalar_bytes = (size_t)task_count * sizeof(float);
        size_t task_int_bytes = (size_t)task_count * sizeof(int);
        size_t views_bytes = (size_t)task_count * (size_t)(nrdetcols + 2) * (size_t)(nrdetrows + 2) * sizeof(float);
        size_t paths_bytes = (size_t)task_count * (size_t)n_pixels * sizeof(float);
        size_t trans_bytes = (size_t)n_pixels * (size_t)n_energy * sizeof(float);

        void *ptrs[22] = {0};
        float *d_volume_data = 0, *d_volume_offsets_xyz = 0, *d_voxel_size = 0, *d_mu = 0;
        float *d_src_samples = 0, *d_src_weights = 0, *d_xdi = 0, *d_ydi = 0, *d_detZ = 0;
        float *d_detX = 0, *d_scales = 0, *d_x0 = 0, *d_y0 = 0, *d_z0 = 0, *d_views = 0, *d_paths = 0, *d_trans = 0;
        int *d_volume_offsets = 0, *d_dims = 0, *d_xy_mask_offsets = 0, *d_vertical = 0;
        unsigned char *d_xy_mask = 0;
        int p = 0;
        cudaError_t err = cudaSuccess;

#define ALLOC_PTR(type, var, bytes)                     \
    do                                                  \
    {                                                   \
        if (err == cudaSuccess)                         \
        {                                               \
            err = cudaMalloc((void **)&(var), (bytes)); \
            ptrs[p++] = (void *)(var);                  \
        }                                               \
    } while (0)

        ALLOC_PTR(float, d_volume_data, volume_bytes);
        ALLOC_PTR(int, d_volume_offsets, volume_offsets_bytes);
        ALLOC_PTR(int, d_dims, dims_bytes);
        ALLOC_PTR(float, d_volume_offsets_xyz, vector3_bytes);
        ALLOC_PTR(float, d_voxel_size, vector3_bytes);
        ALLOC_PTR(float, d_mu, mu_bytes);
        ALLOC_PTR(unsigned char, d_xy_mask, xy_mask_bytes);
        ALLOC_PTR(int, d_xy_mask_offsets, volume_offsets_bytes);
        ALLOC_PTR(float, d_src_samples, src_samples_bytes);
        ALLOC_PTR(float, d_src_weights, src_weights_bytes);
        ALLOC_PTR(float, d_xdi, xdi_bytes);
        ALLOC_PTR(float, d_ydi, xdi_bytes);
        ALLOC_PTR(float, d_detZ, detZ_bytes);
        ALLOC_PTR(float, d_detX, task_detX_bytes);
        ALLOC_PTR(float, d_scales, task_scales_bytes);
        ALLOC_PTR(float, d_x0, task_scalar_bytes);
        ALLOC_PTR(float, d_y0, task_scalar_bytes);
        ALLOC_PTR(float, d_z0, task_scalar_bytes);
        ALLOC_PTR(int, d_vertical, task_int_bytes);
        ALLOC_PTR(float, d_views, views_bytes);
        ALLOC_PTR(float, d_paths, paths_bytes);
        ALLOC_PTR(float, d_trans, trans_bytes);

#undef ALLOC_PTR

        if (err != cudaSuccess)
        {
            cuda_cleanup(ptrs, p);
            delete[] xds;
            delete[] yds;
            delete[] zds;
            delete[] xdi;
            delete[] ydi;
            delete[] detZ;
            return CUDA_PROJECTOR_MEMORY_ERROR;
        }

        err = cudaMemcpy(d_volume_data, volume_data, volume_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_volume_offsets, volume_offsets, volume_offsets_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_dims, dims, dims_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_volume_offsets_xyz, volume_offsets_xyz, vector3_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_voxel_size, voxel_size, vector3_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_mu, mu, mu_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_xy_mask, xy_mask, xy_mask_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_xy_mask_offsets, xy_mask_offsets, volume_offsets_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_src_samples, src_samples, src_samples_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_src_weights, src_weights, src_weights_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_xdi, xdi, xdi_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_ydi, ydi, xdi_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_detZ, detZ, detZ_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemset(d_views, 0, views_bytes);
        if (err != cudaSuccess)
        {
            cuda_cleanup(ptrs, p);
            delete[] xds;
            delete[] yds;
            delete[] zds;
            delete[] xdi;
            delete[] ydi;
            delete[] detZ;
            return CUDA_PROJECTOR_RUNTIME_ERROR;
        }

        int threads = 256;
        int prepare_blocks = (task_count + threads - 1) / threads;
        prepare_dd3_geometry_kernel<<<prepare_blocks, threads>>>(
            d_src_samples, d_volume_offsets_xyz, d_xdi, d_ydi, nrdetcols, n_materials, n_sources,
            d_detX, d_scales, d_x0, d_y0, d_z0, d_vertical);
        err = cudaGetLastError();
        if (err == cudaSuccess)
            err = cudaDeviceSynchronize();

        int row_total = task_count * max_rows;
        int row_blocks = (row_total + threads - 1) / threads;
        if (err == cudaSuccess)
        {
            dd3_row_kernel<<<row_blocks, threads>>>(
                d_volume_data, d_volume_offsets, d_dims, d_volume_offsets_xyz, d_voxel_size,
                d_xy_mask, d_xy_mask_offsets, d_src_samples, d_detZ, d_detX, d_scales,
                d_x0, d_y0, d_z0, d_vertical, nrdetcols, nrdetrows, n_materials, n_sources,
                max_rows, d_views);
            err = cudaGetLastError();
            if (err == cudaSuccess)
                err = cudaDeviceSynchronize();
        }

        int path_total = task_count * n_pixels;
        int path_blocks = (path_total + threads - 1) / threads;
        if (err == cudaSuccess)
        {
            dd3_scale_kernel<<<path_blocks, threads>>>(
                d_detZ, d_src_samples, d_voxel_size, d_detX, d_scales, d_x0, d_y0, d_views,
                nrdetcols, nrdetrows, n_materials, n_pixels, n_sources, d_paths);
            err = cudaGetLastError();
            if (err == cudaSuccess)
                err = cudaDeviceSynchronize();
        }

        int trans_total = n_pixels * n_energy;
        int trans_blocks = (trans_total + threads - 1) / threads;
        if (err == cudaSuccess)
        {
            combine_trans_kernel<<<trans_blocks, threads>>>(
                d_paths, d_mu, d_src_weights, n_materials, n_energy, n_pixels, n_sources, d_trans);
            err = cudaGetLastError();
            if (err == cudaSuccess)
                err = cudaDeviceSynchronize();
            if (err == cudaSuccess)
                err = cudaMemcpy(trans_out, d_trans, trans_bytes, cudaMemcpyDeviceToHost);
        }

        cuda_cleanup(ptrs, p);
        delete[] xds;
        delete[] yds;
        delete[] zds;
        delete[] xdi;
        delete[] ydi;
        delete[] detZ;

        if (err != cudaSuccess)
        {
            return CUDA_PROJECTOR_RUNTIME_ERROR;
        }
        return CUDA_PROJECTOR_SUCCESS;
    }
}
