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

struct CudaPhantom_cache
{
    int valid;
    int device_id;

    const float *h_volume_data;
    const int64_t *h_volume_offsets;
    const int *h_dims;
    const float *h_volume_offsets_xyz;
    const float *h_voxel_size;
    const float *h_mu;
    const unsigned char *h_xy_mask;
    const int64_t *h_xy_mask_offsets;

    int n_materials;
    int n_energy;
    int64_t volume_data_count;
    int64_t xy_mask_count;

    float *d_volume_data;
    int64_t *d_volume_offsets;
    int *d_dims;
    float *d_volume_offsets_xyz;
    float *d_voxel_size;
    float *d_mu;
    unsigned char *d_xy_mask;
    int64_t *d_xy_mask_offsets;
};

static CudaPhantom_cache g_cuda_phantom = {0};

static void clear_phantom_cache()
{
    cudaFree(g_cuda_phantom.d_volume_data);
    cudaFree(g_cuda_phantom.d_volume_offsets);
    cudaFree(g_cuda_phantom.d_dims);
    cudaFree(g_cuda_phantom.d_volume_offsets_xyz);
    cudaFree(g_cuda_phantom.d_voxel_size);
    cudaFree(g_cuda_phantom.d_mu);
    cudaFree(g_cuda_phantom.d_xy_mask);
    cudaFree(g_cuda_phantom.d_xy_mask_offsets);

    memset(&g_cuda_phantom, 0, sizeof(g_cuda_phantom));
}

static int ensure_phantom_cache(
    int device_id,
    const float *volume_data,
    const int64_t *volume_offsets,
    const int *dims,
    const float *volume_offsets_xyz,
    const float *voxel_size,
    const float *mu,
    const unsigned char *xy_mask,
    const int64_t *xy_mask_offsets,
    int n_materials,
    int n_energy,
    int64_t volume_data_count,
    int64_t xy_mask_count)
{
    if (g_cuda_phantom.valid &&
        g_cuda_phantom.device_id == device_id &&
        g_cuda_phantom.h_volume_data == volume_data &&
        g_cuda_phantom.h_volume_offsets == volume_offsets &&
        g_cuda_phantom.h_dims == dims &&
        g_cuda_phantom.h_volume_offsets_xyz == volume_offsets_xyz &&
        g_cuda_phantom.h_voxel_size == voxel_size &&
        g_cuda_phantom.h_mu == mu &&
        g_cuda_phantom.h_xy_mask == xy_mask &&
        g_cuda_phantom.h_xy_mask_offsets == xy_mask_offsets &&
        g_cuda_phantom.n_materials == n_materials &&
        g_cuda_phantom.n_energy == n_energy &&
        g_cuda_phantom.volume_data_count == volume_data_count &&
        g_cuda_phantom.xy_mask_count == xy_mask_count)
    {
        return CUDA_PROJECTOR_SUCCESS;
    }

    clear_phantom_cache();

    g_cuda_phantom.device_id = device_id;
    g_cuda_phantom.h_volume_data = volume_data;
    g_cuda_phantom.h_volume_offsets = volume_offsets;
    g_cuda_phantom.h_dims = dims;
    g_cuda_phantom.h_volume_offsets_xyz = volume_offsets_xyz;
    g_cuda_phantom.h_voxel_size = voxel_size;
    g_cuda_phantom.h_mu = mu;
    g_cuda_phantom.h_xy_mask = xy_mask;
    g_cuda_phantom.h_xy_mask_offsets = xy_mask_offsets;
    g_cuda_phantom.n_materials = n_materials;
    g_cuda_phantom.n_energy = n_energy;
    g_cuda_phantom.volume_data_count = volume_data_count;
    g_cuda_phantom.xy_mask_count = xy_mask_count;

    size_t volume_bytes = (size_t)volume_data_count * sizeof(float);
    size_t volume_offsets_bytes = (size_t)(n_materials + 1) * sizeof(int64_t);
    size_t dims_bytes = (size_t)n_materials * 3 * sizeof(int);
    size_t vector3_bytes = (size_t)n_materials * 3 * sizeof(float);
    size_t mu_bytes = (size_t)n_energy * (size_t)n_materials * sizeof(float);
    size_t xy_mask_bytes = (size_t)xy_mask_count * sizeof(unsigned char);

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void **)&g_cuda_phantom.d_volume_data, volume_bytes);
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&g_cuda_phantom.d_volume_offsets, volume_offsets_bytes);
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&g_cuda_phantom.d_dims, dims_bytes);
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&g_cuda_phantom.d_volume_offsets_xyz, vector3_bytes);
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&g_cuda_phantom.d_voxel_size, vector3_bytes);
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&g_cuda_phantom.d_mu, mu_bytes);
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&g_cuda_phantom.d_xy_mask, xy_mask_bytes);
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&g_cuda_phantom.d_xy_mask_offsets, volume_offsets_bytes);

    if (err != cudaSuccess)
    {
        clear_phantom_cache();
        return CUDA_PROJECTOR_MEMORY_ERROR;
    }

    err = cudaMemcpy(g_cuda_phantom.d_volume_data, volume_data, volume_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(g_cuda_phantom.d_volume_offsets, volume_offsets, volume_offsets_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(g_cuda_phantom.d_dims, dims, dims_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(g_cuda_phantom.d_volume_offsets_xyz, volume_offsets_xyz, vector3_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(g_cuda_phantom.d_voxel_size, voxel_size, vector3_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(g_cuda_phantom.d_mu, mu, mu_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(g_cuda_phantom.d_xy_mask, xy_mask, xy_mask_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(g_cuda_phantom.d_xy_mask_offsets, xy_mask_offsets, volume_offsets_bytes, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        clear_phantom_cache();
        return CUDA_PROJECTOR_RUNTIME_ERROR;
    }

    g_cuda_phantom.valid = 1;
    return CUDA_PROJECTOR_SUCCESS;
}

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

static void prepare_dd3_geometry_host(
    const float *src_samples,
    const float *volume_offsets_xyz,
    const float *xdi,
    const float *ydi,
    int nrdetcols,
    int mat_id,
    int src_id,
    float *detX,
    float *scales,
    float *x0_local,
    float *y0_local,
    float *z0_local,
    int *vertical_flag)
{
    float xoff = volume_offsets_xyz[mat_id * 3 + 0];
    float yoff = volume_offsets_xyz[mat_id * 3 + 1];
    float zoff = volume_offsets_xyz[mat_id * 3 + 2];
    float x0 = src_samples[src_id * 3 + 0];
    float y0 = src_samples[src_id * 3 + 1];
    float z0 = src_samples[src_id * 3 + 2];
    int vertical = fabsf(y0) >= fabsf(x0);
    x0 -= xoff;
    y0 -= yoff;
    z0 -= zoff;

    detX[0] = 1.0e12f;
    if (vertical)
    {
        for (int i = 0; i < nrdetcols + 1; ++i)
        {
            float xr = xdi[i] - xoff;
            float yr = ydi[i] - yoff;
            detX[i + 1] = (x0 * yr - xr * y0) / (yr - y0);
            scales[i + 1] = y0 / (y0 - yr);
        }
    }
    else
    {
        for (int i = 0; i < nrdetcols + 1; ++i)
        {
            float xr = xdi[i] - xoff;
            float yr = ydi[i] - yoff;
            detX[i + 1] = -(y0 * xr - yr * x0) / (xr - x0);
            scales[i + 1] = x0 / (x0 - xr);
        }
        float old_x0 = x0;
        x0 = -y0;
        y0 = -old_x0;
    }

    detX[nrdetcols + 2] = 1.0e12f;
    scales[0] = scales[1];
    for (int i = 1; i <= nrdetcols; ++i)
    {
        scales[i] = 0.5f * (scales[i] + scales[i + 1]);
    }

    *x0_local = x0;
    *y0_local = y0;
    *z0_local = z0;
    *vertical_flag = vertical;
}

__device__ void dd3_project_row_device(
    float imgX,
    float imgXstep,
    int nrcols,
    float imgZstart,
    float imgZstep,
    int nrplanes,
    const float *pImg,
    int64_t pImgBase,
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

    int64_t pImgOffset = pImgBase;
    int view_col = 0;
    if (detX_0 > previousX)
    {
        start_imgX_ind = (int)((detX_0 - imgXStart) * inv_imgXstep - 1.0f);
        if (start_imgX_ind > 0)
        {
            pImgOffset += (int64_t)start_imgX_ind * nrplanes * trans_rowstep;
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
            int64_t pImgCopyOffset = pImgOffset;

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
    const int64_t *volume_offsets,
    const int *dims,
    const float *volume_offsets_xyz,
    const float *voxel_size,
    const unsigned char *xy_mask,
    const int64_t *xy_mask_offsets,
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
    int mat_id,
    int src_id,
    float *views)
{
    int rownr = blockIdx.x * blockDim.x + threadIdx.x;
    int nx = dims[mat_id * 3 + 0];
    int ny = dims[mat_id * 3 + 1];
    int nz = dims[mat_id * 3 + 2];
    float vox_xy = voxel_size[mat_id * 3 + 0];
    float vox_z = voxel_size[mat_id * 3 + 2];
    int vertical = vertical_flags[0];
    int nrcols = vertical ? nx : ny;
    int nrrows = vertical ? ny : nx;
    if (rownr >= nrrows)
    {
        return;
    }

    float x0 = x0_local[0];
    float y0 = y0_local[0];
    float z0 = z0_local[0];
    float mag_fac = y0 / (y0 - ((nrrows - 1) * 0.5f - rownr) * vox_xy);
    float imgXstep = mag_fac * vox_xy;
    float imgX = -x0 / (y0 - ((nrrows - 1) * 0.5f - rownr) * vox_xy) *
                     ((nrrows - 1) * 0.5f - rownr) * vox_xy -
                 (nrcols * 0.5f * vox_xy) * mag_fac;
    float imgZstep = mag_fac * vox_z;
    float imgZ = z0 - (nz * 0.5f * vox_z + z0) * mag_fac;

    int64_t pImgBase;
    int trans_rowstep;
    const unsigned char *mask_row;
    if (vertical)
    {
        pImgBase = volume_offsets[mat_id] + (int64_t)rownr * nx * nz;
        trans_rowstep = 1;
        mask_row = xy_mask + xy_mask_offsets[mat_id] + rownr * nx;
    }
    else
    {
        pImgBase = volume_offsets[mat_id] + (int64_t)rownr * nz;
        trans_rowstep = nx;
        mask_row = xy_mask + xy_mask_offsets[mat_id] + ny * nx + rownr * ny;
    }

    const float *task_detX = detX;
    const float *task_scales = scales;
    float *task_view = views;
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
    int mat_id,
    int n_pixels,
    int src_id,
    float *paths)
{
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= n_pixels)
    {
        return;
    }

    int det_col = pixel / nrdetrows;
    int det_row = pixel - det_col * nrdetrows;
    const float *task_detX = detX;
    const float *task_scales = scales;
    const float *task_view = views;
    int view_idx = (det_col + 1) * (nrdetrows + 2) + det_row + 1;
    float view_value = task_view[view_idx];
    if (view_value == 0.0f)
    {
        paths[pixel] = 0.0f;
        return;
    }

    float x0 = x0_local[0];
    float y0 = y0_local[0];
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
    paths[pixel] = invCos / (detXstep * detZstep) * view_value;
}

__global__ void combine_trans_kernel(
    const float *paths,
    const float *mu,
    int MaterialIndex,
    int n_materials,
    int n_energy,
    int n_pixels,
    float *thisView)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_pixels * n_energy;
    if (idx >= total)
    {
        return;
    }

    int energy = idx % n_energy;
    int pixel = idx / n_energy;
    thisView[idx] = paths[pixel] * mu[energy * n_materials + (MaterialIndex - 1)];
}

__global__ void accumulate_pvalue_kernel(
    const float *paths,
    const float *mu,
    int mat_id,
    int n_materials,
    int n_energy,
    int n_pixels,
    float *pvalue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_pixels * n_energy;
    if (idx >= total)
    {
        return;
    }

    int energy = idx % n_energy;
    int pixel = idx / n_energy;
    pvalue[idx] += paths[pixel] * mu[energy * n_materials + mat_id];
}

__global__ void apply_source_trans_kernel(
    const float *pvalue,
    const float *src_weights,
    int src_id,
    int total,
    float *trans)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }

    trans[idx] += src_weights[src_id] * expf(-pvalue[idx]);
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
    void voxelized_projector_cuda_clear_cache()
    {
        clear_phantom_cache();
    }

    DLLEXPORT
    int voxelized_projector_cuda_batch(
        const float *volume_data,
        const int64_t *volume_offsets,
        const int *dims,
        const float *volume_offsets_xyz,
        const float *voxel_size,
        const float *mu,
        const unsigned char *xy_mask,
        const int64_t *xy_mask_offsets,
        const float *sourcePoints,
        const float *sourceWeights,
        int n_source_samples,
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
        int64_t volume_data_count,
        float *trans,
        int device_id)
    {
        (void)det_start_indices;

        if (!volume_data || !volume_offsets || !dims || !volume_offsets_xyz || !voxel_size || !mu ||
            !xy_mask || !xy_mask_offsets || !sourcePoints || !sourceWeights || !det_cell_coords ||
            !det_mod_types || !det_mod_coords || !det_uvecs || !det_vvecs || !trans)
        {
            return CUDA_PROJECTOR_BAD_ARGUMENT;
        }
        if (n_source_samples <= 0 || det_cells_per_mod <= 0 || n_modules <= 0 || n_materials <= 0 ||
            n_energy <= 0 || n_pixels <= 0 || volume_data_count <= 0)
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

        const int detZ_count = nrdetrows + 1;
        float *xdi = new float[nrdetcols + 1];
        float *ydi = new float[nrdetcols + 1];
        float *detZ = new float[detZ_count + 1];
        detZ[detZ_count] = 1.e12f;
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

        int64_t xy_mask_count = xy_mask_offsets[n_materials];
        int cache_status = ensure_phantom_cache(
            device_id,
            volume_data,
            volume_offsets,
            dims,
            volume_offsets_xyz,
            voxel_size,
            mu,
            xy_mask,
            xy_mask_offsets,
            n_materials,
            n_energy,
            volume_data_count,
            xy_mask_count);

        if (cache_status != CUDA_PROJECTOR_SUCCESS)
        {
            delete[] xds;
            delete[] yds;
            delete[] zds;
            delete[] xdi;
            delete[] ydi;
            delete[] detZ;
            return cache_status;
        }

        float *detX = new float[nrdetcols + 3];
        float *scales = new float[nrdetcols + 2];
        if (!detX || !scales)
        {
            delete[] xds;
            delete[] yds;
            delete[] zds;
            delete[] xdi;
            delete[] ydi;
            delete[] detZ;
            delete[] detX;
            delete[] scales;
            return CUDA_PROJECTOR_MEMORY_ERROR;
        }

        size_t sourcePoints_bytes = (size_t)n_source_samples * 3 * sizeof(float);
        size_t sourceWeights_bytes = (size_t)n_source_samples * sizeof(float);
        size_t detZ_bytes = (size_t)(detZ_count + 1) * sizeof(float);
        size_t detX_bytes = (size_t)(nrdetcols + 3) * sizeof(float);
        size_t scales_bytes = (size_t)(nrdetcols + 2) * sizeof(float);
        size_t scalar_bytes = sizeof(float);
        size_t int_bytes = sizeof(int);
        size_t views_bytes = (size_t)(nrdetcols + 2) * (size_t)(nrdetrows + 2) * sizeof(float);
        size_t paths_bytes = (size_t)n_pixels * sizeof(float);
        size_t spectrum_bytes = (size_t)n_pixels * (size_t)n_energy * sizeof(float);

        void *ptrs[16] = {0};
        float *d_sourcePoints = 0, *d_sourceWeights = 0, *d_detZ = 0;
        float *d_detX = 0, *d_scales = 0, *d_x0 = 0, *d_y0 = 0, *d_z0 = 0;
        float *d_views = 0, *d_paths = 0, *d_pvalue = 0, *d_trans = 0;
        int *d_vertical = 0;
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

        ALLOC_PTR(float, d_sourcePoints, sourcePoints_bytes);
        ALLOC_PTR(float, d_sourceWeights, sourceWeights_bytes);
        ALLOC_PTR(float, d_detZ, detZ_bytes);
        ALLOC_PTR(float, d_detX, detX_bytes);
        ALLOC_PTR(float, d_scales, scales_bytes);
        ALLOC_PTR(float, d_x0, scalar_bytes);
        ALLOC_PTR(float, d_y0, scalar_bytes);
        ALLOC_PTR(float, d_z0, scalar_bytes);
        ALLOC_PTR(int, d_vertical, int_bytes);
        ALLOC_PTR(float, d_views, views_bytes);
        ALLOC_PTR(float, d_paths, paths_bytes);
        ALLOC_PTR(float, d_pvalue, spectrum_bytes);
        ALLOC_PTR(float, d_trans, spectrum_bytes);

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
            delete[] detX;
            delete[] scales;
            return CUDA_PROJECTOR_MEMORY_ERROR;
        }

        err = cudaMemcpy(d_sourcePoints, sourcePoints, sourcePoints_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_sourceWeights, sourceWeights, sourceWeights_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_detZ, detZ, detZ_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemset(d_trans, 0, spectrum_bytes);

        int threads = 256;
        int spectrum_total = n_pixels * n_energy;
        int spectrum_blocks = (spectrum_total + threads - 1) / threads;
        int path_blocks = (n_pixels + threads - 1) / threads;

        for (int src_id = 0; err == cudaSuccess && src_id < n_source_samples; ++src_id)
        {
            err = cudaMemset(d_pvalue, 0, spectrum_bytes);
            for (int mat_id = 0; err == cudaSuccess && mat_id < n_materials; ++mat_id)
            {
                float x0_local = 0.0f;
                float y0_local = 0.0f;
                float z0_local = 0.0f;
                int vertical_flag = 0;
                prepare_dd3_geometry_host(
                    sourcePoints, volume_offsets_xyz, xdi, ydi, nrdetcols, mat_id, src_id,
                    detX, scales, &x0_local, &y0_local, &z0_local, &vertical_flag);

                err = cudaMemcpy(d_detX, detX, detX_bytes, cudaMemcpyHostToDevice);
                if (err == cudaSuccess)
                    err = cudaMemcpy(d_scales, scales, scales_bytes, cudaMemcpyHostToDevice);
                if (err == cudaSuccess)
                    err = cudaMemcpy(d_x0, &x0_local, scalar_bytes, cudaMemcpyHostToDevice);
                if (err == cudaSuccess)
                    err = cudaMemcpy(d_y0, &y0_local, scalar_bytes, cudaMemcpyHostToDevice);
                if (err == cudaSuccess)
                    err = cudaMemcpy(d_z0, &z0_local, scalar_bytes, cudaMemcpyHostToDevice);
                if (err == cudaSuccess)
                    err = cudaMemcpy(d_vertical, &vertical_flag, int_bytes, cudaMemcpyHostToDevice);
                if (err == cudaSuccess)
                    err = cudaMemset(d_views, 0, views_bytes);

                int nx = dims[mat_id * 3 + 0];
                int ny = dims[mat_id * 3 + 1];
                int nrrows = vertical_flag ? ny : nx;
                int row_blocks = (nrrows + threads - 1) / threads;
                if (err == cudaSuccess)
                {
                    dd3_row_kernel<<<row_blocks, threads>>>(
                        g_cuda_phantom.d_volume_data, g_cuda_phantom.d_volume_offsets, g_cuda_phantom.d_dims, g_cuda_phantom.d_volume_offsets_xyz, g_cuda_phantom.d_voxel_size,
                        g_cuda_phantom.d_xy_mask, g_cuda_phantom.d_xy_mask_offsets, d_sourcePoints, d_detZ, d_detX, d_scales,
                        d_x0, d_y0, d_z0, d_vertical, nrdetcols, nrdetrows, mat_id, src_id, d_views);
                    err = cudaGetLastError();
                }

                if (err == cudaSuccess)
                {
                    dd3_scale_kernel<<<path_blocks, threads>>>(
                        d_detZ, d_sourcePoints, g_cuda_phantom.d_voxel_size, d_detX, d_scales, d_x0, d_y0, d_views,
                        nrdetcols, nrdetrows, mat_id, n_pixels, src_id, d_paths);
                    err = cudaGetLastError();
                }

                if (err == cudaSuccess)
                {
                    accumulate_pvalue_kernel<<<spectrum_blocks, threads>>>(
                        d_paths, g_cuda_phantom.d_mu, mat_id, n_materials, n_energy, n_pixels, d_pvalue);
                    err = cudaGetLastError();
                }
            }

            if (err == cudaSuccess)
            {
                apply_source_trans_kernel<<<spectrum_blocks, threads>>>(
                    d_pvalue, d_sourceWeights, src_id, spectrum_total, d_trans);
                err = cudaGetLastError();
            }
        }

        if (err == cudaSuccess)
        {
            err = cudaMemcpy(trans, d_trans, spectrum_bytes, cudaMemcpyDeviceToHost);
        }

        cuda_cleanup(ptrs, p);
        delete[] xds;
        delete[] yds;
        delete[] zds;
        delete[] xdi;
        delete[] ydi;
        delete[] detZ;
        delete[] detX;
        delete[] scales;

        if (err != cudaSuccess)
        {
            return CUDA_PROJECTOR_RUNTIME_ERROR;
        }
        return CUDA_PROJECTOR_SUCCESS;
    }
}

extern "C"
{
    DLLEXPORT
    int voxelized_projector_cuda(
        const float *volume_data,
        const int64_t *volume_offsets,
        const int *dims,
        const float *volume_offsets_xyz,
        const float *voxel_size,
        const float *mu,
        const unsigned char *xy_mask,
        const int64_t *xy_mask_offsets,
        const float *sourcePoints,
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
        int MaterialIndex,
        int MaterialIndexInMemory,
        int64_t volume_data_count,
        float *thisView,
        int device_id)
    {
        (void)det_start_indices;

        if (!volume_data || !volume_offsets || !dims || !volume_offsets_xyz || !voxel_size || !mu ||
            !xy_mask || !xy_mask_offsets || !sourcePoints || !det_cell_coords ||
            !det_mod_types || !det_mod_coords || !det_uvecs || !det_vvecs || !thisView)
        {
            return CUDA_PROJECTOR_BAD_ARGUMENT;
        }
        if (det_cells_per_mod <= 0 || n_modules <= 0 || n_materials <= 0 || n_energy <= 0 ||
            n_pixels <= 0 || MaterialIndex <= 0 || MaterialIndex > n_materials ||
            MaterialIndexInMemory <= 0 || MaterialIndexInMemory > n_materials || volume_data_count <= 0)
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

        const int detZ_count = nrdetrows + 1;
        float *xdi = new float[nrdetcols + 1];
        float *ydi = new float[nrdetcols + 1];
        float *detZ = new float[detZ_count + 1];
        detZ[detZ_count] = 1.e12;
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

        int64_t xy_mask_count = xy_mask_offsets[n_materials];

        int cache_status = ensure_phantom_cache(
            device_id,
            volume_data,
            volume_offsets,
            dims,
            volume_offsets_xyz,
            voxel_size,
            mu,
            xy_mask,
            xy_mask_offsets,
            n_materials,
            n_energy,
            volume_data_count,
            xy_mask_count);

        if (cache_status != CUDA_PROJECTOR_SUCCESS)
        {
            delete[] xds;
            delete[] yds;
            delete[] zds;
            delete[] xdi;
            delete[] ydi;
            delete[] detZ;
            return cache_status;
        }

        float *detX = new float[nrdetcols + 3];
        float *scales = new float[nrdetcols + 2];
        if (!detX || !scales)
        {
            delete[] xds;
            delete[] yds;
            delete[] zds;
            delete[] xdi;
            delete[] ydi;
            delete[] detZ;
            delete[] detX;
            delete[] scales;
            return CUDA_PROJECTOR_MEMORY_ERROR;
        }

        // size_t volume_bytes = (size_t)volume_data_count * sizeof(float);
        // size_t volume_offsets_bytes = (size_t)(n_materials + 1) * sizeof(int64_t);
        // size_t dims_bytes = (size_t)n_materials * 3 * sizeof(int);
        // size_t vector3_bytes = (size_t)n_materials * 3 * sizeof(float);
        // size_t mu_bytes = (size_t)n_energy * (size_t)n_materials * sizeof(float);
        // size_t xy_mask_bytes = (size_t)xy_mask_count * sizeof(unsigned char);
        size_t sourcePoints_bytes = 3 * sizeof(float);
        size_t detZ_bytes = (size_t)(detZ_count + 1) * sizeof(float);
        size_t detX_bytes = (size_t)(nrdetcols + 3) * sizeof(float);
        size_t scales_bytes = (size_t)(nrdetcols + 2) * sizeof(float);
        size_t scalar_bytes = sizeof(float);
        size_t int_bytes = sizeof(int);
        size_t views_bytes = (size_t)(nrdetcols + 2) * (size_t)(nrdetrows + 2) * sizeof(float);
        size_t paths_bytes = (size_t)n_pixels * sizeof(float);
        size_t thisView_bytes = (size_t)n_pixels * (size_t)n_energy * sizeof(float);

        void *ptrs[12] = {0};
        // float *d_volume_data = 0, *d_volume_offsets_xyz = 0, *d_voxel_size = 0, *d_mu = 0;
        float *d_sourcePoints = 0, *d_detZ = 0;
        float *d_detX = 0, *d_scales = 0, *d_x0 = 0, *d_y0 = 0, *d_z0 = 0, *d_views = 0, *d_paths = 0, *d_thisView = 0;
        int *d_vertical = 0;
        // int64_t *d_volume_offsets = 0, *d_xy_mask_offsets = 0;
        // unsigned char *d_xy_mask = 0;
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

        // ALLOC_PTR(float, d_volume_data, volume_bytes);
        // ALLOC_PTR(int64_t, d_volume_offsets, volume_offsets_bytes);
        // ALLOC_PTR(int, d_dims, dims_bytes);
        // ALLOC_PTR(float, d_volume_offsets_xyz, vector3_bytes);
        // ALLOC_PTR(float, d_voxel_size, vector3_bytes);
        // ALLOC_PTR(float, d_mu, mu_bytes);
        // ALLOC_PTR(unsigned char, d_xy_mask, xy_mask_bytes);
        // ALLOC_PTR(int64_t, d_xy_mask_offsets, volume_offsets_bytes);
        ALLOC_PTR(float, d_sourcePoints, sourcePoints_bytes);
        ALLOC_PTR(float, d_detZ, detZ_bytes);
        ALLOC_PTR(float, d_detX, detX_bytes);
        ALLOC_PTR(float, d_scales, scales_bytes);
        ALLOC_PTR(float, d_x0, scalar_bytes);
        ALLOC_PTR(float, d_y0, scalar_bytes);
        ALLOC_PTR(float, d_z0, scalar_bytes);
        ALLOC_PTR(int, d_vertical, int_bytes);
        ALLOC_PTR(float, d_views, views_bytes);
        ALLOC_PTR(float, d_paths, paths_bytes);
        ALLOC_PTR(float, d_thisView, thisView_bytes);

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
            delete[] detX;
            delete[] scales;
            return CUDA_PROJECTOR_MEMORY_ERROR;
        }

        // err = cudaMemcpy(d_volume_data, volume_data, volume_bytes, cudaMemcpyHostToDevice);
        // if (err == cudaSuccess)
        //     err = cudaMemcpy(d_volume_offsets, volume_offsets, volume_offsets_bytes, cudaMemcpyHostToDevice);
        // if (err == cudaSuccess)
        //     err = cudaMemcpy(d_dims, dims, dims_bytes, cudaMemcpyHostToDevice);
        // if (err == cudaSuccess)
        //     err = cudaMemcpy(d_volume_offsets_xyz, volume_offsets_xyz, vector3_bytes, cudaMemcpyHostToDevice);
        // if (err == cudaSuccess)
        //     err = cudaMemcpy(d_voxel_size, voxel_size, vector3_bytes, cudaMemcpyHostToDevice);
        // if (err == cudaSuccess)
        //     err = cudaMemcpy(d_mu, mu, mu_bytes, cudaMemcpyHostToDevice);
        // if (err == cudaSuccess)
        //     err = cudaMemcpy(d_xy_mask, xy_mask, xy_mask_bytes, cudaMemcpyHostToDevice);
        // if (err == cudaSuccess)
        //     err = cudaMemcpy(d_xy_mask_offsets, xy_mask_offsets, volume_offsets_bytes, cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_sourcePoints, sourcePoints, sourcePoints_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_detZ, detZ, detZ_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cuda_cleanup(ptrs, p);
            delete[] xds;
            delete[] yds;
            delete[] zds;
            delete[] xdi;
            delete[] ydi;
            delete[] detZ;
            delete[] detX;
            delete[] scales;
            return CUDA_PROJECTOR_RUNTIME_ERROR;
        }

        int threads = 256;
        int view_total = n_pixels * n_energy;
        int view_blocks = (view_total + threads - 1) / threads;
        int path_blocks = (n_pixels + threads - 1) / threads;

        int mat_id = MaterialIndexInMemory - 1;
        float x0_local = 0.0f;
        float y0_local = 0.0f;
        float z0_local = 0.0f;
        int vertical_flag = 0;
        prepare_dd3_geometry_host(
            sourcePoints, volume_offsets_xyz, xdi, ydi, nrdetcols, mat_id, 0,
            detX, scales, &x0_local, &y0_local, &z0_local, &vertical_flag);

        err = cudaMemcpy(d_detX, detX, detX_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_scales, scales, scales_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_x0, &x0_local, scalar_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_y0, &y0_local, scalar_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_z0, &z0_local, scalar_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_vertical, &vertical_flag, int_bytes, cudaMemcpyHostToDevice);
        if (err == cudaSuccess)
            err = cudaMemset(d_views, 0, views_bytes);

        int nx = dims[mat_id * 3 + 0];
        int ny = dims[mat_id * 3 + 1];
        int nrrows = vertical_flag ? ny : nx;
        int row_blocks = (nrrows + threads - 1) / threads;
        if (err == cudaSuccess)
        {
            dd3_row_kernel<<<row_blocks, threads>>>(
                g_cuda_phantom.d_volume_data, g_cuda_phantom.d_volume_offsets, g_cuda_phantom.d_dims, g_cuda_phantom.d_volume_offsets_xyz, g_cuda_phantom.d_voxel_size,
                g_cuda_phantom.d_xy_mask, g_cuda_phantom.d_xy_mask_offsets, d_sourcePoints, d_detZ, d_detX, d_scales,
                d_x0, d_y0, d_z0, d_vertical, nrdetcols, nrdetrows, mat_id, 0, d_views);
            err = cudaGetLastError();
        }

        if (err == cudaSuccess)
        {
            dd3_scale_kernel<<<path_blocks, threads>>>(
                d_detZ, d_sourcePoints, g_cuda_phantom.d_voxel_size, d_detX, d_scales, d_x0, d_y0, d_views,
                nrdetcols, nrdetrows, mat_id, n_pixels, 0, d_paths);
            err = cudaGetLastError();
        }

        if (err == cudaSuccess)
        {
            combine_trans_kernel<<<view_blocks, threads>>>(
                d_paths, g_cuda_phantom.d_mu, MaterialIndex, n_materials, n_energy, n_pixels, d_thisView);
            err = cudaGetLastError();
        }

        if (err == cudaSuccess)
        {
            err = cudaGetLastError();
            if (err == cudaSuccess)
                err = cudaMemcpy(thisView, d_thisView, thisView_bytes, cudaMemcpyDeviceToHost);
        }

        cuda_cleanup(ptrs, p);
        delete[] xds;
        delete[] yds;
        delete[] zds;
        delete[] xdi;
        delete[] ydi;
        delete[] detZ;
        delete[] detX;
        delete[] scales;

        if (err != cudaSuccess)
        {
            return CUDA_PROJECTOR_RUNTIME_ERROR;
        }
        return CUDA_PROJECTOR_SUCCESS;
    }
}
