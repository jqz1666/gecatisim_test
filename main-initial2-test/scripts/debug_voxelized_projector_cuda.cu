// Minimal CUDA debug harness for gecatsim/clib_build/src/voxelized_projector_cuda.cu.
//
// Windows debug build:
//   nvcc -G -g -lineinfo -Xcompiler="/Zi /MD" -I. scripts\debug_voxelized_projector_cuda.cu -o scripts\debug_voxelized_projector_cuda.exe
//
// Linux debug build:
//   nvcc -G -g -lineinfo -I. scripts/debug_voxelized_projector_cuda.cu -o scripts/debug_voxelized_projector_cuda
//
// CUDA memory check:
//   compute-sanitizer --tool memcheck scripts\debug_voxelized_projector_cuda.exe

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../gecatsim/clib_build/src/voxelized_projector_cuda.cu"

static const char *status_name(int status) {
    switch (status) {
        case CUDA_PROJECTOR_SUCCESS:
            return "CUDA_PROJECTOR_SUCCESS";
        case CUDA_PROJECTOR_BAD_ARGUMENT:
            return "CUDA_PROJECTOR_BAD_ARGUMENT";
        case CUDA_PROJECTOR_DETECTOR_ERROR:
            return "CUDA_PROJECTOR_DETECTOR_ERROR";
        case CUDA_PROJECTOR_MEMORY_ERROR:
            return "CUDA_PROJECTOR_MEMORY_ERROR";
        case CUDA_PROJECTOR_RUNTIME_ERROR:
            return "CUDA_PROJECTOR_RUNTIME_ERROR";
        default:
            return "UNKNOWN_STATUS";
    }
}

static void fill_volume(float *volume, int nx, int ny, int nz) {
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int idx = x + nx * (y + ny * z);
                const float cx = x - 0.5f * (nx - 1);
                const float cy = y - 0.5f * (ny - 1);
                const float cz = z - 0.5f * (nz - 1);
                const float r2 = cx * cx + cy * cy + cz * cz;
                volume[idx] = (r2 <= 9.0f) ? 1.0f : 0.0f;
            }
        }
    }
}

int main(int argc, char **argv) {
    int device_id = 0;
    if (argc > 1) {
        device_id = std::atoi(argv[1]);
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr, "No CUDA device is available: %s\n", cudaGetErrorString(err));
        return 2;
    }
    if (device_id < 0 || device_id >= device_count) {
        std::fprintf(stderr, "Invalid device %d. Available device count: %d\n", device_id, device_count);
        return 2;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::printf("Using CUDA device %d: %s\n", device_id, prop.name);

    const int n_materials = 1;
    const int n_energy = 2;
    const int n_modules = 1;
    const int det_cells_per_mod = 4;
    const int n_pixels = 4;
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const int64_t volume_data_count = (int64_t)nx * ny * nz;

    float volume_data[volume_data_count];
    fill_volume(volume_data, nx, ny, nz);

    int64_t volume_offsets[n_materials + 1] = {0, volume_data_count};
    int dims[n_materials * 3] = {nx, ny, nz};
    float volume_offsets_xyz[n_materials * 3] = {-0.5f * (nx - 1), -0.5f * (ny - 1), -0.5f * (nz - 1)};
    float voxel_size[n_materials * 3] = {1.0f, 1.0f, 1.0f};
    float mu[n_energy * n_materials] = {0.02f, 0.04f};

    unsigned char xy_mask[nx * ny * 2];
    for (int i = 0; i < nx * ny * 2; ++i) {
        xy_mask[i] = 1;
    }
    int64_t xy_mask_offsets[n_materials + 1] = {0, (int64_t)nx * ny * 2};

    float sourcePoints[3] = {0.0f, 80.0f, 0.0f};

    // Detector cell coordinates are ordered as row + col * nRows.
    float det_cell_coords[det_cells_per_mod * 2] = {
        -1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f
    };
    int det_mod_types[n_modules] = {0};
    float det_mod_coords[n_modules * 3] = {0.0f, -80.0f, 0.0f};
    float det_uvecs[n_modules * 3] = {1.0f, 0.0f, 0.0f};
    float det_vvecs[n_modules * 3] = {0.0f, 0.0f, 1.0f};
    int det_start_indices[n_modules] = {0};

    float thisView[n_pixels * n_energy];
    for (int i = 0; i < n_pixels * n_energy; ++i) {
        thisView[i] = -1.0f;
    }

    std::printf("Calling voxelized_projector_cuda with a %dx%dx%d volume and 2x2 detector...\n", nx, ny, nz);
    int status = voxelized_projector_cuda(
        volume_data,
        volume_offsets,
        dims,
        volume_offsets_xyz,
        voxel_size,
        mu,
        xy_mask,
        xy_mask_offsets,
        sourcePoints,
        det_cell_coords,
        det_mod_types,
        det_mod_coords,
        det_uvecs,
        det_vvecs,
        det_start_indices,
        det_cells_per_mod,
        n_modules,
        n_materials,
        n_energy,
        n_pixels,
        1,
        1,
        volume_data_count,
        thisView,
        device_id);

    std::printf("Status: %d (%s)\n", status, status_name(status));
    if (status != CUDA_PROJECTOR_SUCCESS) {
        return 1;
    }

    for (int pixel = 0; pixel < n_pixels; ++pixel) {
        std::printf("pixel %d:", pixel);
        for (int energy = 0; energy < n_energy; ++energy) {
            std::printf(" %.8f", thisView[pixel * n_energy + energy]);
        }
        std::printf("\n");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Final cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
