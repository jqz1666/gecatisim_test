import json
import tempfile
import unittest
from ctypes import CFUNCTYPE, POINTER, c_float, c_int, c_ubyte, c_ushort, c_void_p, cast, addressof
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gecatsim.pyfiles.Phantom_Voxelized import Phantom_Voxelized
from gecatsim.pyfiles.C_Projector_Voxelized_CUDA import (
    is_cuda_projector_available,
    _get_cpu_phantom_data,
    _pointer_signature,
)


class TestVoxelizedGPU(unittest.TestCase):

    def test_phantom_voxelized_does_not_build_gpu_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vp_path = Path(tmp_dir) / "tiny.json"
            vp = {
                "mat_name": ["water", "air"],
                "n_materials": 2,
                "volumefractionmap_filename": ["water.bin", "air.bin"],
                "volumefractionmap_datatype": ["float", "float"],
                "cols": [2, 2],
                "rows": [3, 3],
                "slices": [4, 4],
                "x_size": [1.0, 1.0],
                "y_size": [1.0, 1.0],
                "z_size": [2.0, 2.0],
                "x_offset": [1.5, 1.5],
                "y_offset": [2.0, 2.0],
                "z_offset": [2.5, 2.5],
            }
            vp_path.write_text(json.dumps(vp), encoding="utf-8")

            cfg = SimpleNamespace(
                phantom=SimpleNamespace(filename=str(vp_path), scale=1.0, centerOffset=[0, 0, 0]),
                spec=SimpleNamespace(Evec=np.array([20, 40], dtype=np.float32)),
                clib=SimpleNamespace(),
            )
            mus = np.array([[0.2, 0.01], [0.1, 0.005]], dtype=np.float32)
            volume = np.arange(24, dtype=np.float32)

            with patch("gecatsim.pyfiles.Phantom_Voxelized.my_path.find", return_value=str(vp_path)), \
                    patch("gecatsim.pyfiles.Phantom_Voxelized.set_material", return_value=mus), \
                    patch("gecatsim.pyfiles.Phantom_Voxelized.rawread", return_value=volume), \
                    patch("gecatsim.pyfiles.Phantom_Voxelized.set_voxelized_volume"):
                cfg = Phantom_Voxelized(cfg)

        self.assertFalse(hasattr(cfg.phantom, "gpuVoxelData"))
        self.assertFalse(hasattr(cfg.phantom, "_cudaProjectorHostCache"))

    def test_native_cuda_projector_unavailable_without_cpu_phantom_interface(self):
        cfg = SimpleNamespace(phantom=SimpleNamespace(), physics=SimpleNamespace(), clib=SimpleNamespace())
        self.assertFalse(is_cuda_projector_available(cfg))

    def test_native_cuda_projector_available_for_uint16_phantom(self):
        cfg = SimpleNamespace(phantom=SimpleNamespace(), physics=SimpleNamespace(), clib=SimpleNamespace())
        with patch("gecatsim.pyfiles.C_Projector_Voxelized_CUDA._load_cuda_lib"), \
                patch("gecatsim.pyfiles.C_Projector_Voxelized_CUDA._get_cpu_phantom_data",
                      return_value={"volume_data_type": 1}):
            self.assertTrue(is_cuda_projector_available(cfg))

    def test_cuda_host_metadata_references_cpu_uint16_volume_without_copying_it(self):
        volume = (c_ushort * 8)(*range(8))
        mask = (c_ubyte * 8)(*([1] * 8))
        mus = (c_float * 2)(0.2, 0.1)

        getter_type = CFUNCTYPE(
            c_int, c_int, POINTER(c_void_p), POINTER(c_int), POINTER(c_float), POINTER(c_float),
            POINTER(POINTER(c_ubyte)), POINTER(POINTER(c_float)), POINTER(c_int), POINTER(c_int), POINTER(c_int)
        )

        def getter(material_index, volume_out, dims_out, offsets_out, voxel_size_out,
                   mask_out, mu_out, material_count_out, energy_count_out, data_type_out):
            volume_out[0] = cast(volume, c_void_p)
            mask_out[0] = cast(mask, POINTER(c_ubyte))
            mu_out[0] = cast(mus, POINTER(c_float))
            for axis, dim in enumerate((2, 2, 2)):
                dims_out[axis] = dim
                offsets_out[axis] = 0.0
                voxel_size_out[axis] = 1.0
            material_count_out[0] = 1
            energy_count_out[0] = 2
            data_type_out[0] = 1
            return 0

        cfg = SimpleNamespace(
            phantom=SimpleNamespace(numberOfMaterials=1),
            clib=SimpleNamespace(get_phantom_info_vox_for_cuda=getter_type(getter)),
        )
        host = _get_cpu_phantom_data(cfg)

        self.assertEqual(host["volume_data_type"], 1)
        self.assertEqual(host["volume_pointers"][0], cast(volume, c_void_p).value)
        self.assertNotIn("volume_data", host)

    def test_pointer_signature_is_stable_and_sensitive_to_cpu_addresses(self):
        self.assertEqual(_pointer_signature([1, 2, 3]), _pointer_signature([1, 2, 3]))
        self.assertNotEqual(_pointer_signature([1, 2, 3]), _pointer_signature([1, 2, 4]))

    def test_cuda_host_metadata_references_cpu_volume_without_copying_it(self):
        volumes = [(c_float * 8)(*range(8)), (c_float * 4)(*range(4))]
        masks = [(c_ubyte * 8)(*([1] * 8)), (c_ubyte * 4)(*([1] * 4))]
        mus = (c_float * 4)(0.2, 0.01, 0.1, 0.005)
        material_dims = ((2, 2, 2), (2, 1, 2))

        getter_type = CFUNCTYPE(
            c_int, c_int, POINTER(c_void_p), POINTER(c_int), POINTER(c_float), POINTER(c_float),
            POINTER(POINTER(c_ubyte)), POINTER(POINTER(c_float)), POINTER(c_int), POINTER(c_int), POINTER(c_int)
        )

        def getter(material_index, volume_out, dims_out, offsets_out, voxel_size_out,
                   mask_out, mu_out, material_count_out, energy_count_out, data_type_out):
            index = material_index - 1
            volume_out[0] = cast(volumes[index], c_void_p)
            mask_out[0] = cast(masks[index], POINTER(c_ubyte))
            mu_out[0] = cast(mus, POINTER(c_float))
            for axis in range(3):
                dims_out[axis] = material_dims[index][axis]
                offsets_out[axis] = float(index + axis)
                voxel_size_out[axis] = 1.0
            material_count_out[0] = 2
            energy_count_out[0] = 2
            data_type_out[0] = 0
            return 0

        cfg = SimpleNamespace(
            phantom=SimpleNamespace(numberOfMaterials=2),
            clib=SimpleNamespace(get_phantom_info_vox_for_cuda=getter_type(getter)),
        )
        host = _get_cpu_phantom_data(cfg)

        self.assertNotIn("volume_data", host)
        self.assertNotIn("xy_mask", host)
        self.assertEqual(host["volume_pointers"][0], addressof(volumes[0]))
        self.assertEqual(host["volume_pointers"][1], addressof(volumes[1]))
        self.assertTrue(np.array_equal(host["dims"], np.array(material_dims, dtype=np.int32)))


if __name__ == "__main__":
    unittest.main()
