import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gecatsim.pyfiles.Phantom_Voxelized import Phantom_Voxelized
from gecatsim.pyfiles.C_Projector_Voxelized_CUDA import is_cuda_projector_available
from gecatsim.pyfiles.C_Projector_Voxelized_GPU import is_gpu_projector_available


class TestVoxelizedGPU(unittest.TestCase):

    def test_phantom_voxelized_builds_gpu_cache(self):
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

        gpu_data = cfg.phantom.gpuVoxelData
        self.assertEqual(gpu_data["materials"], ["water", "air"])
        self.assertEqual(gpu_data["mus"].dtype, np.float32)
        self.assertEqual(gpu_data["mus"].shape, (2, 2))
        self.assertEqual(len(gpu_data["volumes"]), 2)
        self.assertEqual(gpu_data["volumes"][0]["data"].dtype, np.float32)
        self.assertTrue(gpu_data["volumes"][0]["data"].flags["C_CONTIGUOUS"])
        self.assertTrue(np.array_equal(gpu_data["volumes"][0]["dims"], np.array([2, 3, 4], dtype=np.int32)))
        self.assertEqual(gpu_data["volumes"][0]["offsets"].dtype, np.float32)
        self.assertEqual(gpu_data["volumes"][0]["voxelsize"].dtype, np.float32)

    def test_gpu_projector_unavailable_without_gpu_cache(self):
        cfg = SimpleNamespace(phantom=SimpleNamespace())
        self.assertFalse(is_gpu_projector_available(cfg))

    def test_native_cuda_projector_unavailable_without_gpu_cache(self):
        cfg = SimpleNamespace(phantom=SimpleNamespace())
        self.assertFalse(is_cuda_projector_available(cfg))


if __name__ == "__main__":
    unittest.main()
