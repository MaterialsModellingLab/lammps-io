import tempfile
from pathlib import Path

import lammps_io as lio
import numpy as np
import pytest

THIS_SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT_DIR = THIS_SCRIPT_DIR.parent
SAMPLE_DIR = PROJECT_ROOT_DIR / "samples"


def test_read_ok1():
    lmp_data = lio.data.read(SAMPLE_DIR / "simple.data")
    assert type(lmp_data) is lio.data.LammpsData
    print(lmp_data)


def test_read_ng1():
    filename = "UnknownFile.data"
    with pytest.raises(Exception) as exc_info:
        lio.data.read(filename)
    assert filename in str(exc_info.value)


def test_write():
    lmp_data = lio.data.read(SAMPLE_DIR / "simple.data")
    with tempfile.NamedTemporaryFile() as f:
        lio.data.write(f.name, lmp_data)

        actual = f.read().decode("utf-8")
    expected = """LAMMPS data file via minimal example

2 atoms
1 atom types

-10 10 xlo xhi
-20 20 ylo yhi
-30 30 zlo zhi

Masses

1 1
2 2

Atoms # atomic

1 1 1 2 3 1 0 -1
2 2 4 5 6 0 0 0

Velocities

1 0.1 0 0
2 -0.1 0 0

"""
    assert actual == expected


def test_get_xyz_regions():
    lmp_data = lio.data.read(SAMPLE_DIR / "simple.data")
    x_region, y_region, z_region = lio.data.get_xyz_regions(lmp_data)
    assert type(x_region) is np.ndarray
    assert x_region.dtype == np.float64
    assert np.allclose(x_region, np.array([-10.0, 10.0]))

    assert type(y_region) is np.ndarray
    assert y_region.dtype == np.float64
    assert np.allclose(y_region, np.array([-20.0, 20.0]))

    assert type(z_region) is np.ndarray
    assert z_region.dtype == np.float64
    assert np.allclose(z_region, np.array([-30.0, 30.0]))


def test_get_atom_poses():
    lmp_data = lio.data.read(SAMPLE_DIR / "simple.data")
    atom_poses = lio.data.get_atom_poses(lmp_data)
    assert atom_poses.shape == (2, 3)
    assert type(atom_poses) is np.ndarray
    assert atom_poses.dtype == np.float64
    assert np.allclose(atom_poses[0], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(atom_poses[1], np.array([4.0, 5.0, 6.0]))


def test_get_atom_velocities():
    lmp_data = lio.data.read(SAMPLE_DIR / "simple.data")
    atom_velocities = lio.data.get_atom_velocities(lmp_data)
    assert atom_velocities.shape == (2, 3)
    assert type(atom_velocities) is np.ndarray
    assert atom_velocities.dtype == np.float64
    assert np.allclose(atom_velocities[0], np.array([0.1, 0.0, 0.0]))
    assert np.allclose(atom_velocities[1], np.array([-0.1, 0.0, 0.0]))


def test_get_atom_poses_dict():
    lmp_data = lio.data.read(SAMPLE_DIR / "simple.data")
    poses_dict = lio.data.get_atom_poses_dict(lmp_data)
    assert type(poses_dict) is dict
    assert poses_dict.keys() == {1, 2}
    assert type(poses_dict[1]) is np.ndarray
    assert np.allclose(poses_dict[1], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(poses_dict[2], np.array([4.0, 5.0, 6.0]))


def test_get_atom_velocities_dict():
    lmp_data = lio.data.read(SAMPLE_DIR / "simple.data")
    velocities_dict = lio.data.get_atom_velocities_dict(lmp_data)
    assert type(velocities_dict) is dict
    assert velocities_dict.keys() == {1, 2}
    assert type(velocities_dict[1]) is np.ndarray
    assert np.allclose(velocities_dict[1], np.array([0.1, 0.0, 0.0]))
    assert np.allclose(velocities_dict[2], np.array([-0.1, 0.0, 0.0]))


def test_update_atom_poses():
    lmp_data = lio.data.read(SAMPLE_DIR / "simple.data")
    new_poses = {
        1: np.array([10.0, 10.0, 10.0], dtype=np.float64),
        2: np.array([-10.0, -10.0, -10.0], dtype=np.float64),
    }
    lio.data.update_atom_poses(lmp_data, new_poses)
    poses_dict = lio.data.get_atom_poses_dict(lmp_data)
    assert type(poses_dict) is dict
    assert poses_dict.keys() == {1, 2}
    assert type(poses_dict[1]) is np.ndarray
    assert np.allclose(poses_dict[1], np.array([10.0, 10.0, 10.0]))
    assert np.allclose(poses_dict[2], np.array([-10.0, -10.0, -10.0]))


def test_update_atom_velocities():
    lmp_data = lio.data.read(SAMPLE_DIR / "simple.data")
    new_velocities = {
        1: np.array([1.0, 1.0, 1.0], dtype=np.float64),
        2: np.array([-1.0, -1.0, -1.0], dtype=np.float64),
    }
    lio.data.update_atom_velocities(lmp_data, new_velocities)
    velocities_dict = lio.data.get_atom_velocities_dict(lmp_data)
    assert type(velocities_dict) is dict
    assert velocities_dict.keys() == {1, 2}
    assert type(velocities_dict[1]) is np.ndarray
    assert np.allclose(velocities_dict[1], np.array([1.0, 1.0, 1.0]))
    assert np.allclose(velocities_dict[2], np.array([-1.0, -1.0, -1.0]))


if __name__ == "__main__":
    pytest.main([__file__])
