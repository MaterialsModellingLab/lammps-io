#!/usr/bin/env python3

import pathlib
import tempfile

import numpy as np
import pytest


def yield_tempfile(num_atoms: int):
    LAMMPS_DATA = """LAMMPS data file via minimal example
# Some line comment
{num_atoms} atoms # some comment
1 atom types

-10.0 10.0 xlo xhi
-20.0 20.0 ylo yhi
-30.0 30.0 zlo zhi

Masses

    1 1.0

Atoms # atomic

    {atoms}

Velocities

    {velocities}

"""

    atoms = np.random.rand(num_atoms, 3) * 10
    velocities = np.random.rand(num_atoms, 3)

    atoms_str = "\n".join(
        f"    {i} 1 {atoms[i - 1, 0]} {atoms[i - 1, 1]} {atoms[i - 1, 2]} 0 0 0"
        for i in range(1, num_atoms + 1)
    )
    velocities_str = "\n".join(
        f"    {i} {velocities[i - 1, 0]} {velocities[i - 1, 1]} {velocities[i - 1, 2]}"
        for i in range(1, num_atoms + 1)
    )

    LAMMPS_DATA = LAMMPS_DATA.format(
        num_atoms=num_atoms, atoms=atoms_str, velocities=velocities_str
    )

    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        f.write(LAMMPS_DATA)
        return f.name, atoms, velocities


X_LIST = np.pow(2, np.arange(10, 20)).tolist()


@pytest.mark.parametrize("x", X_LIST)
def test_ase_read(benchmark, x):
    fname, poses, _velocities = yield_tempfile(x)
    from ase.io.lammpsdata import read_lammps_data as read

    def fn():
        atoms = read(fname)
        return np.array(atoms.get_positions()), np.array(atoms.get_velocities())

    r_poses, _r_velocities = fn()  # Run assertion before benchmarking
    assert np.allclose(r_poses, poses)
    # assert np.allclose(r_velocities, velocities)

    benchmark.pedantic(
        fn,
        rounds=10,
        iterations=1,
    )


@pytest.mark.parametrize("x", X_LIST)
def test_lammps_io_read(benchmark, x):
    import lammps_io as lio

    fname, poses, velocities = yield_tempfile(x)

    def fn():
        lmp_data = lio.data.read(filename=fname)
        return lio.data.get_atom_poses(lmp_data), lio.data.get_atom_velocities(lmp_data)

    r_poses, r_velocities = fn()  # Run assertion before benchmarking
    assert np.allclose(r_poses, poses)
    assert np.allclose(r_velocities, velocities)

    benchmark.pedantic(
        fn,
        rounds=10,
        iterations=1,
    )


@pytest.mark.parametrize("x", X_LIST)
def test_ase_write(benchmark, x):
    fname, _, _ = yield_tempfile(x)
    from ase.io.lammpsdata import read_lammps_data as read
    from ase.io.lammpsdata import write_lammps_data as write

    atoms = read(fname)

    # Set random number to positions, and velocities
    atom_ids = np.arange(1, x + 1)
    pos_update_dict = {atom_id: np.random.rand(3) for atom_id in atom_ids}
    vel_update_dict = {atom_id: np.random.rand(3) for atom_id in atom_ids}

    def fn():
        atom_poses = np.zeros((x, 3))
        for atom_id, p in pos_update_dict.items():
            atom_poses[atom_id - 1] = p
        atoms.set_positions(atom_poses)

        velocities = np.zeros((x, 3))
        for atom_id, v in vel_update_dict.items():
            velocities[atom_id - 1] = v

        atoms.set_velocities(velocities)

        write(fname, atoms, velocities=True)

    fn()  # Run assertion before benchmarking
    atoms = read(fname)
    positions = atoms.get_positions()
    for atom_id in atom_ids:
        assert np.allclose(positions[atom_id - 1], pos_update_dict[atom_id])

    benchmark.pedantic(
        fn,
        rounds=10,
        iterations=1,
    )


@pytest.mark.parametrize("x", X_LIST)
def test_lammps_io_write(benchmark, x):
    import lammps_io as lio

    fname, _, _ = yield_tempfile(x)

    lmp_data = lio.data.read(filename=fname)

    atom_ids = np.arange(1, x + 1)
    pos_update_dict = {atom_id: np.random.rand(3) for atom_id in atom_ids}
    vel_update_dict = {atom_id: np.random.rand(3) for atom_id in atom_ids}

    def fn():
        lio.data.update_atom_poses(lmp_data, pos_update_dict)
        lio.data.update_atom_velocities(lmp_data, vel_update_dict)
        lio.data.write(fname, lmp_data)

    fn()  # Run assertion before benchmarking
    lmp_data = lio.data.read(filename=fname)
    poses_dict = lio.data.get_atom_poses_dict(lmp_data)
    velocities_dict = lio.data.get_atom_velocities_dict(lmp_data)
    for atom_id in atom_ids:
        assert np.allclose(poses_dict[atom_id], pos_update_dict[atom_id])
        assert np.allclose(velocities_dict[atom_id], vel_update_dict[atom_id])

    benchmark.pedantic(
        fn,
        rounds=10,
        iterations=1,
    )


if __name__ == "__main__":
    filename = pathlib.Path(__file__).stem
    pytest.main(
        [
            "-v",
            __file__,
            "--benchmark-only",
            "--benchmark-warmup=True",
            "--benchmark-warmup-iterations=10",
            f"--benchmark-save={filename}",
        ]
    )
