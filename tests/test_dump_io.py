import tempfile
from pathlib import Path

import lammps_io as lio
import numpy as np
import pytest

THIS_SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT_DIR = THIS_SCRIPT_DIR.parent
SAMPLE_DIR = PROJECT_ROOT_DIR / "samples"


def test_read_ok1():
    lmp_dump = lio.dump.read(SAMPLE_DIR / "simple.dump")
    assert type(lmp_dump) is lio.dump.LammpsDump


def test_read_ng1():
    filename = SAMPLE_DIR / "unknown.dump"
    with pytest.raises(Exception) as exc_info:
        lio.dump.read(filename)
    assert filename.name in str(exc_info.value)


def test_write():
    lmp_dump = lio.dump.read(SAMPLE_DIR / "simple.dump")
    with tempfile.NamedTemporaryFile() as f:
        lio.dump.write(f.name, lmp_dump)

        actual = f.read().decode("utf-8")

    expected = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
8
ITEM: BOX BOUNDS pp pp pp
0 2.51984209978975
0 2.51984209978975
0 2.51984209978975
ITEM: ATOMS id type xs ys zs
1 1 0 0 0
2 1 0.5 0 0
3 1 0 0.5 0
4 1 0.5 0.5 0
5 1 0 0 0.5
6 1 0.5 0 0.5
7 1 0 0.5 0.5
8 1 0.5 0.5 0.5
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
8
ITEM: BOX BOUNDS pp pp pp
0 2.51984209978975
0 2.51984209978975
0 2.51984209978975
ITEM: ATOMS id type xs ys zs
1 1 0.00202588 0.00288716 -0.00251757
2 1 0.499508 0.00141205 -0.00158363
3 1 0.000358226 0.499189 -0.00176088
4 1 0.499642 0.498581 0.00103171
5 1 0.0005768 -0.000100382 0.502382
6 1 0.497401 -0.0040412 0.500171
7 1 0.0002759 0.50094 0.498134
8 1 0.500213 0.501132 0.504143
ITEM: TIMESTEP
2
ITEM: NUMBER OF ATOMS
8
ITEM: BOX BOUNDS pp pp pp
0 2.51984209978975
0 2.51984209978975
0 2.51984209978975
ITEM: ATOMS id type xs ys zs
1 1 0.00405213 0.0057745 -0.00503567
2 1 0.499015 0.00282413 -0.00316786
3 1 0.00071678 0.498378 -0.0035222
4 1 0.499284 0.497163 0.00206275
5 1 0.0011541 -0.000200932 0.504764
6 1 0.494801 -0.00808236 0.500343
7 1 0.000552111 0.50188 0.496269
8 1 0.500425 0.502264 0.508287
ITEM: TIMESTEP
3
ITEM: NUMBER OF ATOMS
8
ITEM: BOX BOUNDS pp pp pp
0 2.51984209978975
0 2.51984209978975
0 2.51984209978975
ITEM: ATOMS id type xs ys zs
1 1 0.0060791 0.00866219 -0.00755473
2 1 0.498521 0.00423622 -0.00475332
3 1 0.00107599 0.497566 -0.00528442
4 1 0.498926 0.495744 0.00309248
5 1 0.0017324 -0.00030182 0.507148
6 1 0.4922 -0.0121234 0.500516
7 1 0.000828944 0.50282 0.494406
8 1 0.500636 0.503397 0.51243
ITEM: TIMESTEP
4
ITEM: NUMBER OF ATOMS
8
ITEM: BOX BOUNDS pp pp pp
0 2.51984209978975
0 2.51984209978975
0 2.51984209978975
ITEM: ATOMS id type xs ys zs
1 1 0.00810715 0.0115503 -0.0100751
2 1 0.498026 0.00564833 -0.00634061
3 1 0.00143618 0.496753 -0.00704797
4 1 0.498567 0.494326 0.00412028
5 1 0.00231213 -0.000403224 0.509532
6 1 0.489599 -0.0161639 0.500692
7 1 0.00110671 0.503761 0.492545
8 1 0.500846 0.50453 0.516574"""
    assert actual == expected


def test_create():
    position_frames = np.array(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.1, 0.1, 0.1], [0.6, 0.1, 0.1]],
        ]
    )
    box_frames = np.array(
        [
            [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        ]
    )
    tid_frames = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    ).astype(np.uint64)

    boundary = (
        (lio.dump.Boundary.Periodic, lio.dump.Boundary.Periodic),
        (lio.dump.Boundary.Periodic, lio.dump.Boundary.Periodic),
        (lio.dump.Boundary.Periodic, lio.dump.Boundary.Periodic),
    )

    lmp_dump = lio.dump.create(
        position_frames=position_frames,
        box_frames=box_frames,
        tid_frames=tid_frames,
        boundary=boundary,
        position_type=(
            lio.dump.PositionType.Plain,
            lio.dump.PositionType.Scaled,
            lio.dump.PositionType.PlainUnwrapped,
        ),
    )

    assert type(lmp_dump) is lio.dump.LammpsDump

    expected = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x ys zu
1 1 0 0 0
2 1 0.5 0 0
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x ys zu
1 1 0.1 0.1 0.1
2 1 0.6 0.1 0.1"""
    actual = str(lmp_dump)
    assert actual == expected


def test_get_timesteps():
    lmp_dump = lio.dump.read(SAMPLE_DIR / "simple.dump")
    timesteps = lio.dump.get_timesteps(lmp_dump)

    expected: np.array = np.array([0, 1, 2, 3, 4], dtype=np.uint64)

    np.testing.assert_array_equal(timesteps, expected)


def test_get_atom_poses():
    lmp_dump = lio.dump.read(SAMPLE_DIR / "simple.dump")
    atom_poses = lio.dump.get_atom_poses(lmp_dump)

    expected: np.array = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            [
                [0.00202588, 0.00288716, -0.00251757],
                [0.499508, 0.00141205, -0.00158363],
                [0.000358226, 0.499189, -0.00176088],
                [0.499642, 0.498581, 0.00103171],
                [0.0005768, -0.000100382, 0.502382],
                [0.497401, -0.0040412, 0.500171],
                [0.0002759, 0.50094, 0.498134],
                [0.500213, 0.501132, 0.504143],
            ],
            [
                [0.00405213, 0.0057745, -0.00503567],
                [0.499015, 0.00282413, -0.00316786],
                [0.00071678, 0.498378, -0.0035222],
                [0.499284, 0.497163, 0.00206275],
                [0.0011541, -0.000200932, 0.504764],
                [0.494801, -0.00808236, 0.500343],
                [0.000552111, 0.50188, 0.496269],
                [0.500425, 0.502264, 0.508287],
            ],
            [
                [0.0060791, 0.00866219, -0.00755473],
                [0.498521, 0.00423622, -0.00475332],
                [0.00107599, 0.497566, -0.00528442],
                [0.498926, 0.495744, 0.00309248],
                [0.0017324, -0.00030182, 0.507148],
                [0.4922, -0.0121234, 0.500516],
                [0.000828944, 0.50282, 0.494406],
                [0.500636, 0.503397, 0.51243],
            ],
            [
                [0.00810715, 0.0115503, -0.0100751],
                [0.498026, 0.00564833, -0.00634061],
                [0.00143618, 0.496753, -0.00704797],
                [0.498567, 0.494326, 0.00412028],
                [0.00231213, -0.000403224, 0.509532],
                [0.489599, -0.0161639, 0.500692],
                [0.00110671, 0.503761, 0.492545],
                [0.500846, 0.50453, 0.516574],
            ],
        ]
    )

    np.testing.assert_array_almost_equal(atom_poses, expected)


def test_get_atom_velocities():
    lmp_dump = lio.dump.read(SAMPLE_DIR / "velocity.dump")
    atom_velocities = lio.dump.get_atom_velocities(lmp_dump)

    expected: np.array = np.array(
        [
            [
                [-0.597454, 1.3679, -0.998949],
                [-0.174625, 0.457695, 0.693865],
                [-1.3954, 0.662129, 0.124764],
                [1.27585, -0.389213, -0.603687],
                [-0.255976, -0.587153, 1.47928],
                [0.453517, 0.1196, 0.421513],
                [-0.78773, 0.564013, 0.12557],
                [1.48182, -2.19497, -1.24236],
            ],
            [
                [-1.02839, 1.43386, -1.08496],
                [-0.0732, 0.346971, 0.937611],
                [-1.45397, -0.289424, 0.15153],
                [1.03301, -0.743719, 0.0494713],
                [0.0961302, -0.847445, 1.20619],
                [0.321904, 0.0776747, -0.394918],
                [-1.40752, 0.645962, 0.136571],
                [1.21858, -1.87471, -0.813856],
            ],
            [
                [-1.15287, 0.893084, -0.54154],
                [-0.0581091, 0.112733, 0.141586],
                [-0.91241, -0.0414507, 0.347519],
                [0.180746, -0.679239, 0.327095],
                [0.815266, -2.26181, 0.279676],
                [-0.612073, 0.0948027, -0.179506],
                [-0.620936, 0.469241, 0.204894],
                [0.763623, -0.307375, -0.78613],
            ],
        ]
    )
    print(atom_velocities)

    np.testing.assert_array_almost_equal(atom_velocities, expected)


if __name__ == "__main__":
    pytest.main([__file__])
