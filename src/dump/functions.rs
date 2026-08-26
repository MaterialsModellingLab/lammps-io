use ariadne::{Config, IndexType, Label, Report, ReportKind, Source};
use chumsky::prelude::*;
use numpy::prelude::*;
use numpy::{Ix1, Ix2, Ix3, PyArray, PyReadonlyArray};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use super::parser;
use super::types;

#[pymethods]
impl types::LammpsDump {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
}

#[pyfunction]
pub fn read<'py>(py: Python<'py>, filename: &Bound<'py, PyAny>) -> PyResult<types::LammpsDump> {
    let os = py.import("os")?;
    let fpath = os.getattr("fspath")?.call1((filename,))?;
    let filename: &str = fpath.extract()?;

    let content = std::fs::read_to_string(filename)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("'{}': {}", filename, e)))?;

    let (dump, errs) = parser::dump_parser().parse(&content).into_output_errors();
    if errs.is_empty() && dump.is_some() {
        Ok(dump.unwrap())
    } else {
        let err_msgs = errs
            .into_iter()
            .map(|e| {
                let report = Report::build(ReportKind::Error, ((), e.span().into_range()))
                    .with_config(Config::new().with_index_type(IndexType::Byte))
                    .with_message(e.to_string())
                    .with_label(
                        Label::new(((), e.span().into_range()))
                            .with_message(e.reason().to_string())
                            .with_color(ariadne::Color::Red),
                    )
                    .finish();
                let mut buf = Vec::new();
                report.write(Source::from(&content), &mut buf).unwrap();
                String::from_utf8(buf).unwrap()
            })
            .collect::<Vec<_>>()
            .join("\n");
        Err(pyo3::exceptions::PyValueError::new_err(err_msgs))
    }
}

#[pyfunction]
pub fn write<'py>(
    py: Python<'py>,
    filename: &Bound<'py, PyAny>,
    lammps_dump: &types::LammpsDump,
) -> PyResult<()> {
    let os = py.import("os")?;
    let fpath = os.getattr("fspath")?.call1((filename,))?;
    let filename: &str = fpath.extract()?;

    let content = lammps_dump.__str__();
    std::fs::write(filename, content)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(())
}

#[derive(FromPyObject)]
pub struct BoundaryPair(types::Boundary, types::Boundary);

#[derive(FromPyObject)]
pub struct BoundaryPairTriple(BoundaryPair, BoundaryPair, BoundaryPair);

#[derive(FromPyObject)]
pub struct PositionTypeTriple(
    types::PositionType,
    types::PositionType,
    types::PositionType,
);

/// Creates a new LAMMPS dump from trajectory data.
///
/// # Arguments
///
/// * `position_frames` - Atom positions in **scaled coordinates** (xs, ys, zs) with shape
///   `(n_frames, n_atoms, 3)`. Scaled coordinates are fractional positions relative to the
///   simulation box, typically ranging from 0.0 to 1.0. For example, a particle at (0.5, 0.5, 0.5)
///   is at the center of the box regardless of the actual box dimensions.
///
/// * `box_frames` - Simulation box boundaries with shape `(n_frames, 3, 2)`. Each frame contains
///   three pairs of values representing [xlo, xhi], [ylo, yhi], and [zlo, zhi] box bounds.
///
/// * `tid_frames` - Atom type IDs with shape `(n_frames, n_atoms)`. Each value represents the
///   type/species of the corresponding atom.
///
/// * `boundary` - Boundary conditions for the simulation box as a triple of pairs
///   `((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))`. Each pair specifies the boundary type
///   (Periodic, Fixed, Shrinkwrap, or ShrinkwrapWithMinimumValue) for the lower and upper
///   bounds of each dimension.
///
/// # Returns
///
/// A `LammpsDump` object containing the trajectory frames with atoms positioned using scaled
/// coordinates (xs, ys, zs format).
///
/// # Errors
///
/// Returns an error if:
/// - `position_frames` does not have shape `(n_frames, n_atoms, 3)`
/// - `box_frames` does not have shape `(n_frames, 3, 2)`
/// - The frame counts in `position_frames`, `box_frames`, and `tid_frames` do not match
/// - The atom counts in `position_frames` and `tid_frames` do not match
#[pyfunction]
pub fn create(
    position_frames: PyReadonlyArray<f64, Ix3>,
    box_frames: PyReadonlyArray<f64, Ix3>,
    tid_frames: PyReadonlyArray<u64, Ix2>,
    boundary: BoundaryPairTriple,
    position_type: PositionTypeTriple,
) -> PyResult<types::LammpsDump> {
    let bound_x = boundary.0;
    let bound_y = boundary.1;
    let bound_z = boundary.2;
    if position_frames.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "position_frames must have shape (n_frames, n_atoms, 3)",
        ));
    }

    if box_frames.shape()[1] != 3 || box_frames.shape()[2] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "box_frames must have shape (n_frames, 3, 2)",
        ));
    }

    if position_frames.shape()[0] != box_frames.shape()[0]
        || position_frames.shape()[0] != tid_frames.shape()[0]
        || position_frames.shape()[1] != tid_frames.shape()[1]
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "position_frames, box_frames and tid_frames must have compatible shapes",
        ));
    }

    let box_frames = box_frames.as_array();
    let position_frames = position_frames.as_array();
    let tid_frames = tid_frames.as_array();
    let n_frames = position_frames.shape()[0];
    let n_atoms = position_frames.shape()[1];

    let mut dump = types::LammpsDump { frames: vec![] };
    for i in 0..n_frames {
        let mut atom_map: types::AtomMap = std::collections::BTreeMap::new();
        for j in 0..n_atoms {
            let atom = types::Atom {
                type_id: tid_frames[[i, j]],
                position: (
                    position_frames[[i, j, 0]],
                    position_frames[[i, j, 1]],
                    position_frames[[i, j, 2]],
                ),
                velocity: None,
            };
            atom_map.insert((j + 1) as u64, atom);
        }

        let frame = types::Frame {
            timestep: i as i64,
            num_atoms: n_atoms as u64,
            box_bounds: types::BoxBounds {
                bound_x: (bound_x.0, bound_x.1),
                bound_y: (bound_y.0, bound_y.1),
                bound_z: (bound_z.0, bound_z.1),
                x: (box_frames[[i, 0, 0]], box_frames[[i, 0, 1]]),
                y: (box_frames[[i, 1, 0]], box_frames[[i, 1, 1]]),
                z: (box_frames[[i, 2, 0]], box_frames[[i, 2, 1]]),
            },
            atom_collection: types::AtomCollection {
                position_type: (position_type.0, position_type.1, position_type.2),
                atom_map,
            },
        };
        dump.frames.push(frame);
    }
    Ok(dump)
}

#[pyfunction]
pub fn get_timesteps<'py>(
    py: Python<'py>,
    lammps_dump: &types::LammpsDump,
) -> PyResult<Bound<'py, PyArray<i64, Ix1>>> {
    let frames = &lammps_dump.frames;
    let n_frames = frames.len();
    if n_frames == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "LammpsDump contains no frames",
        ));
    }
    let mut flat: Vec<i64> = Vec::with_capacity(n_frames);
    for frame in frames {
        flat.push(frame.timestep);
    }
    Ok(PyArray::from_slice(py, &flat))
}

#[pyfunction]
pub fn get_atom_poses<'py>(
    py: Python<'py>,
    lammps_dump: &types::LammpsDump,
) -> PyResult<Bound<'py, PyArray<f64, Ix3>>> {
    // Sort by id and return positions for all frames as a 3D array (n_frames, n_atoms, 3)
    let frames = &lammps_dump.frames;
    let n_frames = frames.len();
    let n_atoms = frames[0].num_atoms as usize;
    if n_frames == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "LammpsDump contains no frames",
        ));
    }
    if n_atoms == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Frames in LammpsDump contain no atoms",
        ));
    }

    let mut flat: Vec<f64> = Vec::with_capacity(n_frames * n_atoms * 3);
    for frame in frames {
        for atom in frame.atom_collection.atom_map.values() {
            flat.push(atom.position.0);
            flat.push(atom.position.1);
            flat.push(atom.position.2);
        }
    }
    PyArray::from_slice(py, &flat).reshape([n_frames, n_atoms, 3])
}

#[pyfunction]
pub fn get_atom_velocities<'py>(
    py: Python<'py>,
    lammps_dump: &types::LammpsDump,
) -> PyResult<Bound<'py, PyArray<f64, Ix3>>> {
    // Sort by id and return velocities for all frames as a 3D array (n_frames, n_atoms, 3)
    let frames = &lammps_dump.frames;
    let n_frames = frames.len();
    let n_atoms = frames[0].num_atoms as usize;
    if n_frames == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "LammpsDump contains no frames",
        ));
    }
    if n_atoms == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Frames in LammpsDump contain no atoms",
        ));
    }

    let mut flat: Vec<f64> = Vec::with_capacity(n_frames * n_atoms * 3);
    for frame in frames {
        for atom in frame.atom_collection.atom_map.values() {
            if let Some(velocity) = atom.velocity {
                flat.push(velocity.0);
                flat.push(velocity.1);
                flat.push(velocity.2);
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "LammpsDump do not have velocity information",
                ));
            }
        }
    }
    PyArray::from_slice(py, &flat).reshape([n_frames, n_atoms, 3])
}
