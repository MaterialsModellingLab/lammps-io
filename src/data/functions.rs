use ariadne::{Config, IndexType, Label, Report, ReportKind, Source};
use chumsky::prelude::*;
use numpy::prelude::*;
use numpy::{Ix1, Ix2, PyArray, PyReadonlyArray};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::parser;
use super::types;

#[pymethods]
impl types::LammpsData {
    fn __str__(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("{}\n", self.header.title));
        output.push('\n');
        output.push_str(&format!("{} atoms\n", self.header.num_atoms));
        output.push_str(&format!("{} atom types\n", self.header.atom_types));
        output.push('\n');
        output.push_str(&format!(
            "{} {} xlo xhi\n",
            self.header.region_x.0, self.header.region_x.1
        ));
        output.push_str(&format!(
            "{} {} ylo yhi\n",
            self.header.region_y.0, self.header.region_y.1
        ));
        output.push_str(&format!(
            "{} {} zlo zhi\n",
            self.header.region_z.0, self.header.region_z.1
        ));
        output.push('\n');
        output.push_str("Masses\n\n");
        for (id, mass) in &self.body.masses {
            output.push_str(&format!("{} {}\n", id, mass));
        }
        output.push('\n');
        output.push_str("Atoms # atomic\n\n");
        for (id, atom) in &self.body.atoms {
            output.push_str(&format!(
                "{} {} {} {} {}",
                id, atom.type_id, atom.position.0, atom.position.1, atom.position.2
            ));
            if let Some((nx, ny, nz)) = atom.image_flag {
                output.push_str(&format!(" {} {} {}", nx, ny, nz));
            }
            output.push('\n');
        }
        output.push('\n');
        output.push_str("Velocities\n\n");
        for (id, (vx, vy, vz)) in &self.body.velocities {
            output.push_str(&format!("{} {} {} {}\n", id, vx, vy, vz));
        }
        output.push('\n');
        output
    }
}

#[pyfunction]
pub fn read<'py>(py: Python<'py>, filename: &Bound<'py, PyAny>) -> PyResult<types::LammpsData> {
    let os = py.import("os")?;
    let fspath = os.getattr("fspath")?.call1((filename,))?;
    let filename: &str = fspath.extract()?;

    let content = std::fs::read_to_string(filename)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("'{}': {}", filename, e)))?;

    let (data, errs) = parser::data_parser().parse(&content).into_output_errors();
    if errs.is_empty() && data.is_some() {
        Ok(data.unwrap())
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
    lammps_data: &types::LammpsData,
) -> PyResult<()> {
    let os = py.import("os")?;
    let fspath = os.getattr("fspath")?.call1((filename,))?;
    let filename: &str = fspath.extract()?;

    let content = lammps_data.__str__();
    std::fs::write(filename, content)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(())
}

#[pyfunction]
pub fn get_xyz_regions<'py>(
    py: Python<'py>,
    lammps_data: &types::LammpsData,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let regions: [f64; 6] = [
        lammps_data.header.region_x.0,
        lammps_data.header.region_x.1,
        lammps_data.header.region_y.0,
        lammps_data.header.region_y.1,
        lammps_data.header.region_z.0,
        lammps_data.header.region_z.1,
    ];

    PyArray::<f64, Ix1>::from_slice(py, &regions).reshape([3, 2])
}

#[pyfunction]
pub fn get_atom_poses<'py>(
    py: Python<'py>,
    lammps_data: &types::LammpsData,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let atoms = &lammps_data.body.atoms;

    let flat: Vec<f64> = atoms
        .values()
        .flat_map(|atom| vec![atom.position.0, atom.position.1, atom.position.2])
        .collect();

    PyArray::from_slice(py, &flat).reshape([atoms.len(), 3])
}

#[pyfunction]
pub fn get_atom_velocities<'py>(
    py: Python<'py>,
    lammps_data: &types::LammpsData,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let velocities = &lammps_data.body.velocities;
    let flat: Vec<f64> = velocities
        .values()
        .flat_map(|(vx, vy, vz)| vec![*vx, *vy, *vz])
        .collect();
    PyArray::from_slice(py, &flat).reshape([velocities.len(), 3])
}

#[pyfunction]
pub fn get_atom_poses_dict<'py>(
    py: Python<'py>,
    lammps_data: &types::LammpsData,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    for (&id, atom) in &lammps_data.body.atoms {
        let arr = PyArray::from_slice(py, &[atom.position.0, atom.position.1, atom.position.2]);
        dict.set_item(id, arr)?;
    }

    Ok(dict)
}

#[pyfunction]
pub fn get_atom_velocities_dict<'py>(
    py: Python<'py>,
    lammps_data: &types::LammpsData,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    for (&id, (vx, vy, vz)) in &lammps_data.body.velocities {
        let arr = PyArray::from_slice(py, &[*vx, *vy, *vz]);
        dict.set_item(id, arr)?;
    }

    Ok(dict)
}

#[pyfunction]
pub fn update_atom_poses<'py>(
    lammps_data: &mut types::LammpsData,
    poses_dict: &Bound<'py, PyDict>,
) -> PyResult<()> {
    for (key, value) in poses_dict.iter() {
        let id: u64 = key.extract()?;
        let arr: PyReadonlyArray<f64, Ix1> = value.extract()?;
        if arr.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "position array must have length 3",
            ));
        }
        let slice = arr.as_slice()?;
        if let Some(atom) = lammps_data.body.atoms.get_mut(&id) {
            atom.position = (slice[0], slice[1], slice[2]);
        } else {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "atom id {} not found",
                id
            )));
        }
    }
    Ok(())
}

#[pyfunction]
pub fn update_atom_velocities<'py>(
    lammps_data: &mut types::LammpsData,
    velocities_dict: &Bound<'py, PyDict>,
) -> PyResult<()> {
    for (key, value) in velocities_dict.iter() {
        let id: u64 = key.extract()?;
        // id should exist in lammps_data.body.atoms
        if !lammps_data.body.atoms.contains_key(&id) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "atom id {} not found",
                id
            )));
        }
        let arr: PyReadonlyArray<f64, Ix1> = value.extract()?;
        if arr.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "velocity array must have length 3",
            ));
        }
        let slice = arr.as_slice()?;
        lammps_data
            .body
            .velocities
            .insert(id, (slice[0], slice[1], slice[2]));
    }
    Ok(())
}
