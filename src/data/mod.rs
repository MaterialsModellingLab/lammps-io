use pyo3::prelude::*;

mod functions;
mod parser;
mod types;

pub fn init_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "data")?;

    m.add_class::<types::LammpsData>()?;
    m.add_function(wrap_pyfunction!(functions::read, &m)?)?;
    m.add_function(wrap_pyfunction!(functions::write, &m)?)?;

    m.add_function(wrap_pyfunction!(functions::get_xyz_regions, &m)?)?;
    m.add_function(wrap_pyfunction!(functions::get_atom_poses, &m)?)?;
    m.add_function(wrap_pyfunction!(functions::get_atom_velocities, &m)?)?;

    m.add_function(wrap_pyfunction!(functions::get_atom_poses_dict, &m)?)?;
    m.add_function(wrap_pyfunction!(functions::get_atom_velocities_dict, &m)?)?;

    m.add_function(wrap_pyfunction!(functions::update_atom_poses, &m)?)?;
    m.add_function(wrap_pyfunction!(functions::update_atom_velocities, &m)?)?;

    Ok(m)
}
