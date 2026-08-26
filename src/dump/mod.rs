use pyo3::prelude::*;

mod functions;
mod parser;
mod types;

pub fn init_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "dump")?;

    m.add_class::<types::Boundary>()?;
    m.add_class::<types::PositionType>()?;
    m.add_class::<types::LammpsDump>()?;
    m.add_function(wrap_pyfunction!(functions::read, &m)?)?;
    m.add_function(wrap_pyfunction!(functions::write, &m)?)?;
    m.add_function(wrap_pyfunction!(functions::create, &m)?)?;

    m.add_function(wrap_pyfunction!(functions::get_timesteps, &m)?)?;
    m.add_function(wrap_pyfunction!(functions::get_atom_poses, &m)?)?;
    m.add_function(wrap_pyfunction!(functions::get_atom_velocities, &m)?)?;

    Ok(m)
}
