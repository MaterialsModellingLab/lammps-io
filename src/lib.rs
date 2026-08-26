use pyo3::prelude::*;

mod data;
mod dump;

#[pymodule]
fn lammps_io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_submodule(&data::init_module(py)?)?;
    m.add_submodule(&dump::init_module(py)?)?;

    Ok(())
}
