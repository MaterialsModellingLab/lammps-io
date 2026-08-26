use pyo3::prelude::*;

#[pyclass]
#[derive(Default)]
pub struct LammpsDataHeader {
    pub title: String,
    pub num_atoms: u64,
    pub atom_types: u64,
    pub region_x: (f64, f64),
    pub region_y: (f64, f64),
    pub region_z: (f64, f64),
}

#[derive(Default)]
pub struct AtomItem {
    pub type_id: u64,
    pub position: (f64, f64, f64),
    pub image_flag: Option<(i64, i64, i64)>,
}

#[pyclass]
pub struct LammpsDataBody {
    pub masses: std::collections::BTreeMap<u64, f64>,
    pub atoms: std::collections::BTreeMap<u64, AtomItem>,
    pub velocities: std::collections::BTreeMap<u64, (f64, f64, f64)>,
    // TODO: Support other fields like bonds, angles, etc.
}

#[pyclass]
pub struct LammpsData {
    pub header: LammpsDataHeader,
    pub body: LammpsDataBody,
}
