use pyo3::prelude::*;
use std::fmt;

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Boundary {
    Periodic,
    Fixed,
    Shrinkwrap,
    ShrinkwrapWithMinimumValue,
}

#[derive(Debug, PartialEq)]
pub struct BoxBounds {
    pub bound_x: (Boundary, Boundary),
    pub bound_y: (Boundary, Boundary),
    pub bound_z: (Boundary, Boundary),
    pub x: (f64, f64),
    pub y: (f64, f64),
    pub z: (f64, f64),
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionType {
    Plain,
    Scaled,
    PlainUnwrapped,
    ScaledUnwrapped,
}

#[derive(Debug, PartialEq)]
pub struct Atom {
    pub type_id: u64,
    pub position: (f64, f64, f64),
    pub velocity: Option<(f64, f64, f64)>,
}

pub type AtomMap = std::collections::BTreeMap<u64, Atom>;

#[derive(Debug, PartialEq)]
pub struct AtomCollection {
    pub position_type: (PositionType, PositionType, PositionType),
    pub atom_map: AtomMap,
}

pub struct Frame {
    pub timestep: i64,
    pub num_atoms: u64,
    pub box_bounds: BoxBounds,
    pub atom_collection: AtomCollection,
}

#[pyclass]
pub struct LammpsDump {
    pub frames: Vec<Frame>,
}

impl fmt::Display for Boundary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Boundary::Periodic => "p",
            Boundary::Fixed => "f",
            Boundary::Shrinkwrap => "s",
            Boundary::ShrinkwrapWithMinimumValue => "m",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for PositionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PositionType::Plain => "",
            PositionType::Scaled => "s",
            PositionType::PlainUnwrapped => "u",
            PositionType::ScaledUnwrapped => "su",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for BoxBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}{} {}{} {}{}",
            self.bound_x.0,
            self.bound_x.1,
            self.bound_y.0,
            self.bound_y.1,
            self.bound_z.0,
            self.bound_z.1,
        )?;
        writeln!(f, "{} {}", self.x.0, self.x.1)?;
        writeln!(f, "{} {}", self.y.0, self.y.1)?;
        write!(f, "{} {}", self.z.0, self.z.1)
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {} {}",
            self.type_id, self.position.0, self.position.1, self.position.2
        )
    }
}

impl fmt::Display for AtomCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "id type x{} y{} z{}",
            self.position_type.0, self.position_type.1, self.position_type.2,
        )?;
        for (i, (id, atom)) in self.atom_map.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{} {}", id, atom)?;
        }
        Ok(())
    }
}

impl fmt::Display for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ITEM: TIMESTEP")?;
        writeln!(f, "{}", self.timestep)?;
        writeln!(f, "ITEM: NUMBER OF ATOMS")?;
        writeln!(f, "{}", self.num_atoms)?;
        writeln!(f, "ITEM: BOX BOUNDS {}", self.box_bounds)?;
        write!(f, "ITEM: ATOMS {}", self.atom_collection)?;
        Ok(())
    }
}

impl fmt::Display for LammpsDump {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, frame) in self.frames.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", frame)?;
        }
        Ok(())
    }
}
