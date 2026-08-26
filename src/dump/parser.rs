use core::panic;

use super::types::*;
use chumsky::prelude::*;

type ParserErr<'src> = extra::Err<Rich<'src, char>>;

fn number_parser<'src>() -> impl Parser<'src, &'src str, &'src str, ParserErr<'src>> {
    let digits = text::digits(10).to_slice();

    let frac = just('.').then(digits);

    let exp = just('e')
        .or(just('E'))
        .then(one_of("+-").or_not())
        .then(digits);

    just('-')
        .or_not()
        .then(text::int(10))
        .then(frac.or_not())
        .then(exp.or_not())
        .to_slice()
}

fn timestep_parser<'src>() -> impl Parser<'src, &'src str, i64, ParserErr<'src>> {
    just("ITEM:")
        .padded()
        .ignore_then(just("TIMESTEP"))
        .padded()
        .ignore_then(
            one_of("+-")
                .or_not()
                .then(text::int(10))
                .to_slice()
                .from_str::<i64>()
                .unwrapped(),
        )
}

fn number_of_atoms_parser<'src>() -> impl Parser<'src, &'src str, u64, ParserErr<'src>> {
    just("ITEM:")
        .padded()
        .ignore_then(just("NUMBER").padded())
        .ignore_then(just("OF").padded())
        .ignore_then(just("ATOMS").padded())
        .ignore_then(text::int(10).from_str::<u64>().unwrapped())
}

fn box_bounds_parser<'src>() -> impl Parser<'src, &'src str, BoxBounds, ParserErr<'src>> {
    let boundary_parser = just("p")
        .map(|_| Boundary::Periodic)
        .or(just("f").map(|_| Boundary::Fixed))
        .or(just("s").map(|_| Boundary::Shrinkwrap))
        .or(just("m").map(|_| Boundary::ShrinkwrapWithMinimumValue));

    let boundary_pair_parser = boundary_parser
        .then(boundary_parser)
        .map(|(b1, b2)| (b1, b2));

    let f64_pair_parser = number_parser()
        .separated_by(text::whitespace())
        .collect_exactly::<[&'src str; 2]>()
        .try_map(|[s1, s2], _span| {
            let v1 = s1.parse::<f64>().unwrap();
            let v2 = s2.parse::<f64>().unwrap();
            Ok((v1, v2))
        });

    let boundaries_parser = boundary_pair_parser
        .separated_by(text::whitespace())
        .collect_exactly::<[(Boundary, Boundary); 3]>();

    let values_parser = f64_pair_parser
        .separated_by(text::whitespace())
        .collect_exactly::<[(f64, f64); 3]>();

    just("ITEM:")
        .padded()
        .ignore_then(just("BOX").padded())
        .ignore_then(just("BOUNDS").padded())
        .ignore_then(boundaries_parser)
        .padded()
        .then(values_parser)
        .map(|([bx, by, bz], [vx, vy, vz])| BoxBounds {
            bound_x: bx,
            bound_y: by,
            bound_z: bz,
            x: vx,
            y: vy,
            z: vz,
        })
}

fn atoms_parser<'src>() -> impl Parser<'src, &'src str, AtomCollection, ParserErr<'src>> {
    let header_parser = text::ident()
        .padded()
        .repeated()
        .collect::<Vec<&'src str>>();

    let row_parser = just(' ')
        .repeated()
        .or_not()
        .ignore_then(number_parser())
        .separated_by(just(' ').repeated().at_least(1))
        .at_least(1)
        .collect::<Vec<&'src str>>()
        .then_ignore(just(' ').repeated().or_not());

    let rows_parser = row_parser
        .separated_by(text::newline())
        .at_least(1)
        .collect::<Vec<Vec<&'src str>>>();

    just("ITEM:")
        .padded()
        .ignore_then(just("ATOMS").padded())
        .ignore_then(header_parser)
        .padded()
        .then(rows_parser)
        .try_map(|(headers, rows), _span| {
            let mut atom_map: AtomMap = std::collections::BTreeMap::new();

            // Helper to find header index with error handling
            let get_idx =
                |header: &str| -> Option<usize> { headers.iter().position(|&h| h == header) };
            let get_pos_idx = |coords: &str| -> (usize, PositionType) {
                let idx = headers.iter().position(|&h| h.starts_with(coords)).unwrap();

                let header_str = headers[idx];
                if header_str == coords {
                    (idx, PositionType::Plain)
                } else if header_str == format!("{}s", coords) {
                    (idx, PositionType::Scaled)
                } else if header_str == format!("{}u", coords) {
                    (idx, PositionType::PlainUnwrapped)
                } else if header_str == format!("{}su", coords) {
                    (idx, PositionType::ScaledUnwrapped)
                } else {
                    // TODO: Return error instead of panicking
                    panic!("Unexpected position header: {}", header_str);
                }
            };

            let id_idx = get_idx("id")
                .unwrap_or_else(|| panic!("Missing required 'id' header in ATOMS section"));
            let type_idx = get_idx("type")
                .unwrap_or_else(|| panic!("Missing required 'type' header in ATOMS section"));
            // x or xs or xu is the idx and also should return position_type here
            let (x_idx, x_pos_type) = get_pos_idx("x");
            let (y_idx, y_pos_type) = get_pos_idx("y");
            let (z_idx, z_pos_type) = get_pos_idx("z");

            for row in rows {
                if row.len() != headers.len() {
                    // Skip malformed rows instead of failing the entire parse
                    continue;
                }

                let id = row[id_idx].parse::<u64>().unwrap();
                let type_id = row[type_idx].parse::<u64>().unwrap();
                let x = row[x_idx].parse::<f64>().unwrap();
                let y = row[y_idx].parse::<f64>().unwrap();
                let z = row[z_idx].parse::<f64>().unwrap();

                let velocity = if let (Some(vx_idx), Some(vy_idx), Some(vz_idx)) =
                    (get_idx("vx"), get_idx("vy"), get_idx("vz"))
                {
                    let vx = row[vx_idx].parse::<f64>().unwrap();
                    let vy = row[vy_idx].parse::<f64>().unwrap();
                    let vz = row[vz_idx].parse::<f64>().unwrap();
                    Some((vx, vy, vz))
                } else {
                    None
                };

                let atom = Atom {
                    type_id,
                    position: (x, y, z),
                    velocity: velocity,
                };
                atom_map.insert(id, atom);
            }
            Ok(AtomCollection {
                position_type: (x_pos_type, y_pos_type, z_pos_type),
                atom_map,
            })
        })
}

fn frame_parser<'src>() -> impl Parser<'src, &'src str, Frame, ParserErr<'src>> {
    timestep_parser()
        .padded()
        .then(number_of_atoms_parser().padded())
        .then(box_bounds_parser().padded())
        .then(atoms_parser().padded())
        .map(
            |(((timestep, num_atoms), box_bounds), atom_collection)| Frame {
                timestep,
                num_atoms,
                box_bounds,
                atom_collection,
            },
        )
}

pub fn dump_parser<'src>() -> impl Parser<'src, &'src str, LammpsDump, ParserErr<'src>> {
    frame_parser()
        .padded()
        .repeated()
        .collect()
        .map(|frames: Vec<Frame>| LammpsDump { frames })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_number_parser() {
        let input = "-1.23456e+02";
        let value = number_parser()
            .parse(input)
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert_eq!(value, -123.456);
    }

    #[test]
    fn test_timestep_parser() {
        let input = "ITEM: TIMESTEP 10";
        let timestep = timestep_parser().parse(input).unwrap();
        assert_eq!(timestep, 10);
    }

    #[test]
    fn test_number_of_atoms_parser() {
        let input = "ITEM: NUMBER OF ATOMS
        100";
        let num_atoms = number_of_atoms_parser().parse(input).unwrap();
        assert_eq!(num_atoms, 100);
    }

    #[test]
    fn test_boundary_parser() {
        let input = "ITEM: BOX BOUNDS pf sp mf
        0.0000000000000000e+00 2.5198420997897499e+00
        0.0000000000000000e+00 2.5198420997897499e+00
        0.0000000000000000e+00 2.5198420997897499e+00";
        let actual = box_bounds_parser().parse(input).unwrap();
        let expected = BoxBounds {
            bound_x: (Boundary::Periodic, Boundary::Fixed),
            bound_y: (Boundary::Shrinkwrap, Boundary::Periodic),
            bound_z: (Boundary::ShrinkwrapWithMinimumValue, Boundary::Fixed),
            x: (0.0, 2.519_842_099_789_75),
            y: (0.0, 2.519_842_099_789_75),
            z: (0.0, 2.519_842_099_789_75),
        };
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_atoms_parser() {
        let input = "ITEM: ATOMS id type xs ys zs
1 1 0.0 0.0 0.0
2 1 0.5 0.0 0.0
3 1 0.0 0.5 0.0
4 1 0.5 0.5 0.0
5 1 0.0 0.0 0.5
6 1 0.5 0.0 0.5
7 1 0.0 0.5 0.5
8 1 0.5 0.5 0.5";
        let actual = atoms_parser().parse(input).unwrap();
        assert_eq!(actual.atom_map.len(), 8);
        assert_eq!(
            actual.position_type,
            (
                PositionType::Scaled,
                PositionType::Scaled,
                PositionType::Scaled
            )
        );
        assert_eq!(
            actual.atom_map.get(&1),
            Some(&Atom {
                type_id: 1,
                position: (0.0, 0.0, 0.0),
                velocity: None,
            })
        );
        assert_eq!(
            actual.atom_map.get(&8),
            Some(&Atom {
                type_id: 1,
                position: (0.5, 0.5, 0.5),
                velocity: None,
            })
        );
    }

    #[test]
    fn test_atoms_parser_with_velocities() {
        let input = "ITEM: ATOMS id type xs ys zs vx vy vz
1 1 0.0 0.0 0.0 0.1 0.2 0.3
2 1 0.5 0.0 0.0 0.4 0.5 0.6";
        let actual = atoms_parser().parse(input).unwrap();
        assert_eq!(actual.atom_map.len(), 2);
        assert_eq!(
            actual.position_type,
            (
                PositionType::Scaled,
                PositionType::Scaled,
                PositionType::Scaled
            )
        );
        assert_eq!(
            actual.atom_map.get(&1),
            Some(&Atom {
                type_id: 1,
                position: (0.0, 0.0, 0.0),
                velocity: Some((0.1, 0.2, 0.3)),
            })
        );
        assert_eq!(
            actual.atom_map.get(&2),
            Some(&Atom {
                type_id: 1,
                position: (0.5, 0.0, 0.0),
                velocity: Some((0.4, 0.5, 0.6)),
            })
        );
    }

    #[test]
    fn test_frame_parser() {
        let input = "ITEM: TIMESTEP
    0
    ITEM: NUMBER OF ATOMS
    8
    ITEM: BOX BOUNDS pp pp pp
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
    ITEM: ATOMS id type xs ys zs
    1 1 0 0 0
    2 1 0.5 0 0
    3 1 0 0.5 0
    4 1 0.5 0.5 0
    5 1 0 0 0.5
    6 1 0.5 0 0.5
    7 1 0 0.5 0.5
    8 1 0.5 0.5 0.5
    ";
        let frame = frame_parser().parse(input).unwrap();
        assert_eq!(frame.timestep, 0);
        assert_eq!(frame.num_atoms, 8);
        assert_eq!(
            frame.box_bounds.bound_x,
            (Boundary::Periodic, Boundary::Periodic)
        );
        assert_eq!(
            frame.box_bounds.bound_y,
            (Boundary::Periodic, Boundary::Periodic)
        );
        assert_eq!(
            frame.box_bounds.bound_z,
            (Boundary::Periodic, Boundary::Periodic)
        );
        assert_eq!(frame.box_bounds.x, (0.0, 2.519_842_099_789_75));
        assert_eq!(frame.box_bounds.y, (0.0, 2.519_842_099_789_75));
        assert_eq!(frame.box_bounds.z, (0.0, 2.519_842_099_789_75));
        assert_eq!(frame.atom_collection.atom_map.len(), 8);
    }

    #[test]
    fn test_dump_parser() {
        let input = "ITEM: TIMESTEP
    0
    ITEM: NUMBER OF ATOMS
    8
    ITEM: BOX BOUNDS pp pp pp
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
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
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
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
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
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
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
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
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
    0.0000000000000000e+00 2.5198420997897499e+00
    ITEM: ATOMS id type xs ys zs
    1 1 0.00810715 0.0115503 -0.0100751
    2 1 0.498026 0.00564833 -0.00634061
    3 1 0.00143618 0.496753 -0.00704797
    4 1 0.498567 0.494326 0.00412028
    5 1 0.00231213 -0.000403224 0.509532
    6 1 0.489599 -0.0161639 0.500692
    7 1 0.00110671 0.503761 0.492545
    8 1 0.500846 0.50453 0.516574
    ";
        let dump = dump_parser().parse(input).unwrap();
        assert_eq!(dump.frames.len(), 5);
    }

    #[test]
    fn test_atoms_parser_missing_required_header() {
        // Test with missing 'id' header
        let input = "ITEM: ATOMS type xs ys zs
1 0.0 0.0 0.0
2 0.5 0.0 0.0";
        // assert that the parser panics with the expected message
        let result = std::panic::catch_unwind(|| atoms_parser().parse(input).into_result());
        assert!(
            result.is_err(),
            "Parser should panic when required header is missing"
        );
    }

    #[test]
    fn test_atoms_parser_invalid_number() {
        // Test with invalid number format
        let input = "ITEM: ATOMS id type xs ys zs
1 1 invalid 0.0 0.0";
        let result = atoms_parser().parse(input).into_result();
        assert!(
            result.is_err(),
            "Parser should fail when atom data contains invalid number"
        );
    }

    #[test]
    fn test_box_bounds_parser_invalid_number() {
        // Test with invalid number in box bounds
        let input = "ITEM: BOX BOUNDS pp pp pp
invalid 2.5198420997897499e+00
0.0 2.5198420997897499e+00
0.0 2.5198420997897499e+00";
        let result = box_bounds_parser().parse(input).into_result();
        assert!(
            result.is_err(),
            "Parser should fail when box bounds contain invalid number"
        );
    }
}
