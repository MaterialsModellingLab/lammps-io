use super::types::{AtomItem, LammpsData, LammpsDataBody, LammpsDataHeader};
use chumsky::prelude::*;

type ParserErr<'src> = extra::Err<Rich<'src, char>>;

trait Region {
    const LABEL: &'static str;
}

struct X;
struct Y;
struct Z;

impl Region for X {
    const LABEL: &'static str = "x";
}
impl Region for Y {
    const LABEL: &'static str = "y";
}
impl Region for Z {
    const LABEL: &'static str = "z";
}

fn comment_parser<'src>() -> impl Parser<'src, &'src str, &'src str, ParserErr<'src>> {
    just('#')
        .then(none_of('\n').repeated())
        .to_slice()
        .labelled("comment")
}

fn padded_with_comment<'src, P, O>(p: P) -> impl Parser<'src, &'src str, O, ParserErr<'src>>
where
    P: Parser<'src, &'src str, O, ParserErr<'src>>,
{
    p.padded_by(comment_parser().padded().repeated())
}

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
        .labelled("number")
}

fn num_atoms_parser<'src>() -> impl Parser<'src, &'src str, u64, ParserErr<'src>> {
    text::int(10)
        .from_str::<u64>()
        .unwrapped()
        .padded()
        .then_ignore(just("atoms"))
        .labelled("number of atoms")
}

fn num_atom_types_parser<'src>() -> impl Parser<'src, &'src str, u64, ParserErr<'src>> {
    text::int(10)
        .from_str::<u64>()
        .unwrapped()
        .padded()
        .then_ignore(just("atom"))
        .padded()
        .then_ignore(just("types"))
        .labelled("number of atom types")
}

fn region_parser<'src, R: Region>() -> impl Parser<'src, &'src str, [f64; 2], ParserErr<'src>> {
    let f64_pair_parser = number_parser()
        .separated_by(text::whitespace())
        .collect_exactly::<[&'src str; 2]>()
        .try_map(|[s1, s2], _span| {
            let low = s1.parse::<f64>().unwrap();
            let high = s2.parse::<f64>().unwrap();
            Ok([low, high])
        });
    f64_pair_parser
        .padded()
        .then_ignore(just(format!("{}lo", R::LABEL)))
        .padded()
        .then_ignore(just(format!("{}hi", R::LABEL)))
        .labelled(format!("region {}", R::LABEL))
}

fn header_parser<'src>() -> impl Parser<'src, &'src str, LammpsDataHeader, ParserErr<'src>> {
    // Implementation of the full header parser would go here
    let title_parser = any().then(none_of('\n').repeated()).to_slice();

    title_parser
        .map(|s: &'src str| LammpsDataHeader {
            title: s.trim().to_string(),
            ..Default::default()
        })
        .then(padded_with_comment(num_atoms_parser()))
        .map(|(mut header, num_atoms)| {
            header.num_atoms = num_atoms;
            header
        })
        .then(padded_with_comment(num_atom_types_parser()))
        .map(|(mut header, num_atom_types)| {
            header.atom_types = num_atom_types;
            header
        })
        .then(padded_with_comment(region_parser::<X>()))
        .map(|(mut header, region_x)| {
            header.region_x = (region_x[0], region_x[1]);
            header
        })
        .then(padded_with_comment(region_parser::<Y>()))
        .map(|(mut header, region_y)| {
            header.region_y = (region_y[0], region_y[1]);
            header
        })
        .then(padded_with_comment(region_parser::<Z>()))
        .map(|(mut header, region_z)| {
            header.region_z = (region_z[0], region_z[1]);
            header.region_z = (region_z[0], region_z[1]);
            header
        })
        .labelled("LammpsDataHeader")
}

fn masses_parser<'src>(
) -> impl Parser<'src, &'src str, std::collections::BTreeMap<u64, f64>, ParserErr<'src>> {
    let mass_pair_parser = text::int(10)
        .from_str::<u64>()
        .unwrapped()
        .padded()
        .then(number_parser().from_str::<f64>().unwrapped().padded())
        .map(|(type_id, mass)| (type_id, mass));

    padded_with_comment(just("Masses"))
        .ignore_then(
            padded_with_comment(mass_pair_parser)
                .repeated()
                .collect::<std::collections::BTreeMap<u64, f64>>(),
        )
        .labelled("Masses")
}

fn atom_parser<'src>() -> impl Parser<'src, &'src str, (u64, AtomItem), ParserErr<'src>> {
    let u64_parser = text::int(10).from_str::<u64>().unwrapped().boxed();
    let i64_parser = one_of("+-")
        .or_not()
        .then(text::int(10))
        .to_slice()
        .from_str::<i64>()
        .unwrapped()
        .boxed();
    let i64_triple_parser = i64_parser
        .clone()
        .separated_by(text::whitespace())
        .collect_exactly::<[i64; 3]>();
    let f64_parser = number_parser().from_str::<f64>().unwrapped().boxed();
    let f64_triple_parser = f64_parser
        .clone()
        .separated_by(text::whitespace())
        .collect_exactly::<[f64; 3]>();
    u64_parser
        .clone()
        .padded()
        .then(u64_parser.clone().padded())
        .map(|(id, type_id)| {
            let item = AtomItem {
                type_id,
                ..Default::default()
            };
            (id, item)
        })
        .then(f64_triple_parser.padded())
        .map(|((id, mut item), position)| {
            item.position = (position[0], position[1], position[2]);
            (id, item)
        })
        .then(i64_triple_parser.padded().or_not())
        .map(|((id, mut item), image_flag_opt)| {
            item.image_flag = image_flag_opt.map(|[nx, ny, nz]| (nx, ny, nz));
            (id, item)
        })
        .labelled("Atom")
}

fn atoms_parser<'src>(
) -> impl Parser<'src, &'src str, std::collections::BTreeMap<u64, AtomItem>, ParserErr<'src>> {
    padded_with_comment(just("Atoms"))
        .ignore_then(
            padded_with_comment(atom_parser())
                .repeated()
                .collect::<std::collections::BTreeMap<u64, AtomItem>>(),
        )
        .labelled("Atoms")
}

fn velocity_parser<'src>() -> impl Parser<'src, &'src str, (u64, (f64, f64, f64)), ParserErr<'src>>
{
    let u64_parser = text::int(10).from_str::<u64>().unwrapped().boxed();
    let f64_parser = number_parser().from_str::<f64>().unwrapped().boxed();
    let f64_triple_parser = f64_parser
        .clone()
        .separated_by(text::whitespace())
        .collect_exactly::<[f64; 3]>();
    u64_parser
        .padded()
        .then(f64_triple_parser.padded())
        .map(|(id, vel)| (id, (vel[0], vel[1], vel[2])))
        .labelled("Velocity")
}

fn velocities_parser<'src>(
) -> impl Parser<'src, &'src str, std::collections::BTreeMap<u64, (f64, f64, f64)>, ParserErr<'src>>
{
    padded_with_comment(just("Velocities"))
        .ignore_then(
            padded_with_comment(velocity_parser())
                .repeated()
                .collect::<std::collections::BTreeMap<u64, (f64, f64, f64)>>(),
        )
        .labelled("Velocities")
}

fn body_parser<'src>() -> impl Parser<'src, &'src str, LammpsDataBody, ParserErr<'src>> {
    padded_with_comment(masses_parser())
        .then(padded_with_comment(atoms_parser()))
        .then(padded_with_comment(velocities_parser()))
        .map(|((masses, atoms), velocities)| LammpsDataBody {
            masses,
            atoms,
            velocities,
        })
        .labelled("LammpsDataBody")
}

pub fn data_parser<'src>() -> impl Parser<'src, &'src str, LammpsData, ParserErr<'src>> {
    header_parser()
        .padded()
        .then(body_parser())
        .map(|(header, body)| LammpsData { header, body })
        .labelled("LammpsData")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comment_parser() {
        let parser = comment_parser().padded();
        let res1 = parser.parse("# This is a comment").unwrap();
        assert_eq!(res1, "# This is a comment");

        let res2 = parser.parse("# Another comment\n\n").unwrap();
        assert_eq!(res2, "# Another comment");
    }

    #[test]
    fn test_num_atoms_parser() {
        let parser = num_atoms_parser();
        let res1 = parser.parse("42 atoms").unwrap();
        assert_eq!(res1, 42);
    }

    #[test]
    fn test_num_atom_types_parser() {
        let parser = num_atom_types_parser();
        let res1 = parser.parse("3 atom types").unwrap();
        assert_eq!(res1, 3);
    }

    #[test]
    fn test_region_parser_x() {
        let parser = region_parser::<X>();
        let res1 = parser.parse("-10.0 10.0 xlo xhi").unwrap();
        assert_eq!(res1, [-10.0, 10.0]);
    }

    #[test]
    fn test_region_parser_y() {
        let parser = region_parser::<Y>();
        let res1 = parser.parse("-20.0 20.0 ylo yhi").unwrap();
        assert_eq!(res1, [-20.0, 20.0]);
    }

    #[test]
    fn test_region_parser_z() {
        let parser = region_parser::<Z>();
        let res1 = parser.parse("-30.0 30.0 zlo zhi").unwrap();
        assert_eq!(res1, [-30.0, 30.0]);
    }

    #[test]
    fn test_header_parser() {
        let input = "LAMMPS data file via minimal example
2 atoms # some comment
1 atom types
-10.0 10.0 xlo xhi
-20.0 20.0 ylo yhi
-30.0 30.0 zlo zhi";
        let parser = header_parser();
        let res = parser.parse(input).unwrap();
        assert_eq!(res.title, "LAMMPS data file via minimal example");
        assert_eq!(res.num_atoms, 2);
        assert_eq!(res.atom_types, 1);
        assert_eq!(res.region_x, (-10.0, 10.0));
        assert_eq!(res.region_y, (-20.0, 20.0));
        assert_eq!(res.region_z, (-30.0, 30.0));
    }

    #[test]
    fn test_masses_parser() {
        let input = "Masses
1 1.0  # Some comment
2 2.0
3 3.5";
        let parser = masses_parser();
        let res = parser.parse(input).unwrap();
        assert_eq!(res.len(), 3);
        assert_eq!(res.get(&1), Some(&1.0));
        assert_eq!(res.get(&2), Some(&2.0));
        assert_eq!(res.get(&3), Some(&3.5));
    }

    #[test]
    fn test_atoms_parser() {
        let input = "Atoms # atomic
1 1 1.0 1.0 1.0 0 0 0
2 2 2.0 2.0 2.0 1 1 1
3 1 3.0 3.0 3.0";
        let parser = atoms_parser();
        let res = parser.parse(input).unwrap();
        assert_eq!(res.len(), 3);
        let atom1 = res.get(&1).unwrap();
        assert_eq!(atom1.type_id, 1);
        assert_eq!(atom1.position, (1.0, 1.0, 1.0));
        assert_eq!(atom1.image_flag, Some((0, 0, 0)));
        let atom2 = res.get(&2).unwrap();
        assert_eq!(atom2.type_id, 2);
        assert_eq!(atom2.position, (2.0, 2.0, 2.0));
        assert_eq!(atom2.image_flag, Some((1, 1, 1)));
        let atom3 = res.get(&3).unwrap();
        assert_eq!(atom3.type_id, 1);
        assert_eq!(atom3.position, (3.0, 3.0, 3.0));
        assert_eq!(atom3.image_flag, None);
    }

    #[test]
    fn test_velocities_parser() {
        let input = "Velocities
1 0.1 0.0 0.0
2 -0.1 0.5 0.0
3 0.0 0.0 1.0";
        let parser = velocities_parser();
        let res = parser.parse(input).unwrap();
        assert_eq!(res.len(), 3);
        assert_eq!(res.get(&1), Some(&(0.1, 0.0, 0.0)));
        assert_eq!(res.get(&2), Some(&(-0.1, 0.5, 0.0)));
        assert_eq!(res.get(&3), Some(&(0.0, 0.0, 1.0)));
    }

    #[test]
    fn test_body_parser() {
        let input = "Masses
1 1.0
2 2.0

Atoms # atomic
1 1 1.0 1.0 1.0 0 0 0
2 2 2.0 2.0 2.0 1 1 1

Velocities
1 0.1 0.0 0.0
2 -0.1 0.5 0.0";
        let parser = body_parser();
        let res = parser.parse(input).unwrap();
        assert_eq!(res.masses.len(), 2);
        assert_eq!(res.atoms.len(), 2);
        assert_eq!(res.velocities.len(), 2);
        assert_eq!(res.masses.get(&1), Some(&1.0));
        assert_eq!(res.masses.get(&2), Some(&2.0));
        let atom1 = res.atoms.get(&1).unwrap();
        assert_eq!(atom1.type_id, 1);
        assert_eq!(atom1.position, (1.0, 1.0, 1.0));
        assert_eq!(atom1.image_flag, Some((0, 0, 0)));
        let atom2 = res.atoms.get(&2).unwrap();
        assert_eq!(atom2.type_id, 2);
        assert_eq!(atom2.position, (2.0, 2.0, 2.0));
        assert_eq!(atom2.image_flag, Some((1, 1, 1)));
        assert_eq!(res.velocities.get(&1), Some(&(0.1, 0.0, 0.0)));
        assert_eq!(res.velocities.get(&2), Some(&(-0.1, 0.5, 0.0)));
    }

    #[test]
    fn test_data_parser() {
        let input = "LAMMPS data file via minimal example
        2 atoms
        1 atom types

        -10.0 10.0 xlo xhi
        -20.0 20.0 ylo yhi
        -30.0 30.0 zlo zhi

        Masses
        1 1.0
        2 2.0
        Atoms # atomic
        1 1 1.0 1.0 1.0 0 0 0
        2 2 2.0 2.0 2.0 1 1 1

        Velocities

        1 0.1 0.0 0.0
        2 -0.1 0.5 0.0";
        let parser = data_parser();
        let res = parser.parse(input).unwrap();
        assert_eq!(res.header.title, "LAMMPS data file via minimal example");
        assert_eq!(res.header.num_atoms, 2);
        assert_eq!(res.header.atom_types, 1);
        assert_eq!(res.header.region_x, (-10.0, 10.0));
        assert_eq!(res.header.region_y, (-20.0, 20.0));
        assert_eq!(res.header.region_z, (-30.0, 30.0));
        assert_eq!(res.body.masses.len(), 2);
        assert_eq!(res.body.atoms.len(), 2);
        assert_eq!(res.body.velocities.len(), 2);
    }
}
