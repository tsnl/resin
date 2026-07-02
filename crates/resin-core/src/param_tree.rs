//! Host module / param trees (Python `PyTree` analogue).

use std::fmt;
use std::sync::Arc;

/// One segment of a structured param path (`layers` / `0` / `weight`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParamTreePathElement {
    Name(Arc<str>),
    Index(usize),
}

impl ParamTreePathElement {
    pub fn name(s: impl AsRef<str>) -> Self {
        Self::Name(Arc::from(s.as_ref()))
    }
}

impl fmt::Display for ParamTreePathElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(s) => f.write_str(s),
            Self::Index(i) => write!(f, "{i}"),
        }
    }
}

/// Structured path to a leaf (e.g. `layers` / `0` / `weight`).
pub type ParamTreePath = Box<[ParamTreePathElement]>;

/// Format `path` as a dotted buffer name (`layers.0.weight`).
pub fn format_param_path(path: &[ParamTreePathElement]) -> String {
    let mut s = String::new();
    for (i, el) in path.iter().enumerate() {
        if i > 0 {
            s.push('.');
        }
        use std::fmt::Write;
        let _ = write!(s, "{el}");
    }
    s
}

/// `prefix` (host register name) + structured path → dotted buffer name.
pub fn join_param_path(prefix: &str, path: &[ParamTreePathElement]) -> String {
    let rest = format_param_path(path);
    if prefix.is_empty() {
        rest
    } else if rest.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}.{rest}")
    }
}

fn prepend(el: ParamTreePathElement, path: &[ParamTreePathElement]) -> ParamTreePath {
    let mut out = Vec::with_capacity(1 + path.len());
    out.push(el);
    out.extend_from_slice(path);
    out.into_boxed_slice()
}

/// Structured host tree of leaves (e.g. `View` params).
///
/// Shape lives in the concrete type (`Linear`, `Mlp`, `Vec<…>`); there is no live
/// Object/Array/Leaf enum.
pub trait ParamTree: Sized {
    type Leaf;
    type Map<U>: ParamTree<Leaf = U>;

    fn map<U>(self, f: impl FnMut(Self::Leaf) -> U) -> Self::Map<U>;

    /// Structured paths to each leaf (not pre-joined strings).
    fn flatten(&self) -> impl Iterator<Item = (ParamTreePath, &Self::Leaf)> + '_;

    /// Same shape as `self`, other leaf type — for `sgd` (not derivable from `map` alone).
    fn zip_with<U, O>(self, other: Self::Map<U>, f: impl FnMut(Self::Leaf, U) -> O)
        -> Self::Map<O>;
}

impl<T: ParamTree> ParamTree for Vec<T> {
    type Leaf = T::Leaf;
    type Map<U> = Vec<T::Map<U>>;

    fn map<U>(self, mut f: impl FnMut(Self::Leaf) -> U) -> Self::Map<U> {
        self.into_iter().map(|x| x.map(&mut f)).collect()
    }

    fn flatten(&self) -> impl Iterator<Item = (ParamTreePath, &Self::Leaf)> + '_ {
        self.iter().enumerate().flat_map(|(i, x)| {
            x.flatten()
                .map(move |(path, leaf)| (prepend(ParamTreePathElement::Index(i), &path), leaf))
        })
    }

    fn zip_with<U, O>(
        self,
        other: Self::Map<U>,
        mut f: impl FnMut(Self::Leaf, U) -> O,
    ) -> Self::Map<O> {
        assert_eq!(
            self.len(),
            other.len(),
            "ParamTree::zip_with length mismatch"
        );
        self.into_iter()
            .zip(other)
            .map(|(a, b)| a.zip_with(b, &mut f))
            .collect()
    }
}

impl<T: ParamTree> ParamTree for Option<T> {
    type Leaf = T::Leaf;
    type Map<U> = Option<T::Map<U>>;

    fn map<U>(self, mut f: impl FnMut(Self::Leaf) -> U) -> Self::Map<U> {
        Option::map(self, |x| ParamTree::map(x, &mut f))
    }

    fn flatten(&self) -> impl Iterator<Item = (ParamTreePath, &Self::Leaf)> + '_ {
        self.iter().flat_map(ParamTree::flatten)
    }

    fn zip_with<U, O>(
        self,
        other: Self::Map<U>,
        f: impl FnMut(Self::Leaf, U) -> O,
    ) -> Self::Map<O> {
        match (self, other) {
            (Some(a), Some(b)) => Some(a.zip_with(b, f)),
            (None, None) => None,
            _ => panic!("ParamTree::zip_with Option presence mismatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Linear<T> {
        weight: T,
        bias: Option<T>,
    }

    impl<T> ParamTree for Linear<T> {
        type Leaf = T;
        type Map<U> = Linear<U>;

        fn map<U>(self, mut f: impl FnMut(T) -> U) -> Linear<U> {
            Linear {
                weight: f(self.weight),
                bias: self.bias.map(&mut f),
            }
        }

        fn flatten(&self) -> impl Iterator<Item = (ParamTreePath, &T)> + '_ {
            std::iter::once((
                Box::from([ParamTreePathElement::name("weight")]),
                &self.weight,
            ))
            .chain(
                self.bias
                    .iter()
                    .map(|b| (Box::from([ParamTreePathElement::name("bias")]), b)),
            )
        }

        fn zip_with<U, O>(self, other: Linear<U>, mut f: impl FnMut(T, U) -> O) -> Linear<O> {
            Linear {
                weight: f(self.weight, other.weight),
                bias: match (self.bias, other.bias) {
                    (Some(a), Some(b)) => Some(f(a, b)),
                    (None, None) => None,
                    _ => panic!("bias mismatch"),
                },
            }
        }
    }

    #[test]
    fn flatten_linear_and_vec() {
        let mlp = vec![
            Linear {
                weight: 1i32,
                bias: Some(2),
            },
            Linear {
                weight: 3,
                bias: None,
            },
        ];
        let paths: Vec<_> = mlp
            .flatten()
            .map(|(p, v)| (format_param_path(&p), *v))
            .collect();
        assert_eq!(
            paths,
            vec![
                ("0.weight".into(), 1),
                ("0.bias".into(), 2),
                ("1.weight".into(), 3),
            ]
        );
    }

    #[test]
    fn map_and_zip() {
        let layer = Linear {
            weight: 2i32,
            bias: Some(3),
        };
        let doubled = layer.clone().map(|x| x * 2);
        assert_eq!(doubled.weight, 4);
        assert_eq!(doubled.bias, Some(6));

        let zipped = layer.zip_with(doubled, |a, b| a + b);
        assert_eq!(zipped.weight, 6);
        assert_eq!(zipped.bias, Some(9));
    }
}
