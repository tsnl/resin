//! Host module / param trees (Python `PyTree` analogue).

/// Structured host tree of leaves (e.g. `View` params).
///
/// Shape lives in the concrete type (`Linear`, `Mlp`, `Vec<…>`); there is no live
/// Object/Array/Leaf enum.
pub trait ParamTree: Sized {
    type Leaf;
    type Map<U>: ParamTree<Leaf = U>;

    fn map<U>(self, f: impl FnMut(Self::Leaf) -> U) -> Self::Map<U>;

    /// Dotted paths (e.g. `layers.0.weight`).
    fn flatten(&self) -> impl Iterator<Item = (String, &Self::Leaf)> + '_;

    /// Same shape as `self`, other leaf type — for `sgd` (not derivable from `map` alone).
    fn zip_with<U, O>(
        self,
        other: Self::Map<U>,
        f: impl FnMut(Self::Leaf, U) -> O,
    ) -> Self::Map<O>;
}

impl<T: ParamTree> ParamTree for Vec<T> {
    type Leaf = T::Leaf;
    type Map<U> = Vec<T::Map<U>>;

    fn map<U>(self, mut f: impl FnMut(Self::Leaf) -> U) -> Self::Map<U> {
        self.into_iter().map(|x| x.map(&mut f)).collect()
    }

    fn flatten(&self) -> impl Iterator<Item = (String, &Self::Leaf)> + '_ {
        self.iter().enumerate().flat_map(|(i, x)| {
            x.flatten().map(move |(path, leaf)| {
                let path = if path.is_empty() {
                    i.to_string()
                } else {
                    format!("{i}.{path}")
                };
                (path, leaf)
            })
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

    fn flatten(&self) -> impl Iterator<Item = (String, &Self::Leaf)> + '_ {
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

        fn flatten(&self) -> impl Iterator<Item = (String, &T)> + '_ {
            std::iter::once(("weight".to_string(), &self.weight)).chain(
                self.bias
                    .iter()
                    .map(|b| ("bias".to_string(), b)),
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
        let paths: Vec<_> = mlp.flatten().map(|(p, v)| (p, *v)).collect();
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
