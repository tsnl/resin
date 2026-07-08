use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use resin_core::Tree;
use resin_dsl::ElementType;

use crate::tensor::ConcreteTensor;

/// Cache key from concrete input leaf shapes and dtypes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CacheKey {
    leaves: Vec<LeafSig>,
}

#[derive(Clone, Debug)]
struct LeafSig {
    shape: Box<[usize]>,
    element_type: ElementType,
}

impl PartialEq for LeafSig {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && element_type_tag(self.element_type) == element_type_tag(other.element_type)
    }
}

impl Eq for LeafSig {}

impl Hash for LeafSig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        element_type_tag(self.element_type).hash(state);
    }
}

fn element_type_tag(element_type: ElementType) -> u8 {
    match element_type {
        ElementType::F32 => 0,
    }
}

impl CacheKey {
    pub(crate) fn from_params<P, A>(params: &P) -> Self
    where
        P: Tree<A>,
        A: ConcreteTensor,
    {
        let mut leaves = Vec::new();
        params.for_each_leaf(|leaf| {
            leaves.push(LeafSig {
                shape: leaf.shape().into(),
                element_type: leaf.element_type(),
            });
        });
        Self { leaves }
    }
}

pub(crate) struct CompileCache<A> {
    inner: Mutex<HashMap<CacheKey, A>>,
}

impl<A: Clone> CompileCache<A> {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn get_or_insert_with(
        &self,
        key: CacheKey,
        compile: impl FnOnce() -> Result<A, crate::error::CompileError>,
    ) -> Result<A, crate::error::CompileError> {
        let mut cache = self.inner.lock().expect("compile cache poisoned");
        if let Some(artifact) = cache.get(&key) {
            return Ok(artifact.clone());
        }
        let artifact = compile()?;
        cache.insert(key, artifact.clone());
        Ok(artifact)
    }
}

#[cfg(test)]
mod tests {
    use resin_dsl::ElementType;
    use resin_macros::Tree;

    use super::*;
    use crate::tensor::ConcreteTensor;

    #[derive(Clone)]
    struct TestTensor {
        shape: Box<[usize]>,
    }

    impl ConcreteTensor for TestTensor {
        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn element_type(&self) -> ElementType {
            ElementType::F32
        }

        fn zeros(shape: &[usize], _element_type: ElementType) -> Self {
            Self {
                shape: shape.into(),
            }
        }

        fn from_f32(shape: &[usize], values: &[f32]) -> Self {
            assert_eq!(shape.iter().product::<usize>(), values.len());
            Self {
                shape: shape.into(),
            }
        }

        fn to_f32(&self) -> Vec<f32> {
            let n = self.shape.iter().product::<usize>().max(1);
            vec![0.0; if self.shape.is_empty() { 1 } else { n }]
        }
    }

    #[derive(Tree)]
    struct Params<A> {
        x: A,
        y: A,
    }

    #[test]
    fn cache_key_matches_leaf_shapes() {
        let a = Params {
            x: TestTensor {
                shape: Box::from([2, 3]),
            },
            y: TestTensor {
                shape: Box::from([4]),
            },
        };
        let b = Params {
            x: TestTensor {
                shape: Box::from([2, 3]),
            },
            y: TestTensor {
                shape: Box::from([4]),
            },
        };
        let c = Params {
            x: TestTensor {
                shape: Box::from([2, 4]),
            },
            y: TestTensor {
                shape: Box::from([4]),
            },
        };

        assert_eq!(CacheKey::from_params(&a), CacheKey::from_params(&b));
        assert_ne!(CacheKey::from_params(&a), CacheKey::from_params(&c));
    }

    #[test]
    fn compile_cache_hits_on_repeated_key() {
        let cache = CompileCache::new();
        let key = CacheKey::from_params(&Params {
            x: TestTensor {
                shape: Box::from([1]),
            },
            y: TestTensor {
                shape: Box::from([1]),
            },
        });

        let mut compiles = 0usize;
        let first = cache
            .get_or_insert_with(key.clone(), || {
                compiles += 1;
                Ok(7usize)
            })
            .unwrap();
        let second = cache
            .get_or_insert_with(key, || {
                compiles += 1;
                Ok(9usize)
            })
            .unwrap();

        assert_eq!(first, 7);
        assert_eq!(second, 7);
        assert_eq!(compiles, 1);
    }
}