//! `Symbol` is an interned string type optimized for fast comparisons and hashing.

use fxhash::FxHasher;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt::{Debug, Display, Formatter},
    hash::{Hash, Hasher},
    sync::Arc,
};

#[derive(Default, Clone)]
pub struct Symbol {
    text: Arc<String>,
    hash: u64,
}

impl From<Arc<String>> for Symbol {
    fn from(text: Arc<String>) -> Self {
        let hash = {
            let mut hasher = FxHasher::default();
            text.hash(&mut hasher);
            hasher.finish()
        };
        Self { text, hash }
    }
}
impl From<String> for Symbol {
    fn from(text: String) -> Self {
        Self::from(Arc::new(text))
    }
}
impl From<&str> for Symbol {
    fn from(text: &str) -> Self {
        Self::from(text.to_string())
    }
}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        // Compare by hash for performance. Hash collisions are extremely unlikely.
        // In debug builds, we can add an extra check to ensure correctness.
        if self.hash != other.hash {
            return false;
        }
        debug_assert_eq!(self.text, other.text);
        true
    }
}
impl Eq for Symbol {}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> Ordering {
        self.hash.cmp(&other.hash)
    }
}

impl Hash for Symbol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

type FormatResult = Result<(), std::fmt::Error>;
impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatResult {
        f.write_str(&self.text)
    }
}
impl Debug for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatResult {
        f.write_fmt(format_args!("Symbol({:?})", &self.text))
    }
}

impl Borrow<String> for Symbol {
    fn borrow(&self) -> &String {
        &self.text
    }
}
impl Borrow<str> for Symbol {
    fn borrow(&self) -> &str {
        &self.text
    }
}

impl Symbol {
    pub fn hash(&self) -> u64 {
        self.hash
    }
    pub fn text(&self) -> &String {
        &self.text
    }
    pub fn is_upper_id(&self) -> bool {
        for c in self.text.chars() {
            if c.is_alphabetic() {
                return c.is_uppercase();
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hashbrown::HashMap;
    use rand::prelude::*;
    use std::ops::RangeInclusive;

    #[test]
    fn fuzz_intstr_hash_equality() {
        const RNG_SEED: u64 = 0xF00DCAFE_u64;
        const FUZZ_ITER_COUNT: usize = 1 << 14;
        const IDENTIFIER_LENGTH_RANGE: RangeInclusive<usize> = 1..=128;
        const IDENTIFIER_CHARS: &[u8] =
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";

        fn generate_random_string(rng: &mut StdRng) -> String {
            let len = rng.random_range(IDENTIFIER_LENGTH_RANGE);
            let mut s = Vec::with_capacity(len);
            for _ in 0..len {
                s.push(IDENTIFIER_CHARS[rng.random_range(0..IDENTIFIER_CHARS.len())]);
            }
            String::from_utf8(s).unwrap()
        }

        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let mut hash_map: HashMap<Symbol, String> = HashMap::with_capacity(FUZZ_ITER_COUNT);
        for _ in 0..FUZZ_ITER_COUNT {
            let s: String = generate_random_string(&mut rng);
            let i: Symbol = s.as_str().into();
            if let Some(other_s) = hash_map.get(&i) {
                if &s != other_s {
                    panic!(
                        "Hash collision detected: {:?} and {:?} both map to hash 0x{:x}",
                        s, other_s, i.hash
                    );
                }
            }
            hash_map.insert(i, s);
        }
    }
}
