use super::heap::StructuredBuffer;
use crate::util::RangeAllocator;
use parking_lot::RwLock;
use std::{
    ops::Range,
    sync::{Arc, Weak},
};
