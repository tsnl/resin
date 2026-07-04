use std::sync::Arc;

pub enum List<T> {
    Nil,
    Cons(T, Arc<List<T>>),
}
