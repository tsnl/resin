use super::Ty;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Foreign {
    pub header: Arc<str>,
    pub params: Vec<Ty>,
}

impl Foreign {
    pub fn valid(&self, result: &Ty) -> bool {
        self.params.iter().all(Ty::foreign_value)
            && (*result == Ty::Unit || result.foreign_value())
            && !self.header.is_empty()
            && self
                .header
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_./- :~()".contains(&c))
    }
}
