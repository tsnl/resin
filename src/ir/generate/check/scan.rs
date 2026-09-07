use std::collections::BTreeSet;

use crate::ast::{StmtKind, Term, TermKind};

#[derive(Default)]
pub(super) struct Scan {
    pub references: BTreeSet<String>,
    locals: BTreeSet<String>,
}

impl Scan {
    pub fn bind(&mut self, name: &str) {
        self.locals.insert(name.to_owned());
    }

    pub fn term(&mut self, term: &Term) {
        match &term.val {
            TermKind::Var { name } => {
                if !self.locals.contains(name.val.as_ref()) {
                    self.references.insert(name.val.to_string());
                }
            }
            TermKind::Type { .. } => {}
            TermKind::Try { value } => {
                self.term(value);
            }
            TermKind::Match { value, arms } => {
                self.term(value);
                for arm in arms {
                    let before = self.locals.clone();
                    self.bind(&arm.name.val);
                    self.term(&arm.body);
                    self.locals = before;
                }
            }
            TermKind::If { cond, then, els } => {
                self.term(cond);
                self.term(then);
                self.term(els);
            }
            TermKind::While { cond, body } => {
                self.term(cond);
                self.term(body);
            }
            TermKind::Array { elems }
            | TermKind::Builtin { args: elems, .. }
            | TermKind::Hole { children: elems } => {
                for elem in elems {
                    self.term(elem);
                }
            }
            TermKind::Record { fields } => {
                for (_, value) in fields {
                    self.term(value);
                }
            }
            TermKind::Block { stmts, tail } => {
                let before = self.locals.clone();
                for stmt in stmts {
                    match &stmt.val {
                        StmtKind::Define { name, init } => {
                            self.bind(&name.val);
                            self.term(init);
                        }
                        StmtKind::Declare { name, .. } => {
                            self.bind(&name.val);
                        }
                        StmtKind::Expr { term } => self.term(term),
                        StmtKind::Defer { body } => {
                            self.term(body);
                        }
                        _ => {}
                    }
                }
                self.term(tail);
                self.locals = before;
            }
            TermKind::Call { func, arg } => {
                self.term(func);
                self.term(arg);
            }
            TermKind::Assign { place, value } => {
                self.term(place);
                self.term(value);
            }
            TermKind::Deref { pointer } => self.term(pointer),
            TermKind::Address { place } => self.term(place),
            TermKind::Field { base, .. } | TermKind::FieldHole { base } => self.term(base),
            TermKind::Unit | TermKind::Num { .. } | TermKind::String { .. } => {}
        }
    }
}

/// Strongly connected groups in dependency-first order.
pub(super) fn groups(edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct Walk<'a> {
        edges: &'a [Vec<usize>],
        index: Vec<Option<usize>>,
        low: Vec<usize>,
        active: Vec<bool>,
        stack: Vec<usize>,
        next: usize,
        groups: Vec<Vec<usize>>,
    }
    impl Walk<'_> {
        fn visit(&mut self, node: usize) {
            self.index[node] = Some(self.next);
            self.low[node] = self.next;
            self.next += 1;
            self.stack.push(node);
            self.active[node] = true;
            for &dependency in &self.edges[node] {
                if self.index[dependency].is_none() {
                    self.visit(dependency);
                    self.low[node] = self.low[node].min(self.low[dependency]);
                } else if self.active[dependency] {
                    self.low[node] = self.low[node].min(self.index[dependency].unwrap());
                }
            }
            if self.low[node] == self.index[node].unwrap() {
                let mut group = Vec::new();
                loop {
                    let member = self.stack.pop().unwrap();
                    self.active[member] = false;
                    group.push(member);
                    if member == node {
                        break;
                    }
                }
                self.groups.push(group);
            }
        }
    }
    let n = edges.len();
    let mut walk = Walk {
        edges,
        index: vec![None; n],
        low: vec![0; n],
        active: vec![false; n],
        stack: vec![],
        next: 0,
        groups: vec![],
    };
    for node in 0..n {
        if walk.index[node].is_none() {
            walk.visit(node);
        }
    }
    walk.groups
}

#[cfg(test)]
mod tests {
    #[test]
    fn recursive_groups_follow_dependencies() {
        assert_eq!(
            super::groups(&[vec![1], vec![0, 2], vec![]]),
            vec![vec![2], vec![1, 0]]
        );
    }
}
