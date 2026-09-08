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
