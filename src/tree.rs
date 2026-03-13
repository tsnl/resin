#[macro_export]
macro_rules! define_tree {
	(
		$(
			$visibility:vis enum $name:ident {
				$(
					$variant:ident { $($fname:ident : $ftype:ty),* $(,)? }
				),+
				$(,)?
			}
		)+
	) => {
		$(
			::paste::paste! {
				#[derive(Debug)]
				$visibility enum $name {
					$(
						$variant ( Box< [< $name:snake:lower >]::$variant > )
					),+
				}
				pub mod [< $name:snake:lower >] {
					use super::*;
					$(
						#[derive(Debug)]
						$visibility struct $variant {
							$( pub $fname: $ftype ),*
						}
					)+
				}
				impl $name {
					$(
						pub fn [<new_ $variant:snake:lower>]( $( $fname: $ftype ),* ) -> Self {
							Self::$variant(Box::new( [< $name:snake:lower >]::$variant { $( $fname ),* } ))
						}
					)+
				}
			}
		)+
	};
}

#[cfg(test)]
mod tests {
    define_tree! {
      pub enum Tree {
        Leaf { value: i32 },
        Node { lt: Tree, rt: Tree }
      }
    }

    #[test]
    fn test_tree_creation() {
        let node = Tree::new_node(Tree::new_leaf(1), Tree::new_leaf(2));

        match node {
            Tree::Node(inner) => {
                let tree::Node { lt, rt } = *inner;
                match lt {
                    Tree::Leaf(inner) => {
                        let tree::Leaf { value } = *inner;
                        assert_eq!(value, 1);
                    }
                    _ => {
                        panic!("Unexpected tree type");
                    }
                }
                match rt {
                    Tree::Leaf(inner) => {
                        let tree::Leaf { value } = *inner;
                        assert_eq!(value, 2);
                    }
                    _ => {
                        panic!("Unexpected tree type");
                    }
                }
            }
            _ => {
                panic!("Unexpected tree type");
            }
        }
    }
}
