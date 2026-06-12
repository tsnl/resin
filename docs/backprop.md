# `backprop.md`

This document tersely explains reverse mode symbolic autodifferentiation and how it is 
implemented in Resin.

Given a scalar-valued loss function $L$, we want to find 
$\frac{\partial L}{\partial \theta}$ for every parameter $\theta$ that contributes to 
$L$.

By evaluating how much $L$ changes with respect to changes $\theta$, we can decide 
whether to nudge $\theta$ up or down (and by how much) to influence the value of $L$.

**How do we compute $\frac{\partial L}{\partial \theta}$ efficiently for all**
**parameters?**

Assume $L$ is a term, i.e. a composition of function applications, constants, and 
nudgeable parameters recursively.

```
Term: Const c
    | Var   x
    | Apply a = f(o1, o2, ...)

We use `n` to refer to Term, legacy reasons (node in graph).
```

Consider $L$ is a function over some parameters $n_1$, $n_2$, etc.

Straightforward to evaluate $\forall i . \frac{\partial L}{\partial n_i}$ using regular
differentiation rules.

Now, let's say that $n_0 = f(o_j, \dots)$ WLOG.

We can easily compute $\forall j . \frac{\partial n_0}{\partial o_j}$ as above, but how 
do we use this to get $\frac{\partial L}{\partial n_0}$?

The key is in the **multivariate chain rule**.

$$
\frac{\partial L}{\partial o_j}
= \sum_i \frac{\partial L}{\partial n_i} \cdot \frac{\partial n_i}{\partial o_j}
$$

This formula lets us...
-   Evaluate $\frac{\partial L}{\partial n_i}$
-   Evaluate $\frac{\partial n_i}{\partial o_j}$
-   Multiply these together to find $\frac{\partial L}{\partial o_j}$

Note that we can use DP to walk the computational graph backwards, finding gradient terms as we go:
-   Base case: $\frac{\partial L}{\partial L} = 1$
-   Inductive: want to find some $\frac{\partial L}{\partial o}$ for $n(o)$.
    -   Assume we have $\frac{\partial L}{\partial n}$, the output of some term $n$.
    -   Evaluate $\frac{\partial n}{\partial o}$ symbolically, depends only on form of $n$.
    -   Product $\frac{\partial L}{\partial n} \cdot \frac{\partial n}{\partial o}$

Implementation hinges on a `dn_do()` method that evaluates 
$\frac{\partial n}{\partial o}$ for each operand $o$ that is passed immediately to 
function $n$.

```python
def grad(loss: Node) -> dict[Node, Node]:
    """
    Given a scalar valued `loss` Node in a computational graph, computes and returns the
    derivative of `loss` wrt every other node in the computational graph.

    Computed efficiently using backward-mode symbolic differentiation.
    """

    gradient_map = defaultdict(Node.zero)

    gradient_map[loss] = Node.ones_like(loss)

    for node in toposort(loss):
        # Load the gradient of `node`.
        dl_dn = gradient_map[node]

        # Compute the derivatives of `node` wrt each of its operands.
        # @abstractmethod df_do is critical for each node type.
        dn_do_tuple: tuple[Node, ...] = node.dn_do()

        # Accumulate the gradient contribution for each operand.
        for operand, dn_do in zip(node.operands, dn_do_tuple):
            gradient_map[operand] += dl_dn * dn_do
```

In practice, we inline `dn_do()` into `df_do()`, a method that performs the 
multiplication directly for us to avoid inserting additional nodes. E.g. if
`dn_do()` returns `1/x`, then we multiply, we have an extra computational node
as opposed to directly dividing by `x`.
