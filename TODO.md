function -> network
-   [x] Rename Tensor to Node, rename `tensor.py` to `graph.py`.
-   [ ] Simplify autodiff to work on graphs directly.
    -   [x] Port bulk of old autodiff solution.
    -   [ ] Can we use a single ViewNode without IndexNode?
    -   [ ] Implement df_do for ReductionNode
    -   [ ] Implement df_do for IndexNode
    -   [ ] Implement df_do for ViewNode
-   [ ] Support "graph fusion": connect output node to input (var/parameter) node
-   [ ] Implement compiler: leave var/parameter nodes as buffers that can be written 
    before graph execution.
