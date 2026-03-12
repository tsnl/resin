# `resin`

```
# Resin is a statically-typed functional array programming language for data-parallel
# GPU programming. It is designed to be a high-level language for writing GPU kernels.
#
# Similar to Elm, but with more robust let-polymorphism support and basic dependent
# types. Explicit support for array types.
#
# Users:
# - Jax or PyTorch users who want static typing, better performance, easier distribution.
# - Rendering researchers who want to write GPGPU but also leverage hardware acceleration.
# - Innovators who want to integrate ML models, e.g. GPU-accelerated audio processing,
#   physics simulation, etc.

Linear o i =
  { w: [o][i]Fp32,
    b: [o]Fp32 }

linear model input =
    model.w @ input + model.b

relu x =
    max 0 x

Mlp h i o =
  { l1: Linear h i,
    l2: Linear h h,
    l3: Linear o h }

mlp model input =
    h1 = relu (linear model.l1 input)
    h2 = relu (linear model.l2 h1)
    linear model.l3 h2

mlp_train_loss model input target =
    pred = mlp model input
    loss = cross_entropy pred target
    loss

mlp_train model inputs targets lr epochs =
    if epochs <= 0 then
        model
    else
        loss = mlp_train_loss model inputs targets
        grad = grad mlp_train_loss model inputs targets   # NOTE: (grad mlp_train_loss)
        model = model - lr * grad
        mlp_train model inputs targets lr (epochs - 1)
```
