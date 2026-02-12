import resin.comp as rc


def mlp(
    linear_1: rc.Tensor,
    linear_2: rc.Tensor,
    linear_3: rc.Tensor,
) -> rc.TensorFunction[rc.Tensor]:
    def forward(x: rc.Tensor) -> rc.Tensor:
        x = linear_1 @ x
        x = relu(x)
        x = linear_2 @ x
        x = relu(x)
        x = linear_3 @ x
        return x

    return forward


def relu(x: rc.Tensor) -> rc.Tensor:
    return x.max(rc.Tensor.const(value=0, dtype=x.dtype))


def main():
    pass


if __name__ == "__main__":
    main()
