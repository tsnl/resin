"""Tests for resin.gradient using the interpreter to verify against analytic derivatives."""

import math

import numpy as np
import pytest

import resin.comp as rc


def scalar(x: float) -> rc.Tensor:
    """Create a scalar (shape ()) tensor constant."""
    return rc.Tensor.const(value=x, dtype="fp32")


def eval_grad(
    f: rc.TensorFunction[rc.Tensor],
    *inputs: rc.Tensor,
) -> tuple[float, tuple[float, ...]]:
    """Evaluate f and its gradients at inputs, returning Python floats."""
    g = rc.grad(f)
    value_tensor, grad_tensors = g(*inputs)
    interp = rc.Interpreter()
    value = interp.evaluate(value_tensor).item()
    grads = tuple(interp.evaluate(gt).item() for gt in grad_tensors)
    return value, grads


def numerical_gradient(
    f: rc.TensorFunction[rc.Tensor],
    *inputs: rc.Tensor,
    eps: float = 1e-4,
) -> tuple[float, ...]:
    """Central-difference numerical gradient for scalar-valued f of scalar inputs."""
    grads: list[float] = []
    for i in range(len(inputs)):
        xi = rc.Interpreter().evaluate(inputs[i]).item()

        args_plus = list(inputs)
        args_plus[i] = scalar(xi + eps)
        f_plus = rc.Interpreter().evaluate(f(*args_plus)).item()

        args_minus = list(inputs)
        args_minus[i] = scalar(xi - eps)
        f_minus = rc.Interpreter().evaluate(f(*args_minus)).item()

        grads.append((f_plus - f_minus) / (2 * eps))
    return tuple(grads)


def check_grad(
    f: rc.TensorFunction[rc.Tensor],
    *inputs: rc.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> None:
    """Assert analytical gradients match numerical gradients (fp32 tolerance)."""
    _, analytical = eval_grad(f, *inputs)
    numerical = numerical_gradient(f, *inputs)
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        np.testing.assert_allclose(
            a,
            n,
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient mismatch for input {i}: analytical={a}, numerical={n}",
        )


class TestIdentityAndConstants:
    def test_identity(self) -> None:
        """f(x) = x, df/dx = 1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x

        val, (dx,) = eval_grad(f, scalar(5.0))
        assert val == pytest.approx(5.0)
        assert dx == pytest.approx(1.0)

    def test_negation(self) -> None:
        """f(x) = -x, df/dx = -1."""

        def f(x: rc.Tensor, /) -> rc.Tensor:
            return -x

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(-3.0)
        assert dx == pytest.approx(-1.0)
        check_grad(f, scalar(3.0))

    def test_add_constant(self) -> None:
        """f(x) = x + 5, df/dx = 1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x + rc.Tensor.const(value=5.0)

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(8.0)
        assert dx == pytest.approx(1.0)
        check_grad(f, scalar(3.0))

    def test_sub_constant(self) -> None:
        """f(x) = x - 2, df/dx = 1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x - rc.Tensor.const(value=2.0)

        val, (dx,) = eval_grad(f, scalar(7.0))
        assert val == pytest.approx(5.0)
        assert dx == pytest.approx(1.0)
        check_grad(f, scalar(7.0))

    def test_mul_constant(self) -> None:
        """f(x) = 3x, df/dx = 3."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return rc.Tensor.const(value=3.0) * x

        val, (dx,) = eval_grad(f, scalar(4.0))
        assert val == pytest.approx(12.0)
        assert dx == pytest.approx(3.0)
        check_grad(f, scalar(4.0))

    def test_div_by_constant(self) -> None:
        """f(x) = x / 2, df/dx = 0.5."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x / rc.Tensor.const(value=2.0)

        val, (dx,) = eval_grad(f, scalar(6.0))
        assert val == pytest.approx(3.0)
        assert dx == pytest.approx(0.5)
        check_grad(f, scalar(6.0))


class TestElementaryFunctions:
    def test_reciprocal(self) -> None:
        """f(x) = 1/x, df/dx = -1/x^2."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return rc.Tensor.const(value=1.0) / x

        val, (dx,) = eval_grad(f, scalar(2.0))
        assert val == pytest.approx(0.5)
        assert dx == pytest.approx(-0.25)
        check_grad(f, scalar(2.0))

    def test_square(self) -> None:
        """f(x) = x^2, df/dx = 2x."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x * x

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(9.0)
        assert dx == pytest.approx(6.0)
        check_grad(f, scalar(3.0))

    def test_cube_via_mul(self) -> None:
        """f(x) = x^3 via x*x*x, df/dx = 3x^2."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x * x * x

        val, (dx,) = eval_grad(f, scalar(2.0))
        assert val == pytest.approx(8.0)
        assert dx == pytest.approx(12.0)
        check_grad(f, scalar(2.0))

    def test_pow(self) -> None:
        """f(x) = x^3 via pow, df/dx = 3x^2."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x ** rc.Tensor.const(value=3.0)

        val, (dx,) = eval_grad(f, scalar(2.0))
        assert val == pytest.approx(8.0)
        assert dx == pytest.approx(12.0)
        check_grad(f, scalar(2.0))

    def test_exp(self) -> None:
        """f(x) = exp(x), df/dx = exp(x)."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.exp()

        val, (dx,) = eval_grad(f, scalar(1.0))
        assert val == pytest.approx(math.e)
        assert dx == pytest.approx(math.e)
        check_grad(f, scalar(1.0))

    def test_log(self) -> None:
        """f(x) = log(x), df/dx = 1/x."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.log()

        val, (dx,) = eval_grad(f, scalar(2.0))
        assert val == pytest.approx(math.log(2.0))
        assert dx == pytest.approx(0.5)
        check_grad(f, scalar(2.0))


class TestChainRule:
    def test_exp_of_double(self) -> None:
        """f(x) = exp(2x), df/dx = 2*exp(2x)."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return (rc.Tensor.const(value=2.0) * x).exp()

        x_val = 1.0
        val, (dx,) = eval_grad(f, scalar(x_val))
        assert val == pytest.approx(math.exp(2.0))
        assert dx == pytest.approx(2.0 * math.exp(2.0))
        check_grad(f, scalar(x_val))

    def test_log_of_square(self) -> None:
        """f(x) = log(x^2), df/dx = 2/x."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return (x * x).log()

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(math.log(9.0))
        assert dx == pytest.approx(2.0 / 3.0)
        check_grad(f, scalar(3.0))

    def test_square_of_sum(self) -> None:
        """f(x) = (x + 1)^2, df/dx = 2(x + 1)."""

        def f(x: rc.Tensor) -> rc.Tensor:
            s = x + rc.Tensor.const(value=1.0)
            return s * s

        val, (dx,) = eval_grad(f, scalar(2.0))
        assert val == pytest.approx(9.0)
        assert dx == pytest.approx(6.0)
        check_grad(f, scalar(2.0))

    def test_exp_of_neg(self) -> None:
        """f(x) = exp(-x), df/dx = -exp(-x)."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return (-x).exp()

        val, (dx,) = eval_grad(f, scalar(1.0))
        assert val == pytest.approx(math.exp(-1.0))
        assert dx == pytest.approx(-math.exp(-1.0))
        check_grad(f, scalar(1.0))

    def test_nested_exp(self) -> None:
        """f(x) = exp(exp(x)), df/dx = exp(x) * exp(exp(x))."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.exp().exp()

        x_val = 0.5
        inner = math.exp(x_val)
        expected_val = math.exp(inner)
        expected_dx = inner * expected_val

        val, (dx,) = eval_grad(f, scalar(x_val))
        assert val == pytest.approx(expected_val, rel=1e-5)
        assert dx == pytest.approx(expected_dx, rel=1e-5)
        check_grad(f, scalar(x_val))

    def test_log_of_exp(self) -> None:
        """f(x) = log(exp(x)) = x, df/dx = 1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.exp().log()

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(3.0)
        assert dx == pytest.approx(1.0)
        check_grad(f, scalar(3.0))


class TestTwoVariables:
    def test_add(self) -> None:
        """f(x, y) = x + y, df/dx = 1, df/dy = 1."""

        def f(x: rc.Tensor, y: rc.Tensor) -> rc.Tensor:
            return x + y

        val, (dx, dy) = eval_grad(f, scalar(3.0), scalar(4.0))
        assert val == pytest.approx(7.0)
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(1.0)
        check_grad(f, scalar(3.0), scalar(4.0))

    def test_sub(self) -> None:
        """f(x, y) = x - y, df/dx = 1, df/dy = -1."""

        def f(x: rc.Tensor, y: rc.Tensor) -> rc.Tensor:
            return x - y

        val, (dx, dy) = eval_grad(f, scalar(5.0), scalar(2.0))
        assert val == pytest.approx(3.0)
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(-1.0)
        check_grad(f, scalar(5.0), scalar(2.0))

    def test_mul(self) -> None:
        """f(x, y) = x * y, df/dx = y, df/dy = x."""

        def f(x: rc.Tensor, y: rc.Tensor) -> rc.Tensor:
            return x * y

        val, (dx, dy) = eval_grad(f, scalar(3.0), scalar(4.0))
        assert val == pytest.approx(12.0)
        assert dx == pytest.approx(4.0)
        assert dy == pytest.approx(3.0)
        check_grad(f, scalar(3.0), scalar(4.0))

    def test_div(self) -> None:
        """f(x, y) = x / y, df/dx = 1/y, df/dy = -x/y^2."""

        def f(x: rc.Tensor, y: rc.Tensor) -> rc.Tensor:
            return x / y

        val, (dx, dy) = eval_grad(f, scalar(6.0), scalar(3.0))
        assert val == pytest.approx(2.0)
        assert dx == pytest.approx(1.0 / 3.0)
        assert dy == pytest.approx(-6.0 / 9.0)
        check_grad(f, scalar(6.0), scalar(3.0))

    def test_pow(self) -> None:
        """f(x, y) = x^y, df/dx = y*x^(y-1), df/dy = log(x)*x^y."""

        def f(x: rc.Tensor, y: rc.Tensor) -> rc.Tensor:
            return x**y

        x_val, y_val = 2.0, 3.0
        val, (dx, dy) = eval_grad(f, scalar(x_val), scalar(y_val))
        assert val == pytest.approx(8.0)
        assert dx == pytest.approx(y_val * x_val ** (y_val - 1))  # 12
        assert dy == pytest.approx(math.log(x_val) * x_val**y_val)  # log(2)*8
        check_grad(f, scalar(x_val), scalar(y_val))


class TestGradientAccumulation:
    """Tests for correct gradient accumulation when a tensor appears in multiple paths."""

    def test_x_plus_x(self) -> None:
        """f(x) = x + x, df/dx = 2."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x + x

        val, (dx,) = eval_grad(f, scalar(5.0))
        assert val == pytest.approx(10.0)
        assert dx == pytest.approx(2.0)
        check_grad(f, scalar(5.0))

    def test_x_times_x(self) -> None:
        """f(x) = x * x, df/dx = 2x. Gradient accumulated from both operand slots."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x * x

        val, (dx,) = eval_grad(f, scalar(4.0))
        assert val == pytest.approx(16.0)
        assert dx == pytest.approx(8.0)
        check_grad(f, scalar(4.0))

    def test_quadratic(self) -> None:
        """f(x) = x^2 + x, df/dx = 2x + 1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x * x + x

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(12.0)
        assert dx == pytest.approx(7.0)
        check_grad(f, scalar(3.0))

    def test_two_paths(self) -> None:
        """f(x) = exp(x) + log(x), df/dx = exp(x) + 1/x."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.exp() + x.log()

        x_val = 2.0
        val, (dx,) = eval_grad(f, scalar(x_val))
        assert val == pytest.approx(math.exp(x_val) + math.log(x_val))
        assert dx == pytest.approx(math.exp(x_val) + 1.0 / x_val)
        check_grad(f, scalar(x_val))

    def test_x_used_three_ways(self) -> None:
        """f(x) = x * x * x, df/dx = 3x^2. Three gradient contributions."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x * x * x

        val, (dx,) = eval_grad(f, scalar(2.0))
        assert val == pytest.approx(8.0)
        assert dx == pytest.approx(12.0)
        check_grad(f, scalar(2.0))


class TestPolynomials:
    def test_full_quadratic(self) -> None:
        """f(x) = 2x^2 + 5x + 1, df/dx = 4x + 5. At x=3: f=34, f'=17."""

        def f(x: rc.Tensor) -> rc.Tensor:
            a = rc.Tensor.const(value=2.0)
            b = rc.Tensor.const(value=5.0)
            c = rc.Tensor.const(value=1.0)
            return a * x * x + b * x + c

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(34.0)
        assert dx == pytest.approx(17.0)
        check_grad(f, scalar(3.0))

    def test_quadratic_at_zero(self) -> None:
        """f(x) = x^2 + x + 1, at x=0: f=1, f'=1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x * x + x + rc.Tensor.const(value=1.0)

        val, (dx,) = eval_grad(f, scalar(0.0))
        assert val == pytest.approx(1.0)
        assert dx == pytest.approx(1.0)
        check_grad(f, scalar(0.0))

    def test_quadratic_at_negative(self) -> None:
        """f(x) = x^2, at x=-3: f=9, f'=-6."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x * x

        val, (dx,) = eval_grad(f, scalar(-3.0))
        assert val == pytest.approx(9.0)
        assert dx == pytest.approx(-6.0)
        check_grad(f, scalar(-3.0))


class TestCaching:
    """Tests that grad's internal graph caching works correctly across evaluations."""

    def test_same_shape_different_values(self) -> None:
        """Calling the same grad function at different points must give correct results."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x * x

        g = rc.grad(f)
        for x_val in [0.0, 1.0, -2.0, 5.5, 0.1]:
            value_t, (grad_t,) = g(scalar(x_val))
            interp = rc.Interpreter()
            val = interp.evaluate(value_t).item()
            dx = interp.evaluate(grad_t).item()
            assert val == pytest.approx(x_val**2, abs=1e-5)
            assert dx == pytest.approx(2 * x_val, abs=1e-5)


class TestComposite:
    """More complex compositions that exercise multiple gradient rules together."""

    def test_exp_div(self) -> None:
        """f(x, y) = exp(x) / y, df/dx = exp(x)/y, df/dy = -exp(x)/y^2."""

        def f(x: rc.Tensor, y: rc.Tensor) -> rc.Tensor:
            return x.exp() / y

        x_val, y_val = 1.0, 2.0
        val, (dx, dy) = eval_grad(f, scalar(x_val), scalar(y_val))
        assert val == pytest.approx(math.exp(1.0) / 2.0)
        assert dx == pytest.approx(math.exp(1.0) / 2.0)
        assert dy == pytest.approx(-math.exp(1.0) / 4.0)
        check_grad(f, scalar(x_val), scalar(y_val))

    def test_product_of_logs(self) -> None:
        """f(x, y) = log(x) * log(y), df/dx = log(y)/x, df/dy = log(x)/y."""

        def f(x: rc.Tensor, y: rc.Tensor) -> rc.Tensor:
            return x.log() * y.log()

        x_val, y_val = 2.0, 3.0
        val, (dx, dy) = eval_grad(f, scalar(x_val), scalar(y_val))
        assert val == pytest.approx(math.log(2.0) * math.log(3.0))
        assert dx == pytest.approx(math.log(y_val) / x_val)
        assert dy == pytest.approx(math.log(x_val) / y_val)
        check_grad(f, scalar(x_val), scalar(y_val))

    def test_nested_arithmetic(self) -> None:
        """f(x) = (x + 1) * (x - 1) = x^2 - 1, df/dx = 2x."""

        def f(x: rc.Tensor) -> rc.Tensor:
            one = rc.Tensor.const(value=1.0)
            return (x + one) * (x - one)

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(8.0)
        assert dx == pytest.approx(6.0)
        check_grad(f, scalar(3.0))


class TestMaxMinGradients:
    """Tests for max/min operator gradients."""

    def test_max_first_operand_larger(self) -> None:
        """f(x) = max(x, 0) where x > 0, df/dx = 1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.__max__(rc.Tensor.const(value=0.0))

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(3.0)
        assert dx == pytest.approx(1.0)

    def test_max_second_operand_larger(self) -> None:
        """f(x) = max(x, 0) where x < 0, df/dx = 0."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.__max__(rc.Tensor.const(value=0.0))

        val, (dx,) = eval_grad(f, scalar(-3.0))
        assert val == pytest.approx(0.0)
        assert dx == pytest.approx(0.0)

    def test_max_with_constant(self) -> None:
        """f(x) = max(x, 5) where x > 5, df/dx = 1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.__max__(rc.Tensor.const(value=5.0))

        val, (dx,) = eval_grad(f, scalar(7.0))
        assert val == pytest.approx(7.0)
        assert dx == pytest.approx(1.0)

    def test_max_clipped_by_constant(self) -> None:
        """f(x) = max(x, 5) where x < 5, df/dx = 0."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.__max__(rc.Tensor.const(value=5.0))

        val, (dx,) = eval_grad(f, scalar(2.0))
        assert val == pytest.approx(5.0)
        assert dx == pytest.approx(0.0)

    def test_min_first_operand_smaller(self) -> None:
        """f(x) = min(x, 5) where x < 5, df/dx = 1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.__min__(rc.Tensor.const(value=5.0))

        val, (dx,) = eval_grad(f, scalar(2.0))
        assert val == pytest.approx(2.0)
        assert dx == pytest.approx(1.0)

    def test_min_second_operand_smaller(self) -> None:
        """f(x) = min(x, 5) where x > 5, df/dx = 0."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.__min__(rc.Tensor.const(value=5.0))

        val, (dx,) = eval_grad(f, scalar(8.0))
        assert val == pytest.approx(5.0)
        assert dx == pytest.approx(0.0)

    def test_min_clamp_upper(self) -> None:
        """f(x) = min(x, 10) where x < 10, passes through."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.__min__(rc.Tensor.const(value=10.0))

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(3.0)
        assert dx == pytest.approx(1.0)

    def test_min_clamp_upper_active(self) -> None:
        """f(x) = min(x, 10) where x > 10, clamped."""

        def f(x: rc.Tensor) -> rc.Tensor:
            return x.__min__(rc.Tensor.const(value=10.0))

        val, (dx,) = eval_grad(f, scalar(15.0))
        assert val == pytest.approx(10.0)
        assert dx == pytest.approx(0.0)


class TestReLUPatterns:
    """Tests for ReLU and related activation function patterns used in neural networks."""

    def test_relu_positive(self) -> None:
        """ReLU(x) = max(x, 0), x > 0 -> f = x, f' = 1."""

        def relu(x: rc.Tensor) -> rc.Tensor:
            return x.__max__(rc.Tensor.const(value=0.0))

        val, (dx,) = eval_grad(relu, scalar(5.0))
        assert val == pytest.approx(5.0)
        assert dx == pytest.approx(1.0)

    def test_relu_negative(self) -> None:
        """ReLU(x) = max(x, 0), x < 0 -> f = 0, f' = 0."""

        def relu(x: rc.Tensor) -> rc.Tensor:
            return x.__max__(rc.Tensor.const(value=0.0))

        val, (dx,) = eval_grad(relu, scalar(-5.0))
        assert val == pytest.approx(0.0)
        assert dx == pytest.approx(0.0)

    def test_relu_at_zero(self) -> None:
        """ReLU(x) = max(x, 0), x = 0 -> f = 0, f' = 0 (subgradient)."""

        def relu(x: rc.Tensor) -> rc.Tensor:
            return x.__max__(rc.Tensor.const(value=0.0))

        val, (dx,) = eval_grad(relu, scalar(0.0))
        assert val == pytest.approx(0.0)
        assert dx == pytest.approx(0.0)

    def test_relu_of_linear(self) -> None:
        """f(x) = ReLU(2x - 3). At x=2: 2*2-3=1>0, f=1, f'=2. At x=1: 2*1-3=-1<0, f=0, f'=0."""

        def f(x: rc.Tensor) -> rc.Tensor:
            linear = rc.Tensor.const(value=2.0) * x - rc.Tensor.const(value=3.0)
            return linear.__max__(rc.Tensor.const(value=0.0))

        val, (dx,) = eval_grad(f, scalar(2.0))
        assert val == pytest.approx(1.0)
        assert dx == pytest.approx(2.0)

        val, (dx,) = eval_grad(f, scalar(1.0))
        assert val == pytest.approx(0.0)
        assert dx == pytest.approx(0.0)

    def test_relu_composed_with_square(self) -> None:
        """f(x) = ReLU(x)^2. At x=3: f=9, f'=6. At x=-3: f=0, f'=0."""

        def f(x: rc.Tensor) -> rc.Tensor:
            r = x.__max__(rc.Tensor.const(value=0.0))
            return r * r

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(9.0)
        assert dx == pytest.approx(6.0)

        val, (dx,) = eval_grad(f, scalar(-3.0))
        assert val == pytest.approx(0.0)
        assert dx == pytest.approx(0.0)

    def test_sum_of_relus(self) -> None:
        """f(x) = ReLU(x) + ReLU(-x) = |x|. At x=3: f=3, f'=1. At x=-3: f=3, f'=-1."""

        def f(x: rc.Tensor) -> rc.Tensor:
            pos = x.__max__(rc.Tensor.const(value=0.0))
            neg = (-x).__max__(rc.Tensor.const(value=0.0))
            return pos + neg

        val, (dx,) = eval_grad(f, scalar(3.0))
        assert val == pytest.approx(3.0)
        assert dx == pytest.approx(1.0)

        val, (dx,) = eval_grad(f, scalar(-3.0))
        assert val == pytest.approx(3.0)
        assert dx == pytest.approx(-1.0)

    def test_clamp(self) -> None:
        """clamp(x, lo, hi) = min(max(x, lo), hi). Tests gradient through both max and min."""

        def clamp(x: rc.Tensor) -> rc.Tensor:
            lo = rc.Tensor.const(value=-1.0)
            hi = rc.Tensor.const(value=1.0)
            return x.__max__(lo).__min__(hi)

        # In range: gradient passes through
        val, (dx,) = eval_grad(clamp, scalar(0.5))
        assert val == pytest.approx(0.5)
        assert dx == pytest.approx(1.0)

        # Clamped low: gradient is zero
        val, (dx,) = eval_grad(clamp, scalar(-5.0))
        assert val == pytest.approx(-1.0)
        assert dx == pytest.approx(0.0)

        # Clamped high: gradient is zero
        val, (dx,) = eval_grad(clamp, scalar(5.0))
        assert val == pytest.approx(1.0)
        assert dx == pytest.approx(0.0)

    def test_leaky_relu(self) -> None:
        """LeakyReLU(x) = max(x, 0.01*x). At x>0: f=x, f'=1. At x<0: f=0.01*x, f'=0.01."""

        def leaky_relu(x: rc.Tensor) -> rc.Tensor:
            alpha = rc.Tensor.const(value=0.01)
            return x.__max__(alpha * x)

        val, (dx,) = eval_grad(leaky_relu, scalar(3.0))
        assert val == pytest.approx(3.0)
        assert dx == pytest.approx(1.0)

        val, (dx,) = eval_grad(leaky_relu, scalar(-3.0))
        assert val == pytest.approx(-0.03)
        assert dx == pytest.approx(0.01)

    def test_relu_chain(self) -> None:
        """f(x) = ReLU(ReLU(x) - 2). At x=5: inner=5, f=3, f'=1. At x=1: inner=1, f=0, f'=0."""

        def f(x: rc.Tensor) -> rc.Tensor:
            r1 = x.__max__(rc.Tensor.const(value=0.0))
            return (r1 - rc.Tensor.const(value=2.0)).__max__(rc.Tensor.const(value=0.0))

        val, (dx,) = eval_grad(f, scalar(5.0))
        assert val == pytest.approx(3.0)
        assert dx == pytest.approx(1.0)

        val, (dx,) = eval_grad(f, scalar(1.0))
        assert val == pytest.approx(0.0)
        assert dx == pytest.approx(0.0)

        val, (dx,) = eval_grad(f, scalar(-1.0))
        assert val == pytest.approx(0.0)
        assert dx == pytest.approx(0.0)

    def test_softplus_vs_relu(self) -> None:
        """Softplus log(1 + exp(x)) approximates ReLU. Verify both have similar gradients for large |x|."""

        def softplus(x: rc.Tensor) -> rc.Tensor:
            return (rc.Tensor.const(value=1.0) + x.exp()).log()

        def relu(x: rc.Tensor) -> rc.Tensor:
            return x.__max__(rc.Tensor.const(value=0.0))

        # For large positive x, both gradients should be ~1
        _, (sp_dx,) = eval_grad(softplus, scalar(10.0))
        _, (relu_dx,) = eval_grad(relu, scalar(10.0))
        assert sp_dx == pytest.approx(1.0, abs=1e-4)
        assert relu_dx == pytest.approx(1.0)

        # For large negative x, both gradients should be ~0
        _, (sp_dx,) = eval_grad(softplus, scalar(-10.0))
        _, (relu_dx,) = eval_grad(relu, scalar(-10.0))
        assert sp_dx == pytest.approx(0.0, abs=1e-4)
        assert relu_dx == pytest.approx(0.0)


class TestComparisonGradientErrors:
    """Comparison operators should raise ValueError when gradient is attempted."""

    def test_eq_raises(self) -> None:
        def f(x: rc.Tensor) -> rc.Tensor:
            return x.eq(rc.Tensor.const(value=1.0))

        with pytest.raises(ValueError, match="Cannot compute gradients for comparison"):
            rc.grad(f)(scalar(1.0))

    def test_ne_raises(self) -> None:
        def f(x: rc.Tensor) -> rc.Tensor:
            return x.ne(rc.Tensor.const(value=1.0))

        with pytest.raises(ValueError, match="Cannot compute gradients for comparison"):
            rc.grad(f)(scalar(1.0))

    def test_lt_raises(self) -> None:
        def f(x: rc.Tensor) -> rc.Tensor:
            return x < rc.Tensor.const(value=1.0)

        with pytest.raises(ValueError, match="Cannot compute gradients for comparison"):
            rc.grad(f)(scalar(0.0))

    def test_gt_raises(self) -> None:
        def f(x: rc.Tensor) -> rc.Tensor:
            return x > rc.Tensor.const(value=1.0)

        with pytest.raises(ValueError, match="Cannot compute gradients for comparison"):
            rc.grad(f)(scalar(2.0))

    def test_le_raises(self) -> None:
        def f(x: rc.Tensor) -> rc.Tensor:
            return x <= rc.Tensor.const(value=1.0)

        with pytest.raises(ValueError, match="Cannot compute gradients for comparison"):
            rc.grad(f)(scalar(0.0))

    def test_ge_raises(self) -> None:
        def f(x: rc.Tensor) -> rc.Tensor:
            return x >= rc.Tensor.const(value=1.0)

        with pytest.raises(ValueError, match="Cannot compute gradients for comparison"):
            rc.grad(f)(scalar(2.0))
