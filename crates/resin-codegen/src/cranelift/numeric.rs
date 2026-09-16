use super::runtime;
use super::{function::Operand, scalar_type, unsupported};
use crate::Error;
use cranelift_codegen::ir::{
    self, InstBuilder,
    condcodes::{FloatCC, IntCC},
};
use cranelift_frontend::FunctionBuilder;
use cranelift_object::ObjectModule;
use resin_types::prelude::*;

pub(super) fn builtin(
    module: &mut ObjectModule,
    builder: &mut FunctionBuilder<'_>,
    name: &str,
    args: &[Operand],
    failure_name: &str,
) -> Result<ir::Value, Error> {
    let Some(first) = args.first() else {
        return Err(unsupported(format!("builtin {name:?}")));
    };
    if args.iter().any(|arg| arg.ty != first.ty) {
        return Err(unsupported(format!("mixed operands for {name:?}")));
    }
    let a = first.value;
    if args.len() == 1 {
        return Ok(match name {
            "+" if first.ty.is_numeric() => a,
            "-" if first.ty.is_integer() => builder.ins().ineg(a),
            "-" if first.ty.is_numeric() => builder.ins().fneg(a),
            "~" if first.ty.is_integer() => builder.ins().bnot(a),
            "!" if first.ty == Ty::Bool => builder.ins().icmp_imm_s(IntCC::Equal, a, 0),
            _ => return Err(unsupported(format!("unary builtin {name:?}"))),
        });
    }
    if args.len() != 2 {
        return Err(unsupported(format!("builtin {name:?}")));
    }
    let b = args[1].value;
    if matches!(name, "==" | "!=" | "<" | "<=" | ">" | ">=") {
        return Ok(comparison(builder, name, &first.ty, a, b));
    }
    if first.ty.is_integer() {
        return integer(module, builder, name, first, b, failure_name);
    }
    Ok(match name {
        "&&" if first.ty == Ty::Bool => builder.ins().band(a, b),
        "||" if first.ty == Ty::Bool => builder.ins().bor(a, b),
        "+" if first.ty.is_numeric() => builder.ins().fadd(a, b),
        "-" if first.ty.is_numeric() => builder.ins().fsub(a, b),
        "*" if first.ty.is_numeric() => builder.ins().fmul(a, b),
        "/" if first.ty.is_numeric() => builder.ins().fdiv(a, b),
        _ => return Err(unsupported(format!("binary builtin {name:?}"))),
    })
}

fn comparison(
    builder: &mut FunctionBuilder<'_>,
    name: &str,
    ty: &Ty,
    a: ir::Value,
    b: ir::Value,
) -> ir::Value {
    if matches!(ty, Ty::Float32 | Ty::Float64) {
        let condition = match name {
            "==" => FloatCC::Equal,
            "!=" => FloatCC::NotEqual,
            "<" => FloatCC::LessThan,
            "<=" => FloatCC::LessThanOrEqual,
            ">" => FloatCC::GreaterThan,
            ">=" => FloatCC::GreaterThanOrEqual,
            _ => unreachable!(),
        };
        return builder.ins().fcmp(condition, a, b);
    }
    let condition = match (name, signed(ty)) {
        ("==", _) => IntCC::Equal,
        ("!=", _) => IntCC::NotEqual,
        ("<", true) => IntCC::SignedLessThan,
        ("<=", true) => IntCC::SignedLessThanOrEqual,
        (">", true) => IntCC::SignedGreaterThan,
        (">=", true) => IntCC::SignedGreaterThanOrEqual,
        ("<", false) => IntCC::UnsignedLessThan,
        ("<=", false) => IntCC::UnsignedLessThanOrEqual,
        (">", false) => IntCC::UnsignedGreaterThan,
        (">=", false) => IntCC::UnsignedGreaterThanOrEqual,
        _ => unreachable!(),
    };
    builder.ins().icmp(condition, a, b)
}

fn integer(
    module: &mut ObjectModule,
    builder: &mut FunctionBuilder<'_>,
    name: &str,
    first: &Operand,
    b: ir::Value,
    failure_name: &str,
) -> Result<ir::Value, Error> {
    let a = first.value;
    Ok(match name {
        "+" => builder.ins().iadd(a, b),
        "-" => builder.ins().isub(a, b),
        "*" => builder.ins().imul(a, b),
        "&" => builder.ins().band(a, b),
        "|" => builder.ins().bor(a, b),
        "^" => builder.ins().bxor(a, b),
        "/" | "%" => divide(module, builder, name, first, b, failure_name)?,
        "<<" | ">>" => {
            let bits = scalar_type(&first.ty)?.bits();
            let invalid =
                builder
                    .ins()
                    .icmp_imm_s(IntCC::UnsignedGreaterThanOrEqual, b, i64::from(bits));
            runtime::guard_symbol(
                module,
                builder,
                invalid,
                "shift count out of range",
                failure_name,
            )?;
            if name == "<<" {
                builder.ins().ishl(a, b)
            } else if signed(&first.ty) {
                builder.ins().sshr(a, b)
            } else {
                builder.ins().ushr(a, b)
            }
        }
        _ => return Err(unsupported(format!("integer builtin {name:?}"))),
    })
}

fn divide(
    module: &mut ObjectModule,
    builder: &mut FunctionBuilder<'_>,
    name: &str,
    first: &Operand,
    divisor: ir::Value,
    failure_name: &str,
) -> Result<ir::Value, Error> {
    let original = scalar_type(&first.ty)?;
    // Widen first to avoid target restrictions on narrow integer division.
    // Truncating the result afterward preserves wrapping at the source width.
    let a = resize(
        builder,
        first.value,
        original,
        ir::types::I64,
        signed(&first.ty),
    );
    let b = resize(
        builder,
        divisor,
        original,
        ir::types::I64,
        signed(&first.ty),
    );
    let invalid = builder.ins().icmp_imm_s(IntCC::Equal, b, 0);
    runtime::guard_symbol(module, builder, invalid, "division by zero", failure_name)?;
    let value = if signed(&first.ty) {
        // Resin's signed division wraps MIN/-1. Cranelift traps for that pair;
        // use a safe divisor and explicitly select the language's wrapped value.
        let minimum = builder.ins().icmp_imm_s(IntCC::Equal, a, i64::MIN);
        let negative_one = builder.ins().icmp_imm_s(IntCC::Equal, b, -1);
        let wraps = builder.ins().band(minimum, negative_one);
        let one = builder.ins().iconst(ir::types::I64, 1);
        let safe = builder.ins().select(wraps, one, b);
        if name == "/" {
            builder.ins().sdiv(a, safe)
        } else {
            builder.ins().srem(a, safe)
        }
    } else if name == "/" {
        builder.ins().udiv(a, b)
    } else {
        builder.ins().urem(a, b)
    };
    Ok(resize(
        builder,
        value,
        ir::types::I64,
        original,
        signed(&first.ty),
    ))
}

pub(super) fn convert(
    module: &mut ObjectModule,
    builder: &mut FunctionBuilder<'_>,
    operand: &Operand,
    to: &Ty,
    failure_name: &str,
) -> Result<ir::Value, Error> {
    let from = &operand.ty;
    let source = scalar_type(from)?;
    let target = scalar_type(to)?;
    let value = operand.value;
    if from == to {
        return Ok(value);
    }
    if from.is_integer() && to.is_integer() {
        integer_bounds(module, builder, operand, to, failure_name)?;
        return Ok(resize(builder, value, source, target, signed(from)));
    }
    if from.is_integer() && matches!(to, Ty::Float32 | Ty::Float64) {
        let integer = resize(builder, value, source, ir::types::I64, signed(from));
        return Ok(if signed(from) {
            builder.ins().fcvt_from_sint(target, integer)
        } else {
            builder.ins().fcvt_from_uint(target, integer)
        });
    }
    if matches!(from, Ty::Float32 | Ty::Float64) && to.is_integer() {
        return float_to_integer(module, builder, operand, to, failure_name);
    }
    if *from == Ty::Float32 && *to == Ty::Float64 {
        return Ok(builder.ins().fpromote(target, value));
    }
    if *from == Ty::Float64 && *to == Ty::Float32 {
        return Ok(demote(builder, value));
    }
    Err(unsupported(format!("numeric cast {from:?} to {to:?}")))
}

fn integer_bounds(
    module: &mut ObjectModule,
    builder: &mut FunctionBuilder<'_>,
    operand: &Operand,
    to: &Ty,
    failure_name: &str,
) -> Result<(), Error> {
    let from = scalar_type(&operand.ty)?;
    let target = scalar_type(to)?;
    let (low, high) = range(target.bits(), signed(to));
    let (source_low, source_high) = range(from.bits(), signed(&operand.ty));
    if low > source_low {
        let invalid = builder
            .ins()
            .icmp_imm_s(IntCC::SignedLessThan, operand.value, low as i64);
        runtime::guard_symbol(
            module,
            builder,
            invalid,
            "numeric conversion out of range",
            failure_name,
        )?;
    }
    if high < source_high {
        let condition = if signed(&operand.ty) {
            IntCC::SignedGreaterThan
        } else {
            IntCC::UnsignedGreaterThan
        };
        let invalid = builder
            .ins()
            .icmp_imm_s(condition, operand.value, high as i64);
        runtime::guard_symbol(
            module,
            builder,
            invalid,
            "numeric conversion out of range",
            failure_name,
        )?;
    }
    Ok(())
}

fn float_to_integer(
    module: &mut ObjectModule,
    builder: &mut FunctionBuilder<'_>,
    operand: &Operand,
    to: &Ty,
    failure_name: &str,
) -> Result<ir::Value, Error> {
    let target = scalar_type(to)?;
    let source = scalar_type(&operand.ty)?;
    let bits = target.bits();
    let low = if signed(to) {
        -(2_f64.powi(bits as i32 - 1))
    } else {
        0.0
    };
    let high = 2_f64.powi(bits as i32 - i32::from(signed(to)));
    let truncated = builder.ins().trunc(operand.value);
    let lower = float_constant(builder, source, low);
    let upper = float_constant(builder, source, high);
    let above = builder
        .ins()
        .fcmp(FloatCC::GreaterThanOrEqual, truncated, lower);
    let below = builder.ins().fcmp(FloatCC::LessThan, truncated, upper);
    let valid = builder.ins().band(above, below);
    let invalid = builder.ins().icmp_imm_s(IntCC::Equal, valid, 0);
    runtime::guard_symbol(
        module,
        builder,
        invalid,
        "numeric conversion out of range",
        failure_name,
    )?;
    let result = if signed(to) {
        builder.ins().fcvt_to_sint(ir::types::I64, truncated)
    } else {
        builder.ins().fcvt_to_uint(ir::types::I64, truncated)
    };
    Ok(resize(builder, result, ir::types::I64, target, signed(to)))
}

// Resin's explicit f64 -> f32 conversion flushes values below the minimum normal
// f32 to signed zero and any value beyond MAX to infinity, even when hardware
// rounding would produce MAX. Preserve those contracts independently of CPU mode.
fn demote(builder: &mut FunctionBuilder<'_>, value: ir::Value) -> ir::Value {
    let converted = builder.ins().fdemote(ir::types::F32, value);
    let magnitude = builder.ins().fabs(value);
    let minimum = builder.ins().f64const(f64::from(f32::MIN_POSITIVE));
    let tiny = builder.ins().fcmp(FloatCC::LessThan, magnitude, minimum);
    let zero = builder.ins().f32const(0.0);
    let signed_zero = builder.ins().fcopysign(zero, converted);
    let normal = builder.ins().select(tiny, signed_zero, converted);
    let maximum = builder.ins().f64const(f64::from(f32::MAX));
    let huge = builder.ins().fcmp(FloatCC::GreaterThan, magnitude, maximum);
    let infinity = builder.ins().f32const(f32::INFINITY);
    let signed_infinity = builder.ins().fcopysign(infinity, converted);
    builder.ins().select(huge, signed_infinity, normal)
}

fn float_constant(builder: &mut FunctionBuilder<'_>, ty: ir::Type, value: f64) -> ir::Value {
    if ty == ir::types::F32 {
        builder.ins().f32const(value as f32)
    } else {
        builder.ins().f64const(value)
    }
}

fn resize(
    builder: &mut FunctionBuilder<'_>,
    value: ir::Value,
    from: ir::Type,
    to: ir::Type,
    signed: bool,
) -> ir::Value {
    if from == to {
        value
    } else if from.bits() > to.bits() {
        builder.ins().ireduce(to, value)
    } else if signed {
        builder.ins().sextend(to, value)
    } else {
        builder.ins().uextend(to, value)
    }
}

fn range(bits: u32, signed: bool) -> (i128, i128) {
    if signed {
        (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1)
    } else {
        (0, (1_i128 << bits) - 1)
    }
}

fn signed(ty: &Ty) -> bool {
    matches!(ty, Ty::Int8 | Ty::Int16 | Ty::Int32 | Ty::Int64)
}
