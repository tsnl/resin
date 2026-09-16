//! Calls into the scalar/pointer C runtime ABI and immutable diagnostic data.

use super::{failure, function::Body};
use crate::Error;
use cranelift_codegen::ir::{self, InstBuilder};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{DataDescription, Linkage, Module};
use cranelift_object::ObjectModule;

pub(super) fn call(
    module: &mut ObjectModule,
    builder: &mut FunctionBuilder<'_>,
    name: &str,
    args: &[(ir::Type, ir::Value)],
    result: Option<ir::Type>,
) -> Result<Option<ir::Value>, Error> {
    let mut signature = module.make_signature();
    signature
        .params
        .extend(args.iter().map(|(ty, _)| ir::AbiParam::new(*ty)));
    signature.returns.extend(result.map(ir::AbiParam::new));
    let function = module
        .declare_function(name, Linkage::Import, &signature)
        .map_err(failure)?;
    let reference = module.declare_func_in_func(function, builder.func);
    let values = args.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    let instruction = builder.ins().call(reference, &values);
    Ok(result.map(|_| builder.inst_results(instruction)[0]))
}

/// The trailing NUL is outside the supplied bytes' logical length.
pub(super) fn data(
    module: &mut ObjectModule,
    builder: &mut FunctionBuilder<'_>,
    bytes: &[u8],
) -> Result<ir::Value, Error> {
    let data = module
        .declare_anonymous_data(false, false)
        .map_err(failure)?;
    let mut description = DataDescription::new();
    description.define(
        bytes
            .iter()
            .copied()
            .chain([0])
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    description.set_align(1);
    module.define_data(data, &description).map_err(failure)?;
    let reference = module.declare_data_in_func(data, builder.func);
    Ok(builder.ins().symbol_value(ir::types::I64, reference))
}

/// Continue only when `invalid` is false. The failure block preserves Resin's
/// diagnostic and exit behavior; the trap marks the non-returning call's end.
pub(super) fn guard_symbol(
    module: &mut ObjectModule,
    builder: &mut FunctionBuilder<'_>,
    invalid: ir::Value,
    message: &str,
    failure: &str,
) -> Result<(), Error> {
    let failed = builder.create_block();
    let next = builder.create_block();
    builder.ins().brif(invalid, failed, &[], next, &[]);
    builder.switch_to_block(failed);
    let message = data(module, builder, message.as_bytes())?;
    call(module, builder, failure, &[(ir::types::I64, message)], None)?;
    builder.ins().trap(ir::TrapCode::unwrap_user(1));
    builder.switch_to_block(next);
    Ok(())
}

impl Body<'_, '_> {
    pub fn runtime_call(
        &mut self,
        name: &str,
        args: &[(ir::Type, ir::Value)],
        result: Option<ir::Type>,
    ) -> Result<Option<ir::Value>, Error> {
        let name = self
            .types
            .runtime
            .get(name)
            .map_or(name, |name| name.as_ref());
        call(self.module, &mut self.builder, name, args, result)
    }

    pub fn guard(&mut self, invalid: ir::Value, message: &str) -> Result<(), Error> {
        let failure = self
            .types
            .runtime
            .get("resin_fail")
            .map_or("resin_fail", |name| name.as_ref());
        guard_symbol(self.module, &mut self.builder, invalid, message, failure)
    }
}
