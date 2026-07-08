//! Public API: build graphs with [`dsl`], bind functions with [`jit::Jit::jit`],
//! and run them on concrete backend arrays.
//!
//! Middle-end IR and lowered programs are internal to the JIT pipeline.

pub use resin_core as core;
pub use resin_dataset as dataset;
pub use resin_dsl as dsl;
pub use resin_jit as jit;
pub use resin_macros as macros;

#[cfg(test)]
mod tests {
    use crate::dsl::{ElementType, Tensor};
    use crate::jit::Jit;
    use resin_macros::Tree;

    #[cfg(feature = "cpu")]
    use crate::jit::backends::cpu::CpuJit;
    #[cfg(feature = "wgpu")]
    use crate::jit::backends::wgpu::WgpuJit;

    #[derive(Tree)]
    struct Inputs<J: Jit> {
        a: J::Tensor,
        b: J::Tensor,
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn cpu_jit_with_generic_inputs() {
        let jit = CpuJit;
        let add = jit.jit(|inputs: &InputsMapped<Tensor>| inputs.a.clone() + inputs.b.clone());
        let params: Inputs<CpuJit> = Inputs {
            a: jit.zeros(&[2, 3], ElementType::F32),
            b: jit.zeros(&[2, 3], ElementType::F32),
        };

        add.call(&params).expect("cpu jit add");
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_jit_with_generic_inputs() {
        use crate::jit::{JitError, RunError};

        let jit = WgpuJit;
        let add = jit.jit(|inputs: &InputsMapped<Tensor>| inputs.a.clone() - inputs.b.clone());
        let params: Inputs<WgpuJit> = Inputs {
            a: jit.zeros(&[4], ElementType::F32),
            b: jit.zeros(&[4], ElementType::F32),
        };

        assert!(matches!(
            add.call(&params),
            Err(JitError::Run(RunError::NotImplemented))
        ));
    }

    #[test]
    fn backends_implement_jit_trait() {
        fn assert_jit<J: Jit>(_jit: J) {}
        #[cfg(feature = "cpu")]
        assert_jit(CpuJit);
        #[cfg(feature = "wgpu")]
        assert_jit(WgpuJit);
    }
}