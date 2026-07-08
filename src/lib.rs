//! Public API: build graphs with [`dsl`], bind functions with [`jit::Jit::jit`],
//! and run them on concrete backend arrays.
//!
//! Middle-end IR and lowered programs are internal to the JIT pipeline.

pub use resin_core as core;
pub use resin_dataset as dataset;
pub use resin_dsl as dsl;
pub use resin_gaussians as gaussians;
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
        use crate::jit::ConcreteTensor;

        // Skip when the machine has no GPU adapter (CI without Metal/Vulkan).
        if crate::jit::backends::wgpu::shared_context_available() == false {
            eprintln!("skip wgpu_jit_with_generic_inputs: no GPU");
            return;
        }

        let jit = WgpuJit;
        let add = jit.jit(|inputs: &InputsMapped<Tensor>| inputs.a.clone() + inputs.b.clone());
        let params: Inputs<WgpuJit> = Inputs {
            a: crate::jit::backends::wgpu::WgpuTensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]),
            b: crate::jit::backends::wgpu::WgpuTensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]),
        };

        let out = add.call(&params).expect("wgpu jit add");
        assert_eq!(out.to_f32(), vec![11.0, 22.0, 33.0, 44.0]);
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