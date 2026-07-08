//! Fixed-function hardware kernels: ray tracing and rasterization.
//!
//! These kernels are *visibility only* — they output which primitive was hit
//! (with barycentrics), never shaded appearance. Shading composes in the
//! graph from `Remap` gathers and elementwise math (`docs/hw-nodes.md`).

use resin_core::{
    hw::{RASTER_PIXEL_WIDTH, TRACE_HIT_WIDTH},
    ElementType,
};
use serde::{Deserialize, Serialize};

use crate::error::IrError;
use resin_core::Accessor;

/// Closest-hit ray tracing against triangle geometry.
///
/// Args (in order): `origins [N,3] F4`, `directions [N,3] F4`, `t_min [N] F4`,
/// `t_max [N] F4`, `vertices [V,3] F4`, `triangles [T,3] U4`.
/// Output: `[N, TRACE_HIT_WIDTH] F4` hit records `(t, u, v, prim, hit)`.
///
/// Backends may run this on ray-tracing hardware (acceleration structure +
/// ray query) or via a compute/CPU fallback; results agree up to
/// watertightness at triangle edges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrTraceRaysKernel {
    pub arg_accessors: Vec<Accessor>,
    pub arg_element_types: Vec<ElementType>,
    pub element_type: ElementType,
    pub shape: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

/// Depth-tested triangle rasterization into a visibility buffer.
///
/// Args (in order): `clip_positions [V,4] F4`, `triangles [T,3] U4`.
/// Output: `[H, W, RASTER_PIXEL_WIDTH] F4` pixels `(prim, hit, u, v)` for the
/// depth-nearest triangle covering each pixel center (perspective-correct
/// barycentrics). Pixel `(r, c)` samples NDC `x = (c+0.5)/W·2−1`,
/// `y = 1−(r+0.5)/H·2`; depth is NDC `z ∈ [0,1]`, closer-wins; no culling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrRasterizeKernel {
    pub arg_accessors: Vec<Accessor>,
    pub arg_element_types: Vec<ElementType>,
    pub element_type: ElementType,
    pub shape: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

const TRACE_ARGS: usize = 6;
const RASTER_ARGS: usize = 2;

impl IrTraceRaysKernel {
    /// One thread per ray.
    pub fn thread_shape(&self) -> &[u32] {
        &self.shape[..1]
    }

    pub fn ray_count(&self) -> u32 {
        self.shape[0]
    }

    pub fn vertex_count(&self) -> u32 {
        self.arg_accessors[4].shape[0]
    }

    pub fn triangle_count(&self) -> u32 {
        self.arg_accessors[5].shape[0]
    }

    pub fn validate(&self) -> Result<(), IrError> {
        if self.arg_accessors.len() != TRACE_ARGS
            || self.arg_element_types.len() != TRACE_ARGS
        {
            return Err(IrError::HwArgCount {
                kernel: "trace_rays",
                expected: TRACE_ARGS,
                got: self.arg_accessors.len().min(self.arg_element_types.len()),
            });
        }
        let expected_types = [
            ElementType::F4,
            ElementType::F4,
            ElementType::F4,
            ElementType::F4,
            ElementType::F4,
            ElementType::U4,
        ];
        for (i, (&got, &want)) in self
            .arg_element_types
            .iter()
            .zip(expected_types.iter())
            .enumerate()
        {
            if got != want {
                return Err(IrError::HwArgElementType {
                    kernel: "trace_rays",
                    arg: i,
                    expected: want,
                    got,
                });
            }
        }
        if self.element_type != ElementType::F4 {
            return Err(IrError::HwOutputElementType {
                kernel: "trace_rays",
                got: self.element_type,
            });
        }
        // Rank checks precede every leading-dim read below: validate() must
        // return errors for malformed (e.g. deserialized) IR, never panic.
        if self.shape.len() != 2 || self.shape[1] != TRACE_HIT_WIDTH as u32 {
            return Err(IrError::HwOutputShape {
                kernel: "trace_rays",
                got: self.shape.clone(),
            });
        }
        let expect = |arg: usize, shape: &[u32]| -> Result<(), IrError> {
            if self.arg_accessors[arg].shape.as_ref() != shape {
                return Err(IrError::HwArgShape {
                    kernel: "trace_rays",
                    arg,
                    expected: shape.into(),
                    got: self.arg_accessors[arg].shape.clone(),
                });
            }
            Ok(())
        };
        let rows3 = |arg: usize| -> Result<u32, IrError> {
            let shape = &self.arg_accessors[arg].shape;
            if shape.len() != 2 || shape[1] != 3 {
                return Err(IrError::HwArgShape {
                    kernel: "trace_rays",
                    arg,
                    expected: Box::from([shape.first().copied().unwrap_or(0), 3]),
                    got: shape.clone(),
                });
            }
            Ok(shape[0])
        };
        let n = self.ray_count();
        expect(0, &[n, 3])?;
        expect(1, &[n, 3])?;
        expect(2, &[n])?;
        expect(3, &[n])?;
        rows3(4)?;
        rows3(5)?;
        Ok(())
    }
}

impl IrRasterizeKernel {
    /// One thread per pixel (for compute-style executors; hardware backends
    /// use a render pass instead).
    pub fn thread_shape(&self) -> &[u32] {
        &self.shape[..2]
    }

    pub fn height(&self) -> u32 {
        self.shape[0]
    }

    pub fn width(&self) -> u32 {
        self.shape[1]
    }

    pub fn vertex_count(&self) -> u32 {
        self.arg_accessors[0].shape[0]
    }

    pub fn triangle_count(&self) -> u32 {
        self.arg_accessors[1].shape[0]
    }

    pub fn validate(&self) -> Result<(), IrError> {
        if self.arg_accessors.len() != RASTER_ARGS
            || self.arg_element_types.len() != RASTER_ARGS
        {
            return Err(IrError::HwArgCount {
                kernel: "rasterize",
                expected: RASTER_ARGS,
                got: self.arg_accessors.len().min(self.arg_element_types.len()),
            });
        }
        if self.arg_element_types[0] != ElementType::F4 {
            return Err(IrError::HwArgElementType {
                kernel: "rasterize",
                arg: 0,
                expected: ElementType::F4,
                got: self.arg_element_types[0],
            });
        }
        if self.arg_element_types[1] != ElementType::U4 {
            return Err(IrError::HwArgElementType {
                kernel: "rasterize",
                arg: 1,
                expected: ElementType::U4,
                got: self.arg_element_types[1],
            });
        }
        if self.element_type != ElementType::F4 {
            return Err(IrError::HwOutputElementType {
                kernel: "rasterize",
                got: self.element_type,
            });
        }
        if self.shape.len() != 3 || self.shape[2] != RASTER_PIXEL_WIDTH as u32 {
            return Err(IrError::HwOutputShape {
                kernel: "rasterize",
                got: self.shape.clone(),
            });
        }
        // The background must be zeroed (hit = 0) before geometry lands.
        if !self.clear_output_before_dispatch {
            return Err(IrError::HwMustClear { kernel: "rasterize" });
        }
        // Rank checks precede the leading-dim reads (vertex_count /
        // triangle_count index shape[0]): malformed IR must error, not panic.
        for (arg, trailing) in [(0usize, 4u32), (1, 3)] {
            let shape = &self.arg_accessors[arg].shape;
            if shape.len() != 2 || shape[1] != trailing {
                return Err(IrError::HwArgShape {
                    kernel: "rasterize",
                    arg,
                    expected: Box::from([shape.first().copied().unwrap_or(0), trailing]),
                    got: shape.clone(),
                });
            }
        }
        Ok(())
    }
}
