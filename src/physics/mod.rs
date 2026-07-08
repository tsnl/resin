//! Rigid-body physics, compiled as tensor programs.
//!
//! A stress test of the compiler on a non-ML workload — and a demonstration
//! that a physics engine can *be* a tensor program: [`Scene::trace_step`]
//! expresses a whole simulation frame (gravity, joints, collisions,
//! friction) as one pure `State -> State` graph in the [`crate::dsl`], so it
//! runs unchanged on any backend and is reverse-mode differentiable end to
//! end.
//!
//! **Batched from the start.** Every [`State`] tensor carries a leading
//! *worlds* axis: `pos` is `[worlds, bodies, 3]`. Internally each vector
//! component is a `[bodies, worlds]` matrix, so gathering a constraint's
//! bodies is one matmul that touches every world at once — the compiled
//! step advances thousands of independent simulations with the same handful
//! of kernels, and no per-world control flow anywhere (the pattern GPU
//! engines like MuJoCo-Warp and Genesis are built around). Simulating one
//! world just means `worlds = 1`.
//!
//! The solver is XPBD (position-based dynamics); see [`solver`] for the
//! algorithm and [`contact`] for how collisions become branch-free tensor
//! ops. Honest limits, chosen for readability: spheres, cuboids, and a
//! ground plane; no broadphase (all candidate pairs, relu-gated); Jacobi
//! constraint rounds instead of Gauss-Seidel; f32 everywhere.
//!
//! ```no_run
//! use resin::jit::CpuJit;
//! use resin::physics::{Body, Scene, Shape, Simulation, WORLD};
//!
//! let mut scene = Scene::new();
//! let ball = scene.add(Body {
//!     shape: Shape::Sphere { radius: 0.5 },
//!     position: [0.0, 3.0, 0.0],
//!     restitution: 0.7,
//!     ..Body::default()
//! });
//! let mut sim = Simulation::new(CpuJit, scene, 1);
//! for _ in 0..60 {
//!     sim.step().unwrap();
//! }
//! println!("ball landed at {:?}", sim.positions(0)[ball]);
//! ```

mod contact;
pub mod math;
pub mod mjcf;
mod scene;
mod solver;
#[cfg(test)]
mod tests;

pub use scene::{Body, Joint, Motor, Scene, Shape, WORLD};
pub use solver::State;

use std::sync::Arc;

use crate::dsl::Tensor;
use crate::jit::{Array, Error, Jit, JittedFn};

type StepFn = Box<dyn Fn(&State<Tensor>) -> State<Tensor> + Send + Sync>;

/// A scene bound to a backend: compiles the step function once per state
/// shape and advances `worlds` independent copies of the scene in lockstep.
pub struct Simulation<J: Jit> {
    scene: Arc<Scene>,
    step: JittedFn<J, State<Array>, State<Tensor>, StepFn>,
    state: State<Array>,
}

impl<J: Jit> Simulation<J> {
    pub fn new(jit: J, scene: Scene, worlds: usize) -> Self {
        let scene = Arc::new(scene);
        let state = scene.initial_state(worlds);
        let traced = scene.clone();
        let step: StepFn = Box::new(move |state| traced.trace_step(state));
        Simulation {
            scene,
            step: jit.jit(step),
            state,
        }
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Advance every world by one frame (`scene.dt`).
    pub fn step(&mut self) -> Result<(), Error> {
        self.state = self.step.call(&self.state)?;
        Ok(())
    }

    /// The packed state; mutate through [`State`]'s public arrays to set up
    /// per-world initial conditions.
    pub fn state(&self) -> &State<Array> {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut State<Array> {
        &mut self.state
    }

    /// Body positions in one world, indexed by the ids [`Scene::add`]
    /// returned (row [`WORLD`] is the static world body at the origin).
    pub fn positions(&self, world: usize) -> Vec<[f32; 3]> {
        rows3(&self.state.pos, world, self.scene.bodies.len())
    }

    pub fn velocities(&self, world: usize) -> Vec<[f32; 3]> {
        rows3(&self.state.vel, world, self.scene.bodies.len())
    }

    pub fn angular_velocities(&self, world: usize) -> Vec<[f32; 3]> {
        rows3(&self.state.ang_vel, world, self.scene.bodies.len())
    }

    /// Orientation quaternions `(w, x, y, z)` in one world.
    pub fn orientations(&self, world: usize) -> Vec<[f32; 4]> {
        let n = self.scene.bodies.len();
        self.state.quat.data()[world * n * 4..(world + 1) * n * 4]
            .chunks_exact(4)
            .map(|q| [q[0], q[1], q[2], q[3]])
            .collect()
    }
}

fn rows3(array: &Array, world: usize, n: usize) -> Vec<[f32; 3]> {
    array.data()[world * n * 3..(world + 1) * n * 3]
        .chunks_exact(3)
        .map(|v| [v[0], v[1], v[2]])
        .collect()
}
