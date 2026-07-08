//! Scene description: bodies, joints, and simulation parameters.
//!
//! Everything here is plain host-side data. The solver freezes it into the
//! traced graph as constants, so it is fixed for the life of a compiled
//! step function; only positions, orientations, and velocities are runtime
//! state.

/// Collision shape, in the body's local frame (centered on the center of
/// mass, axis-aligned).
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Sphere { radius: f32 },
    Cuboid { half_extents: [f32; 3] },
}

impl Shape {
    /// Diagonal of the body-frame inertia tensor (both shapes have diagonal
    /// inertia about their own axes).
    pub(crate) fn inertia(&self, mass: f32) -> [f32; 3] {
        match *self {
            Shape::Sphere { radius } => {
                let i = 0.4 * mass * radius * radius;
                [i, i, i]
            }
            Shape::Cuboid {
                half_extents: [hx, hy, hz],
            } => [
                mass / 3.0 * (hy * hy + hz * hz),
                mass / 3.0 * (hx * hx + hz * hz),
                mass / 3.0 * (hx * hx + hy * hy),
            ],
        }
    }

    /// A cuboid's 8 corners in the local frame (its ground-contact points).
    pub(crate) fn corners(&self) -> Vec<[f32; 3]> {
        let Shape::Cuboid {
            half_extents: [hx, hy, hz],
        } = *self
        else {
            return vec![];
        };
        let mut corners = Vec::with_capacity(8);
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    corners.push([sx * hx, sy * hy, sz * hz]);
                }
            }
        }
        corners
    }
}

/// One rigid body. Build with struct-update syntax:
///
/// ```
/// # use resin::physics::{Body, Shape};
/// let ball = Body {
///     shape: Shape::Sphere { radius: 0.5 },
///     position: [0.0, 3.0, 0.0],
///     restitution: 0.7,
///     ..Body::default()
/// };
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub shape: Shape,
    /// Mass in kg. `f32::INFINITY` pins the body in place (a static anchor).
    pub mass: f32,
    pub position: [f32; 3],
    /// Unit quaternion `(w, x, y, z)`.
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    /// World-frame angular velocity, radians/s about each axis.
    pub angular_velocity: [f32; 3],
    /// 0 = no bounce, 1 = perfectly elastic.
    pub restitution: f32,
    /// Coulomb friction coefficient.
    pub friction: f32,
    /// Bodies with `collides: false` pass through everything but still
    /// participate in joints (useful for joint anchors).
    pub collides: bool,
}

impl Default for Body {
    fn default() -> Self {
        Body {
            shape: Shape::Sphere { radius: 0.5 },
            mass: 1.0,
            position: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            restitution: 0.0,
            friction: 0.5,
            collides: true,
        }
    }
}

/// A constraint between two bodies (indices returned by [`Scene::add`]).
/// Anchors and axes are in each body's local frame. Use [`WORLD`] to attach
/// to a fixed point in space: the world body sits at the origin with
/// identity orientation, so its "local" coordinates are world coordinates.
#[derive(Clone, Copy, Debug)]
pub enum Joint {
    /// Holds two anchor points coincident (ball-and-socket).
    Ball {
        a: usize,
        b: usize,
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
    },
    /// Holds two anchor points at a fixed distance (a rigid rod: resists
    /// both stretch and compression).
    Distance {
        a: usize,
        b: usize,
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        length: f32,
    },
    /// Ball joint plus alignment of two local axes: only rotation about the
    /// shared axis remains free (a door hinge).
    Hinge {
        a: usize,
        b: usize,
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        axis_a: [f32; 3],
        axis_b: [f32; 3],
    },
    /// Keeps two local axes parallel without constraining positions — the
    /// angular half of a hinge on its own. A rigid weld is a Ball plus two
    /// AxisAligns for two different axis pairs.
    AxisAlign {
        a: usize,
        b: usize,
        axis_a: [f32; 3],
        axis_b: [f32; 3],
    },
}

/// A torque motor on a hinge joint. Controls enter the compiled step as a
/// `[worlds, motors]` tensor; motor `m` applies torque
/// `gear * ctrl[:, m]` about its hinge axis — on the child body positively,
/// on the parent as the reaction.
#[derive(Clone, Copy, Debug)]
pub struct Motor {
    /// Index into [`Scene::joints`]; must be a [`Joint::Hinge`].
    pub joint: usize,
    pub gear: f32,
}

/// Index of the implicit static "world" body every scene starts with.
///
/// Giving the world a body row means every constraint — including ground
/// contacts and joints anchored in space — is uniformly "between two
/// bodies"; the world's infinite mass makes its side of each correction
/// vanish, with no special cases in the solver.
pub const WORLD: usize = 0;

/// A physics scene plus simulation parameters. Mutate freely, then hand to
/// [`crate::physics::Simulation::new`] (or trace steps directly with
/// [`Scene::trace_step`]); the topology is frozen into each compiled step.
#[derive(Clone)]
pub struct Scene {
    /// `bodies[WORLD]` is the static world body; leave it in place.
    pub bodies: Vec<Body>,
    pub joints: Vec<Joint>,
    pub motors: Vec<Motor>,
    /// Frame timestep in seconds; one `step()` advances this far.
    pub dt: f32,
    /// Physics substeps per frame. The most effective accuracy/stiffness
    /// knob: many cheap substeps beat many solver iterations (Macklin et
    /// al., "Small Steps in Physics Simulation", 2019).
    pub substeps: usize,
    /// Constraint-projection rounds per substep.
    pub iterations: usize,
    pub gravity: [f32; 3],
    /// Whether the ground plane `y = 0` (normal +y) collides.
    pub ground: bool,
}

impl Scene {
    pub fn new() -> Self {
        let world = Body {
            mass: f32::INFINITY,
            collides: false,
            ..Body::default()
        };
        Scene {
            bodies: vec![world],
            joints: vec![],
            motors: vec![],
            dt: 1.0 / 60.0,
            substeps: 4,
            iterations: 4,
            gravity: [0.0, -9.81, 0.0],
            ground: true,
        }
    }

    /// Add a body and return its index (stable; also its row in the packed
    /// state tensors).
    pub fn add(&mut self, body: Body) -> usize {
        self.bodies.push(body);
        self.bodies.len() - 1
    }
}

impl Default for Scene {
    fn default() -> Self {
        Scene::new()
    }
}
