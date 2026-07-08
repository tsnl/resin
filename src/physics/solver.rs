//! The simulation step: position-based rigid-body dynamics as a tensor graph.
//!
//! One [`Scene::trace_step`] call builds a whole frame as a pure
//! `State -> State` graph. Per substep:
//!
//! 1. **integrate** — apply gravity, advance positions and orientations;
//! 2. **project** — iteratively nudge positions/orientations until the
//!    constraints (joints, non-penetration) hold;
//! 3. **reconstruct** — the new velocity is simply how far each body moved,
//!    divided by the substep time;
//! 4. **contact impulses** — velocity-level fixes at contacts: restitution
//!    (bounce) and Coulomb friction.
//!
//! Working on positions instead of forces is what makes position-based
//! dynamics simple and robust: a violated constraint says "these points are
//! this far from where they should be", and the fix is to move them —
//! weighted by *generalized inverse mass*, so heavy (or unmovable) bodies
//! move less. The method is XPBD, after Müller et al., "Detailed Rigid Body
//! Simulation with Extended Position Based Dynamics" (2020).
//!
//! Three tricks make this a fixed tensor graph with no branches, no
//! indexing, and a topology frozen at trace time:
//!
//! - each constraint gathers its two bodies' rows with constant one-hot
//!   matmuls and scatter-adds its corrections back the same way;
//! - `relu` clamps make inactive contact candidates contribute exactly zero,
//!   replacing collision detection's branching;
//! - all constraints in a round are solved simultaneously (Jacobi), and each
//!   body divides by the number of corrections aimed at it, which keeps
//!   simultaneous contacts (e.g. a box on four corners) from over-shooting.

use resin_macros::Tree;

use crate::dsl::Tensor;
use crate::jit::Array;
use crate::physics::math::{EPS, Quats, Vec3s, gate, host_column, one_hot_rows};
use crate::physics::scene::{Joint, Scene, WORLD};

/// Dynamic state of all bodies in all worlds, as packed row-major tensors
/// with a leading worlds axis: `pos [W, N, 3]`, `quat [W, N, 4]` unit
/// `(w, x, y, z)`, `vel [W, N, 3]`, `ang_vel [W, N, 3]` (world frame).
/// Row [`WORLD`] of every world is the inert static-world body. The input
/// and output tree of a compiled step.
#[derive(Tree, Clone)]
pub struct State<T> {
    pub pos: T,
    pub quat: T,
    pub vel: T,
    pub ang_vel: T,
}

/// The state unpacked into per-component columns the math layer works on.
pub(super) struct Cols {
    pub x: Vec3s,
    pub q: Quats,
    pub v: Vec3s,
    pub w: Vec3s,
}

impl Cols {
    fn unpack(state: &State<Tensor>) -> Cols {
        Cols {
            x: Vec3s::from_rows(&state.pos),
            q: Quats::from_rows(&state.quat),
            v: Vec3s::from_rows(&state.vel),
            w: Vec3s::from_rows(&state.ang_vel),
        }
    }

    fn pack(&self) -> State<Tensor> {
        State {
            pos: self.x.to_rows(),
            quat: self.q.to_rows(),
            vel: self.v.to_rows(),
            ang_vel: self.w.to_rows(),
        }
    }
}

/// Per-body mass properties as constant `[N, 1]` columns. Static bodies get
/// zero inverse mass and inertia, which silently zeroes their side of every
/// correction below.
pub(super) struct BodyTable {
    pub inv_mass: Tensor,
    pub inv_inertia: Vec3s,
    pub inertia: Vec3s,
    /// 1 for dynamic bodies, 0 for static — gates gravity.
    movable: Tensor,
}

impl BodyTable {
    fn new(scene: &Scene) -> BodyTable {
        let mut inv_mass = Vec::new();
        let mut inertia = Vec::new();
        let mut inv_inertia = Vec::new();
        let mut movable = Vec::new();
        for body in &scene.bodies {
            assert!(
                body.mass > 0.0,
                "mass must be positive (INFINITY for static)"
            );
            if body.mass.is_finite() {
                let i = body.shape.inertia(body.mass);
                inv_mass.push(1.0 / body.mass);
                inertia.push(i);
                inv_inertia.push([1.0 / i[0], 1.0 / i[1], 1.0 / i[2]]);
                movable.push(1.0);
            } else {
                inv_mass.push(0.0);
                inertia.push([0.0; 3]);
                inv_inertia.push([0.0; 3]);
                movable.push(0.0);
            }
        }
        BodyTable {
            inv_mass: host_column(&inv_mass),
            inv_inertia: Vec3s::from_host(&inv_inertia),
            inertia: Vec3s::from_host(&inertia),
            movable: host_column(&movable),
        }
    }
}

/// The rows of body state one side of a constraint batch touches, gathered
/// with a one-hot matrix (kept for scattering corrections back).
pub(super) struct Picked {
    pub select: Tensor,
    pub x: Vec3s,
    pub q: Quats,
    pub inv_mass: Tensor,
    pub inv_inertia: Vec3s,
}

pub(super) fn pick(cols: &Cols, table: &BodyTable, rows: &[usize], n: usize) -> Picked {
    let select = one_hot_rows(rows, n);
    Picked {
        x: cols.x.gather(&select),
        q: cols.q.gather(&select),
        inv_mass: select.matmul(&table.inv_mass),
        inv_inertia: table.inv_inertia.gather(&select),
        select,
    }
}

/// Apply a world-frame inverse inertia tensor: rotate into the body frame,
/// scale by the diagonal, rotate back — `I⁻¹ v = R (I_body⁻¹ ∘ Rᵀ v)`.
pub(super) fn inv_inertia_times(q: &Quats, inv_inertia: &Vec3s, v: &Vec3s) -> Vec3s {
    q.rotate(&inv_inertia.mul_each(&q.rotate_inv(v)))
}

/// How much one body yields to a unit impulse along `dir` applied at `arm`
/// from its center of mass: `1/m + (arm × dir) · I⁻¹ (arm × dir)`.
/// Zero for static bodies.
pub(super) fn constraint_mass(side: &Picked, arm: &Vec3s, dir: &Vec3s) -> Tensor {
    let lever = arm.cross(dir);
    side.inv_mass.clone() + lever.dot(&inv_inertia_times(&side.q, &side.inv_inertia, &lever))
}

/// Corrections accumulated per body over one solver round. Used twice with
/// different units: position/orientation nudges in the projection rounds,
/// velocity changes in the contact-impulse pass.
pub(super) struct Corrections {
    pub linear: Vec3s,
    pub angular: Vec3s,
    /// How many active constraints touched each body (for averaging).
    pub count: Tensor,
}

impl Corrections {
    fn zeros(bodies: usize, worlds: usize) -> Corrections {
        Corrections {
            linear: Vec3s::zeros(bodies, worlds),
            angular: Vec3s::zeros(bodies, worlds),
            count: Tensor::zeros(&[bodies, worlds]),
        }
    }

    /// Divide by the number of corrections per body — `max(count, 1)`, so a
    /// lone constraint applies fully (Jacobi averaging).
    fn averaged(&self) -> (Vec3s, Vec3s) {
        let scale = Tensor::scalar(1.0)
            / ((self.count.clone() - Tensor::scalar(1.0)).relu() + Tensor::scalar(1.0));
        (self.linear.scale(&scale), self.angular.scale(&scale))
    }
}

/// Credit one side of a constraint batch with an impulse applied at `arm`:
/// the body moves by `impulse / m` and turns by `I⁻¹ (arm × impulse)`.
pub(super) fn apply_to_side(
    side: &Picked,
    arm: &Vec3s,
    impulse: &Vec3s,
    active: &Tensor,
    out: &mut Corrections,
) {
    let linear = impulse.scale(&side.inv_mass);
    let angular = inv_inertia_times(&side.q, &side.inv_inertia, &arm.cross(impulse));
    out.linear = out.linear.add(&linear.scatter_add(&side.select));
    out.angular = out.angular.add(&angular.scatter_add(&side.select));
    out.count = out.count.clone() + side.select.transpose().matmul(active);
}

/// The heart of XPBD: shrink a (signed) constraint `error` by moving side A
/// along `+dir` and side B along `-dir`, each in proportion to its
/// generalized inverse mass. Returns the impulse magnitude Δλ.
pub(super) fn project(
    a: &Picked,
    arm_a: &Vec3s,
    b: &Picked,
    arm_b: &Vec3s,
    dir: &Vec3s,
    error: &Tensor,
    out: &mut Corrections,
) -> Tensor {
    let total_mass = constraint_mass(a, arm_a, dir) + constraint_mass(b, arm_b, dir);
    let dlambda = error.clone() / (total_mass + Tensor::scalar(EPS));
    let impulse = dir.scale(&dlambda);
    let active = gate(&error.abs());
    apply_to_side(a, arm_a, &impulse, &active, out);
    apply_to_side(b, arm_b, &impulse.neg(), &active, out);
    dlambda
}

impl Scene {
    /// Initial packed state: every world starts as a copy of the body
    /// descriptions (perturb per world through the returned arrays).
    pub fn initial_state(&self, worlds: usize) -> State<Array> {
        assert!(worlds >= 1, "need at least one world");
        let n = self.bodies.len();
        let mut pos = Vec::with_capacity(3 * n);
        let mut quat = Vec::with_capacity(4 * n);
        let mut vel = Vec::with_capacity(3 * n);
        let mut ang_vel = Vec::with_capacity(3 * n);
        for body in &self.bodies {
            pos.extend(body.position);
            quat.extend(unit4(body.orientation));
            vel.extend(body.velocity);
            ang_vel.extend(body.angular_velocity);
        }
        let tile = |per_world: &[f32], dim: usize| {
            Array::from_f32(&[worlds, n, dim], &per_world.repeat(worlds))
        };
        State {
            pos: tile(&pos, 3),
            quat: tile(&quat, 4),
            vel: tile(&vel, 3),
            ang_vel: tile(&ang_vel, 3),
        }
    }

    /// Trace one frame (`self.dt`) with all motors idle.
    pub fn trace_step(&self, state: &State<Tensor>) -> State<Tensor> {
        self.trace_step_driven(state, None)
    }

    /// Trace one frame (`self.dt`) as a pure tensor graph, with motor
    /// controls (`[worlds, motors]`, or `None` for no torque).
    ///
    /// This is the composable core: [`crate::physics::Simulation`] jits it
    /// directly, and because every op is differentiable, several steps can be
    /// chained inside a larger traced function — e.g. a rollout with a loss
    /// on the final state, differentiated back to the initial conditions or
    /// the controls.
    pub fn trace_step_driven(&self, state: &State<Tensor>, ctrl: Option<&Tensor>) -> State<Tensor> {
        let n = self.bodies.len();
        let world = &self.bodies[WORLD];
        assert!(
            world.mass.is_infinite() && !world.collides,
            "bodies[WORLD] must stay the static world body"
        );
        assert_eq!(state.pos.shape()[1..], [n, 3], "state does not match scene");
        if let Some(ctrl) = ctrl {
            assert_eq!(
                ctrl.shape()[1..],
                [self.motors.len()],
                "ctrl must be [worlds, motors]"
            );
        }

        let table = BodyTable::new(self);
        let h = self.dt / self.substeps as f32;
        let mut cols = Cols::unpack(state);
        for _ in 0..self.substeps {
            cols = self.substep(cols, &table, h, ctrl);
        }
        cols.pack()
    }

    fn substep(&self, mut cols: Cols, table: &BodyTable, h: f32, ctrl: Option<&Tensor>) -> Cols {
        let n = self.bodies.len();
        let worlds = cols.x.x.shape()[1];
        let hs = Tensor::scalar(h);

        // 1. Integrate (semi-implicit Euler). Gravity only moves movable
        // bodies; everything else already scales by inverse mass.
        let g = Vec3s::from_host(&[self.gravity]);
        cols.v = cols.v.add(&g.scale(&hs).scale(&table.movable));
        if let Some(ctrl) = ctrl {
            self.apply_motors(&mut cols, table, ctrl, &hs);
        }
        cols.x = cols.x.add(&cols.v.scale(&hs));

        // Gyroscopic torque: without it, spinning bodies with unequal
        // inertia axes (a tumbling box) precess wrongly.
        // ω += h · I⁻¹(−ω × (I ω)).
        let momentum = cols
            .q
            .rotate(&table.inertia.mul_each(&cols.q.rotate_inv(&cols.w)));
        let gyro = cols.w.cross(&momentum).neg();
        cols.w = cols
            .w
            .add(&inv_inertia_times(&cols.q, &table.inv_inertia, &gyro).scale(&hs));

        cols.q = cols.q.rotated_by(&cols.w.scale(&hs));

        // Contact approach speeds right after integration: the restitution
        // pass below needs to know how fast each contact was closing
        // *before* projection swallowed the overlap.
        let pre_speeds: Vec<Tensor> = self
            .contact_batches(&cols, table)
            .iter()
            .map(|c| c.normal_speed(&cols))
            .collect();

        // 2. Project constraints. Contact geometry is rebuilt every round
        // from the freshly nudged positions; λ accumulates each contact's
        // total corrective push for the velocity pass below, and the summed
        // nudges feed the velocity update.
        let mut lambdas: Vec<Tensor> = self
            .contact_batches(&cols, table)
            .iter()
            .map(|c| c.depth.zeros_like())
            .collect();
        let mut moved = Vec3s::zeros(n, worlds);
        let mut turned = Vec3s::zeros(n, worlds);
        for _ in 0..self.iterations {
            let mut fix = Corrections::zeros(n, worlds);
            for (contact, lambda) in self.contact_batches(&cols, table).iter().zip(&mut lambdas) {
                *lambda = lambda.clone()
                    + project(
                        &contact.a,
                        &contact.arm_a,
                        &contact.b,
                        &contact.arm_b,
                        &contact.normal,
                        &contact.depth,
                        &mut fix,
                    );
            }
            self.project_joints(&cols, table, &mut fix);

            let (dx, dtheta) = fix.averaged();
            cols.x = cols.x.add(&dx);
            cols.q = cols.q.rotated_by(&dtheta);
            moved = moved.add(&dx);
            turned = turned.add(&dtheta);
        }

        // 3. Position-based velocity update: what projection moved, over
        // time. Equal (to first order) to the textbook (x - x_prev)/h and
        // q ⊗ q_prev⁻¹ reconstruction, but built from the small corrections
        // themselves — subtracting nearly-equal positions in f32 would bleed
        // noise into the velocities of unconstrained bodies.
        cols.v = cols.v.add(&moved.scale(&Tensor::scalar(1.0 / h)));
        cols.w = cols.w.add(&turned.scale(&Tensor::scalar(1.0 / h)));

        // 4. Contact impulses: restitution and friction, gated by which
        // contacts actually pushed during projection (λ > 0).
        let batches = self.contact_batches(&cols, table);
        if !batches.is_empty() {
            let mut kick = Corrections::zeros(n, worlds);
            for ((contact, pre), lambda) in batches.iter().zip(&pre_speeds).zip(&lambdas) {
                contact.bounce_and_rub(&cols, pre, lambda, h, self.gravity, &mut kick);
            }
            let (dv, dw) = kick.averaged();
            cols.v = cols.v.add(&dv);
            cols.w = cols.w.add(&dw);
        }

        cols
    }

    /// Torque motors: motor `m` twists its hinge's child body about the
    /// world-frame hinge axis by `gear * ctrl[:, m]`, and the parent by the
    /// reaction. Torques integrate into angular velocity like gravity does
    /// into linear velocity: ω += h · I⁻¹ τ.
    fn apply_motors(&self, cols: &mut Cols, table: &BodyTable, ctrl: &Tensor, hs: &Tensor) {
        let n = self.bodies.len();
        let motors = self.motors.len();
        for (m, motor) in self.motors.iter().enumerate() {
            let Joint::Hinge { a, b, axis_b, .. } = self.joints[motor.joint] else {
                panic!("motor {m} does not target a hinge joint");
            };
            // Column m of ctrl as a [1, worlds] row, like a gathered scalar.
            let mut mask = vec![0.0; motors];
            mask[m] = motor.gear;
            let torque = (ctrl.clone() * Tensor::constant(&[motors], &mask))
                .sum_axes(&[1])
                .transpose();

            let side_a = pick(cols, table, &[a], n);
            let side_b = pick(cols, table, &[b], n);
            let axis = side_b.q.rotate(&Vec3s::from_host(&[unit3(axis_b)]));
            let twist = axis.scale(&(torque * hs.clone()));
            let dw_b = inv_inertia_times(&side_b.q, &side_b.inv_inertia, &twist);
            let dw_a = inv_inertia_times(&side_a.q, &side_a.inv_inertia, &twist).neg();
            cols.w = cols.w.add(&dw_b.scatter_add(&side_b.select));
            cols.w = cols.w.add(&dw_a.scatter_add(&side_a.select));
        }
    }

    /// Project every joint. Joints are traced one at a time (batches of 1):
    /// scenes have few joints, and plain beats clever.
    fn project_joints(&self, cols: &Cols, table: &BodyTable, fix: &mut Corrections) {
        for joint in &self.joints {
            match *joint {
                Joint::Ball {
                    a,
                    b,
                    anchor_a,
                    anchor_b,
                } => {
                    let (side_a, arm_a) = self.anchored(cols, table, a, anchor_a);
                    let (side_b, arm_b) = self.anchored(cols, table, b, anchor_b);
                    let delta = side_b.x.add(&arm_b).sub(&side_a.x.add(&arm_a));
                    // Positive error along `delta` pulls the anchors together.
                    project(
                        &side_a,
                        &arm_a,
                        &side_b,
                        &arm_b,
                        &delta.normalized(),
                        &delta.length(),
                        fix,
                    );
                }
                Joint::Distance {
                    a,
                    b,
                    anchor_a,
                    anchor_b,
                    length,
                } => {
                    let (side_a, arm_a) = self.anchored(cols, table, a, anchor_a);
                    let (side_b, arm_b) = self.anchored(cols, table, b, anchor_b);
                    let delta = side_b.x.add(&arm_b).sub(&side_a.x.add(&arm_a));
                    // Signed: positive when stretched (pulls together),
                    // negative when compressed (pushes apart).
                    let error = delta.length() - Tensor::scalar(length);
                    project(
                        &side_a,
                        &arm_a,
                        &side_b,
                        &arm_b,
                        &delta.normalized(),
                        &error,
                        fix,
                    );
                }
                Joint::Hinge {
                    a,
                    b,
                    anchor_a,
                    anchor_b,
                    axis_a,
                    axis_b,
                } => {
                    let (side_a, arm_a) = self.anchored(cols, table, a, anchor_a);
                    let (side_b, arm_b) = self.anchored(cols, table, b, anchor_b);
                    let delta = side_b.x.add(&arm_b).sub(&side_a.x.add(&arm_a));
                    project(
                        &side_a,
                        &arm_a,
                        &side_b,
                        &arm_b,
                        &delta.normalized(),
                        &delta.length(),
                        fix,
                    );
                    self.align_axes(&side_a, unit3(axis_a), &side_b, unit3(axis_b), fix);
                }
                Joint::AxisAlign {
                    a,
                    b,
                    axis_a,
                    axis_b,
                } => {
                    let side_a = pick(cols, table, &[a], self.bodies.len());
                    let side_b = pick(cols, table, &[b], self.bodies.len());
                    self.align_axes(&side_a, unit3(axis_a), &side_b, unit3(axis_b), fix);
                }
            }
        }
    }

    /// One side of a joint: its gathered body row and the world-frame arm
    /// from the center of mass to the (rotated) local anchor.
    fn anchored(
        &self,
        cols: &Cols,
        table: &BodyTable,
        body: usize,
        anchor: [f32; 3],
    ) -> (Picked, Vec3s) {
        let side = pick(cols, table, &[body], self.bodies.len());
        let arm = side.q.rotate(&Vec3s::from_host(&[anchor]));
        (side, arm)
    }

    /// The angular half of a hinge: rotate both bodies so their axes become
    /// parallel. For axes `p` and `q`, the correction rotation vector is
    /// `p × q` (its length ≈ the angle between them, exact enough for the
    /// small errors a solver round sees).
    fn align_axes(
        &self,
        side_a: &Picked,
        axis_a: [f32; 3],
        side_b: &Picked,
        axis_b: [f32; 3],
        fix: &mut Corrections,
    ) {
        let world_a = side_a.q.rotate(&Vec3s::from_host(&[axis_a]));
        let world_b = side_b.q.rotate(&Vec3s::from_host(&[axis_b]));
        let cross = world_a.cross(&world_b);
        let angle = cross.length();
        let dir = cross.normalized();

        // The angular analogue of `project`: no arms, inertia only.
        let mass_a = dir.dot(&inv_inertia_times(&side_a.q, &side_a.inv_inertia, &dir));
        let mass_b = dir.dot(&inv_inertia_times(&side_b.q, &side_b.inv_inertia, &dir));
        let dlambda = angle.clone() / (mass_a + mass_b + Tensor::scalar(EPS));
        let impulse = dir.scale(&dlambda);
        let active = gate(&angle);

        let turn_a = inv_inertia_times(&side_a.q, &side_a.inv_inertia, &impulse);
        let turn_b = inv_inertia_times(&side_b.q, &side_b.inv_inertia, &impulse).neg();
        fix.angular = fix.angular.add(&turn_a.scatter_add(&side_a.select));
        fix.count = fix.count.clone() + side_a.select.transpose().matmul(&active);
        fix.angular = fix.angular.add(&turn_b.scatter_add(&side_b.select));
        fix.count = fix.count.clone() + side_b.select.transpose().matmul(&active);
    }
}

fn unit3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    assert!(len > 1e-6, "axis must be non-zero");
    [v[0] / len, v[1] / len, v[2] / len]
}

fn unit4(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    assert!(len > 1e-6, "orientation quaternion must be non-zero");
    [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}
