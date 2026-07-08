//! Contacts: candidate enumeration, geometry, and velocity-level response.
//!
//! There is no collision *detection* pass. Every place two things could
//! possibly touch — each sphere or cuboid corner against the ground, each
//! sphere against each other sphere — becomes a permanent *candidate* at
//! trace time, and its penetration depth is `relu`-clamped so separated
//! candidates contribute exactly zero to the solve. That is O(n²) where a
//! real engine's broadphase would prune, but it needs no branches and no
//! dynamic topology, which is what a fixed tensor graph can express.

use crate::dsl::Tensor;
use crate::physics::math::{EPS, Vec3s, gate, host_column, min};
use crate::physics::scene::{Scene, Shape, WORLD};
use crate::physics::solver::{
    BodyTable, Cols, Corrections, Picked, apply_to_side, constraint_mass, pick,
};

/// One batch of contact candidates between sides A and B, with the geometry
/// evaluated at the current positions. B is the world body for ground
/// contacts. `normal` points from B toward A; `arm_*` reach from each
/// center of mass to the contact point; `depth >= 0` is the penetration
/// (zero when separated, thanks to relu).
pub(super) struct Contacts {
    pub a: Picked,
    pub b: Picked,
    pub arm_a: Vec3s,
    pub arm_b: Vec3s,
    pub normal: Vec3s,
    pub depth: Tensor,
    restitution: Tensor,
    friction: Tensor,
}

impl Scene {
    /// All contact batches for the current positions: at most one batch of
    /// ground candidates and one of sphere-sphere pairs. Rebuilt (cheaply,
    /// at trace time) whenever positions have moved.
    pub(super) fn contact_batches(&self, cols: &Cols, table: &BodyTable) -> Vec<Contacts> {
        let mut batches = Vec::new();
        if let Some(ground) = self.ground_contacts(cols, table) {
            batches.push(ground);
        }
        if let Some(spheres) = self.sphere_contacts(cols, table) {
            batches.push(spheres);
        }
        batches
    }

    /// Ground candidates: a sphere touches the plane at the point one radius
    /// below its center; a cuboid at whichever of its 8 corners are lowest —
    /// so all 8 are candidates and relu picks the penetrating ones.
    fn ground_contacts(&self, cols: &Cols, table: &BodyTable) -> Option<Contacts> {
        if !self.ground {
            return None;
        }
        let mut rows = Vec::new();
        let mut points = Vec::new(); // body-local candidate points
        let mut drop = Vec::new(); // extra reach along -normal (sphere radius)
        let mut restitution = Vec::new();
        let mut friction = Vec::new();
        for (row, body) in self.bodies.iter().enumerate() {
            if !body.collides || body.mass.is_infinite() {
                continue;
            }
            let candidates = match body.shape {
                Shape::Sphere { radius } => vec![([0.0; 3], radius)],
                Shape::Cuboid { .. } => {
                    body.shape.corners().into_iter().map(|c| (c, 0.0)).collect()
                }
            };
            for (point, radius) in candidates {
                rows.push(row);
                points.push(point);
                drop.push(radius);
                restitution.push(body.restitution);
                friction.push(body.friction);
            }
        }
        if rows.is_empty() {
            return None;
        }

        let n = self.bodies.len();
        let a = pick(cols, table, &rows, n);
        let b = pick(cols, table, &vec![WORLD; rows.len()], n);
        let normal = Vec3s::from_host(&[[0.0, 1.0, 0.0]]);
        // Contact point = center + rotated local point − radius · normal.
        let reach: Vec<[f32; 3]> = drop.iter().map(|&r| [0.0, -r, 0.0]).collect();
        let arm_a =
            a.q.rotate(&Vec3s::from_host(&points))
                .add(&Vec3s::from_host(&reach));
        let point = a.x.add(&arm_a);
        // The world's arm is the contact point itself (world sits at the
        // origin); its infinite mass ignores it, but the geometry is honest.
        let arm_b = point.clone();
        let depth = (-point.dot(&normal)).relu();
        Some(Contacts {
            a,
            b,
            arm_a,
            arm_b,
            normal,
            depth,
            restitution: host_column(&restitution),
            friction: host_column(&friction),
        })
    }

    /// Every unordered pair of collidable spheres (with at least one side
    /// dynamic) is a candidate. Cuboid-cuboid and cuboid-sphere collision
    /// need contact manifolds and are out of scope.
    fn sphere_contacts(&self, cols: &Cols, table: &BodyTable) -> Option<Contacts> {
        let mut rows_a = Vec::new();
        let mut rows_b = Vec::new();
        let mut radius_a = Vec::new();
        let mut radius_b = Vec::new();
        let mut restitution = Vec::new();
        let mut friction = Vec::new();
        let spheres: Vec<(usize, f32)> = self
            .bodies
            .iter()
            .enumerate()
            .filter(|(_, body)| body.collides)
            .filter_map(|(row, body)| match body.shape {
                Shape::Sphere { radius } => Some((row, radius)),
                Shape::Cuboid { .. } => None,
            })
            .collect();
        for (i, &(row_a, ra)) in spheres.iter().enumerate() {
            for &(row_b, rb) in &spheres[i + 1..] {
                if self.bodies[row_a].mass.is_infinite() && self.bodies[row_b].mass.is_infinite() {
                    continue;
                }
                rows_a.push(row_a);
                rows_b.push(row_b);
                radius_a.push(ra);
                radius_b.push(rb);
                restitution.push(
                    self.bodies[row_a]
                        .restitution
                        .max(self.bodies[row_b].restitution),
                );
                friction.push((self.bodies[row_a].friction * self.bodies[row_b].friction).sqrt());
            }
        }
        if rows_a.is_empty() {
            return None;
        }

        let n = self.bodies.len();
        let a = pick(cols, table, &rows_a, n);
        let b = pick(cols, table, &rows_b, n);
        let delta = a.x.sub(&b.x);
        let normal = delta.normalized();
        let ra = host_column(&radius_a);
        let rb = host_column(&radius_b);
        let depth = (ra.clone() + rb.clone() - delta.length()).relu();
        // Contact points sit one radius from each center along the normal.
        let arm_a = normal.scale(&ra).neg();
        let arm_b = normal.scale(&rb);
        Some(Contacts {
            a,
            b,
            arm_a,
            arm_b,
            normal,
            depth,
            restitution: host_column(&restitution),
            friction: host_column(&friction),
        })
    }
}

impl Contacts {
    /// Velocity of A's contact point relative to B's.
    fn relative_velocity(&self, cols: &Cols) -> Vec3s {
        let at_a = cols
            .v
            .gather(&self.a.select)
            .add(&cols.w.gather(&self.a.select).cross(&self.arm_a));
        let at_b = cols
            .v
            .gather(&self.b.select)
            .add(&cols.w.gather(&self.b.select).cross(&self.arm_b));
        at_a.sub(&at_b)
    }

    /// Contact-point approach speed along the normal (negative = closing).
    pub(super) fn normal_speed(&self, cols: &Cols) -> Tensor {
        self.relative_velocity(cols).dot(&self.normal)
    }

    /// The velocity pass (Müller et al. 2020, §3.6): after positions are
    /// projected, contacts still carry the *velocity* the projection implied.
    /// Two fixes, gated by whether the contact pushed during projection
    /// (`lambda > 0`):
    ///
    /// - **restitution** — replace the outgoing normal speed with
    ///   `e ·` (the approach speed before the solve), so a ball with e = 0.8
    ///   leaves with 80% of the speed it arrived; speeds below `2·|g|·h` are
    ///   swallowed to keep resting contacts from jittering forever;
    /// - **friction** — oppose the tangential sliding speed, but a Coulomb
    ///   cone caps the change at `μ · λ / h` (can't rub harder than you
    ///   press).
    ///
    /// The desired velocity change becomes an impulse via the same
    /// generalized masses as `project`, split across both bodies.
    pub(super) fn bounce_and_rub(
        &self,
        cols: &Cols,
        pre_speed: &Tensor,
        lambda: &Tensor,
        h: f32,
        gravity: [f32; 3],
        out: &mut Corrections,
    ) {
        let pushed = gate(lambda);

        let rel = self.relative_velocity(cols);
        let speed_n = rel.dot(&self.normal);
        let slide = rel.sub(&self.normal.scale(&speed_n));
        let slide_speed = slide.length();

        let g =
            (gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]).sqrt();
        let approach = (-pre_speed.clone()).relu();
        let worth_bouncing = gate(&(approach.clone() - Tensor::scalar(2.0 * g * h)).relu());
        let target = self.restitution.clone() * approach * worth_bouncing;
        let fix_n = (target - speed_n) * pushed.clone();

        let cap = self.friction.clone() * lambda.clone() / Tensor::scalar(h);
        let fix_t = min(&cap, &slide_speed) * pushed;

        let want = self
            .normal
            .scale(&fix_n)
            .sub(&slide.normalized().scale(&fix_t));
        let dir = want.normalized();
        let total_mass = constraint_mass(&self.a, &self.arm_a, &dir)
            + constraint_mass(&self.b, &self.arm_b, &dir);
        let impulse = want.scale(&(Tensor::scalar(1.0) / (total_mass + Tensor::scalar(EPS))));

        let active = gate(&want.length());
        apply_to_side(&self.a, &self.arm_a, &impulse, &active, out);
        apply_to_side(&self.b, &self.arm_b, &impulse.neg(), &active, out);
    }
}
