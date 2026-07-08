//! Behavioral tests, written as worked examples: each builds a small scene,
//! runs the compiled simulation, and checks physics a reader can verify by
//! hand (or against the discrete update rule the solver actually implements).

use crate::dsl::{Tensor, grad_wrt};
use crate::jit::{Array, CpuJit, Jit};
use crate::physics::{Body, Joint, Scene, Shape, Simulation, WORLD};

fn sphere(radius: f32) -> Shape {
    Shape::Sphere { radius }
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    norm3([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}

/// Graph tracing, lowering, and autodiff walk the expression graph
/// recursively; a rollout of many steps is deep enough to outgrow the 2 MiB
/// default test-thread stack, so the heavier tests run on a roomier one.
fn with_big_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn_scoped(scope, f)
            .expect("spawn test thread")
            .join()
            .expect("test thread panicked")
    })
}

#[test]
fn free_fall_matches_the_discrete_update_rule() {
    // With no constraints the solver is exactly semi-implicit Euler at the
    // substep rate. Mirror that f32 recurrence on the host and expect
    // near-identical numbers.
    let mut scene = Scene::new();
    scene.ground = false;
    scene.add(Body {
        position: [1.0, 10.0, -2.0],
        mass: 2.0,
        ..Body::default()
    });
    let mut sim = Simulation::new(CpuJit, scene.clone(), 1);

    let frames = 30;
    for _ in 0..frames {
        sim.step().expect("step");
    }

    let h = scene.dt / scene.substeps as f32;
    let (mut y, mut vy) = (10.0f32, 0.0f32);
    for _ in 0..frames * scene.substeps {
        vy += scene.gravity[1] * h;
        y += vy * h;
    }

    let ball = sim.positions(0)[1];
    let vel = sim.velocities(0)[1];
    assert!((ball[0] - 1.0).abs() < 1e-4, "x drifted: {}", ball[0]);
    assert!((ball[2] + 2.0).abs() < 1e-4, "z drifted: {}", ball[2]);
    assert!((ball[1] - y).abs() < 1e-4, "y: got {}, want {y}", ball[1]);
    assert!((vel[1] - vy).abs() < 1e-3, "vy: got {}, want {vy}", vel[1]);

    // And the closed-form parabola, loosely (the discretization error).
    let t = scene.dt * frames as f32;
    let exact = 10.0 + 0.5 * scene.gravity[1] * t * t;
    assert!((ball[1] - exact).abs() < 0.05);
}

#[test]
fn sphere_rests_on_the_ground() {
    let mut scene = Scene::new();
    let ball = scene.add(Body {
        shape: sphere(0.5),
        position: [0.0, 0.5, 0.0],
        ..Body::default()
    });
    let mut sim = Simulation::new(CpuJit, scene, 1);
    for _ in 0..60 {
        sim.step().expect("step");
    }
    let pos = sim.positions(0)[ball];
    assert!(
        (pos[1] - 0.5).abs() < 0.02,
        "rest height {} (want ~0.5)",
        pos[1]
    );
    assert!(norm3(sim.velocities(0)[ball]) < 0.05, "still moving");
}

#[test]
fn dropped_sphere_bounces_with_restitution() {
    let mut scene = Scene::new();
    let ball = scene.add(Body {
        shape: sphere(0.25),
        position: [0.0, 2.0, 0.0],
        restitution: 0.8,
        ..Body::default()
    });
    let mut sim = Simulation::new(CpuJit, scene, 1);

    // Watch for the first apex after the ball has bounced back upward.
    let mut prev_y = 2.0f32;
    let mut rising = false;
    let mut apex = None;
    for _ in 0..240 {
        sim.step().expect("step");
        let y = sim.positions(0)[ball][1];
        if !rising && y > prev_y + 1e-5 {
            rising = true;
        }
        if rising && y < prev_y - 1e-5 {
            apex = Some(prev_y);
            break;
        }
        prev_y = y;
    }

    let apex = apex.expect("sphere never bounced");
    // e = 0.8 keeps 64% of the energy: apex ≈ 0.64 · 1.75 + 0.25 ≈ 1.37.
    assert!((0.7..1.8).contains(&apex), "bounce apex {apex} implausible");
}

#[test]
fn pendulum_on_a_static_anchor_keeps_rod_length() {
    let mut scene = Scene::new();
    scene.ground = false;
    scene.substeps = 8;
    // An explicit static body as the anchor (also exercises statics).
    let anchor = scene.add(Body {
        shape: sphere(0.1),
        mass: f32::INFINITY,
        position: [0.0, 5.0, 0.0],
        collides: false,
        ..Body::default()
    });
    let bob = scene.add(Body {
        shape: sphere(0.2),
        position: [1.5, 5.0, 0.0],
        ..Body::default()
    });
    scene.joints.push(Joint::Ball {
        a: anchor,
        b: bob,
        anchor_a: [0.0; 3],
        anchor_b: [-1.5, 0.0, 0.0],
    });
    let mut sim = Simulation::new(CpuJit, scene, 1);

    // Released horizontally, the bob can never move faster than free fall
    // through the rod length allows.
    let max_speed = (2.0 * 9.81 * 1.5f32).sqrt();
    let (mut min_len, mut max_len) = (f32::INFINITY, 0.0f32);
    let mut lowest_x = f32::INFINITY;
    for _ in 0..180 {
        sim.step().expect("step");
        let p = sim.positions(0);
        let len = dist3(p[anchor], p[bob]);
        min_len = min_len.min(len);
        max_len = max_len.max(len);
        lowest_x = lowest_x.min(p[bob][0]);
        assert!(
            norm3(sim.velocities(0)[bob]) < 1.1 * max_speed,
            "pendulum gained energy"
        );
    }

    assert!((min_len - 1.5).abs() < 0.03, "rod shrank to {min_len}");
    assert!((max_len - 1.5).abs() < 0.03, "rod stretched to {max_len}");
    assert!(lowest_x < 0.4, "pendulum barely swung (min x {lowest_x})");
    assert!(
        dist3(sim.positions(0)[anchor], [0.0, 5.0, 0.0]) < 1e-5,
        "anchor moved"
    );
}

#[test]
fn rod_chain_hangs_from_the_world() {
    let mut scene = Scene::new();
    scene.ground = false;
    scene.substeps = 8;
    let link = 0.5f32;
    // A horizontal chain of three rods anchored directly to a point in
    // space (WORLD's local frame is the world frame).
    let mut prev = WORLD;
    let mut prev_anchor = [0.0, 3.0, 0.0];
    let mut ids = vec![];
    for i in 1..=3 {
        let body = scene.add(Body {
            shape: sphere(0.08),
            mass: 0.5,
            position: [link * i as f32, 3.0, 0.0],
            collides: false,
            ..Body::default()
        });
        scene.joints.push(Joint::Distance {
            a: prev,
            b: body,
            anchor_a: prev_anchor,
            anchor_b: [0.0; 3],
            length: link,
        });
        prev = body;
        prev_anchor = [0.0; 3];
        ids.push(body);
    }
    let mut sim = Simulation::new(CpuJit, scene, 1);
    for _ in 0..120 {
        sim.step().expect("step");
    }

    let p = sim.positions(0);
    let lengths = [
        dist3([0.0, 3.0, 0.0], p[ids[0]]),
        dist3(p[ids[0]], p[ids[1]]),
        dist3(p[ids[1]], p[ids[2]]),
    ];
    for (i, len) in lengths.iter().enumerate() {
        assert!(
            (len - link).abs() < 0.03,
            "link {i} length {len} (want {link})"
        );
    }
    assert!(p[ids[2]][1] < 2.5, "chain never fell: {:?}", p[ids[2]]);
}

#[test]
fn tilted_box_falls_and_settles_flat() {
    let mut scene = Scene::new();
    scene.substeps = 8;
    let tilt = 0.15f32;
    let brick = scene.add(Body {
        shape: Shape::Cuboid {
            half_extents: [0.4, 0.3, 0.5],
        },
        mass: 2.0,
        position: [0.0, 1.2, 0.0],
        orientation: [(tilt / 2.0).cos(), 0.0, 0.0, (tilt / 2.0).sin()],
        ..Body::default()
    });
    let mut sim = Simulation::new(CpuJit, scene, 1);
    for _ in 0..240 {
        sim.step().expect("step");
    }

    let pos = sim.positions(0)[brick];
    assert!(pos[1].is_finite(), "exploded");
    // Flat on its y-face the center sits at 0.3; allow having tipped onto
    // another face, but it must rest on the surface, not inside it.
    assert!(
        (0.25..0.55).contains(&pos[1]),
        "not resting on a face: y = {}",
        pos[1]
    );
    assert!(norm3(sim.velocities(0)[brick]) < 0.1, "still sliding");
    assert!(
        norm3(sim.angular_velocities(0)[brick]) < 0.5,
        "still spinning"
    );
}

#[test]
fn equal_spheres_collide_head_on_and_swap_velocities() {
    let mut scene = Scene::new();
    scene.ground = false;
    scene.gravity = [0.0; 3];
    scene.substeps = 8;
    let bouncy = Body {
        shape: sphere(0.5),
        restitution: 1.0,
        friction: 0.0,
        ..Body::default()
    };
    let a = scene.add(Body {
        position: [-1.5, 0.0, 0.0],
        velocity: [2.0, 0.0, 0.0],
        ..bouncy
    });
    let b = scene.add(Body {
        position: [1.5, 0.0, 0.0],
        velocity: [-2.0, 0.0, 0.0],
        ..bouncy
    });
    let mut sim = Simulation::new(CpuJit, scene, 1);
    for _ in 0..90 {
        sim.step().expect("step");
    }

    let (p, v) = (sim.positions(0), sim.velocities(0));
    assert!(dist3(p[a], p[b]) > 1.05, "spheres stuck together");
    // An elastic head-on hit between equal masses swaps the velocities.
    assert!(v[a][0] < -1.0, "A should rebound left: {:?}", v[a]);
    assert!(v[b][0] > 1.0, "B should rebound right: {:?}", v[b]);
    assert!((v[a][0] + v[b][0]).abs() < 0.15, "momentum drifted");
}

#[test]
fn hinged_plank_swings_but_keeps_its_axis() {
    let mut scene = Scene::new();
    scene.ground = false;
    scene.substeps = 8;
    scene.iterations = 6;
    // A plank hinged to a point in space; gravity swings it about z.
    let plank = scene.add(Body {
        shape: Shape::Cuboid {
            half_extents: [0.5, 0.05, 0.2],
        },
        position: [0.6, 2.0, 0.0],
        collides: false,
        ..Body::default()
    });
    scene.joints.push(Joint::Hinge {
        a: WORLD,
        b: plank,
        anchor_a: [0.0, 2.0, 0.0],
        anchor_b: [-0.6, 0.0, 0.0],
        axis_a: [0.0, 0.0, 1.0],
        axis_b: [0.0, 0.0, 1.0],
    });
    let mut sim = Simulation::new(CpuJit, scene, 1);

    let rotate = |q: [f32; 4], v: [f32; 3]| {
        let qv = [q[1], q[2], q[3]];
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let t = cross(qv, v).map(|c| 2.0 * c);
        let u = cross(qv, t);
        [
            v[0] + q[0] * t[0] + u[0],
            v[1] + q[0] * t[1] + u[1],
            v[2] + q[0] * t[2] + u[2],
        ]
    };

    let mut worst_alignment = 1.0f32;
    let mut worst_gap = 0.0f32;
    let mut lowest_x = f32::INFINITY;
    for _ in 0..150 {
        sim.step().expect("step");
        let q = sim.orientations(0)[plank];
        worst_alignment = worst_alignment.min(rotate(q, [0.0, 0.0, 1.0])[2]);
        let p = sim.positions(0)[plank];
        let arm = rotate(q, [-0.6, 0.0, 0.0]);
        let anchor = [p[0] + arm[0], p[1] + arm[1], p[2] + arm[2]];
        worst_gap = worst_gap.max(dist3(anchor, [0.0, 2.0, 0.0]));
        lowest_x = lowest_x.min(p[0]);
    }

    assert!(
        worst_alignment > 0.98,
        "hinge axis wandered: cos = {worst_alignment}"
    );
    assert!(worst_gap < 0.05, "hinge anchor separated by {worst_gap}");
    assert!(lowest_x < 0.35, "plank never swung (min x {lowest_x})");
}

#[test]
fn worlds_evolve_independently() {
    // Two worlds share one compiled step; give them different initial
    // velocities and check world 0 matches a single-world run while world 1
    // does its own thing.
    let mut scene = Scene::new();
    scene.ground = false;
    let ball = scene.add(Body {
        position: [0.0, 5.0, 0.0],
        ..Body::default()
    });

    let mut pair = Simulation::new(CpuJit, scene.clone(), 2);
    let n = pair.scene().bodies.len();
    // vel[world = 1][body = ball].x — packed [W, N, 3], row-major.
    pair.state_mut().vel.data_mut()[(n + ball) * 3] = 3.0;

    let mut solo = Simulation::new(CpuJit, scene, 1);

    for _ in 0..30 {
        pair.step().expect("pair step");
        solo.step().expect("solo step");
    }

    let world0 = pair.positions(0)[ball];
    let world1 = pair.positions(1)[ball];
    let reference = solo.positions(0)[ball];
    assert!(
        dist3(world0, reference) < 1e-5,
        "world 0 diverged from solo run"
    );
    assert!(
        (world1[0] - 1.5).abs() < 1e-3,
        "world 1 ignored its velocity"
    );
    assert!(
        (world0[0]).abs() < 1e-6,
        "world 0 caught world 1's velocity"
    );
    assert!(
        (world0[1] - world1[1]).abs() < 1e-4,
        "same fall height in both"
    );
}

#[test]
fn motor_torque_spins_a_hinged_wheel() {
    // A wheel hinged to the world in zero gravity: constant motor torque
    // must spin it up about the hinge axis at ω ≈ τ t / I.
    let mut scene = Scene::new();
    scene.ground = false;
    scene.gravity = [0.0; 3];
    let wheel = scene.add(Body {
        shape: sphere(0.5),
        position: [0.0, 1.0, 0.0],
        collides: false,
        ..Body::default()
    });
    scene.joints.push(Joint::Hinge {
        a: WORLD,
        b: wheel,
        anchor_a: [0.0, 1.0, 0.0],
        anchor_b: [0.0; 3],
        axis_a: [0.0, 0.0, 1.0],
        axis_b: [0.0, 0.0, 1.0],
    });
    scene.motors.push(crate::physics::Motor {
        joint: 0,
        gear: 2.0,
    });

    // Drive it through a jitted controlled step.
    #[derive(resin_macros::Tree, Clone)]
    struct StepIn<T> {
        state: crate::physics::State<T>,
        ctrl: T,
    }
    let traced = scene.clone();
    let step = CpuJit.jit(move |input: &StepIn<Tensor>| {
        traced.trace_step_driven(&input.state, Some(&input.ctrl))
    });

    let mut state = scene.initial_state(1);
    for _ in 0..60 {
        state = step
            .call(&StepIn {
                state,
                ctrl: Array::from_f32(&[1, 1], &[0.5]),
            })
            .expect("driven step");
    }

    // τ = gear · ctrl = 1.0; I = 2/5 m r² = 0.1; after 1 s, ω ≈ 10 rad/s.
    let w = state.ang_vel.data();
    let wheel_w = [w[wheel * 3], w[wheel * 3 + 1], w[wheel * 3 + 2]];
    assert!(
        (wheel_w[2] - 10.0).abs() < 0.2,
        "expected ~10 rad/s about z, got {wheel_w:?}"
    );
    assert!(
        wheel_w[0].abs() < 0.05 && wheel_w[1].abs() < 0.05,
        "off-axis spin: {wheel_w:?}"
    );
}

#[test]
fn rollout_gradients_match_finite_differences() {
    with_big_stack(|| {
        // Differentiate "where does the ball end up" with respect to "how
        // was it thrown" straight through the compiled physics. The scene
        // keeps the ground plane so the (inactive) contact graph is part of
        // what is differentiated.
        let mut scene = Scene::new();
        scene.substeps = 2;
        scene.iterations = 2;
        scene.add(Body {
            position: [0.0, 4.0, 0.0],
            ..Body::default()
        });
        let steps = 10;
        let target = [1.0f32, 3.5, 0.5];

        let loss_of = |state: &crate::physics::State<Tensor>| {
            let mut rolled = scene.trace_step(state);
            for _ in 1..steps {
                rolled = scene.trace_step(&rolled);
            }
            // Squared distance from the ball (body row 1) to the target:
            // mask out every other row, sum rows away, compare.
            let ball_row = Tensor::constant(&[2, 1], &[0.0, 1.0]);
            let ball = (rolled.pos * ball_row).sum_axes(&[0, 1]).squeeze_all();
            let diff = ball - Tensor::constant(&[3], &target);
            (diff.clone() * diff).sum_axes(&[0]).squeeze_all()
        };

        #[derive(resin_macros::Tree)]
        struct LossAndGrad<T> {
            loss: T,
            /// d loss / d initial velocities, shaped like `State::vel`.
            grad_vel: T,
        }

        let grad_fn = CpuJit.jit(move |state: &crate::physics::State<Tensor>| {
            let loss = loss_of(state);
            let grads = grad_wrt(&loss, state).expect("differentiable rollout");
            LossAndGrad {
                loss,
                grad_vel: grads.vel,
            }
        });

        let base = scene.initial_state(1);
        let out = grad_fn.call(&base).expect("grad rollout");

        // Central finite differences on the ball's initial velocity: the
        // ball is body row 1 in world 0, so its vx sits at flat index 3.
        let ball_x = 3;
        for axis in 0..3 {
            let eval = |sign: f32| {
                let mut state = scene.initial_state(1);
                state.vel.data_mut()[ball_x + axis] += sign * 1e-3;
                grad_fn.call(&state).expect("fd rollout").loss.scalar()
            };
            let want = (eval(1.0) - eval(-1.0)) / 2e-3;
            let got = out.grad_vel.data()[ball_x + axis];
            assert!(
                (got - want).abs() <= 0.05 * want.abs().max(0.1),
                "d loss/d v[{axis}]: autodiff {got} vs finite difference {want}"
            );
        }
    });
}
