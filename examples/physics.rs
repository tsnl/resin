//! Rigid-body physics demo: a bouncing ball, a swinging pendulum, and a
//! tumbling box, simulated by one compiled tensor program and drawn as
//! ASCII frames. Three worlds run in lockstep in one batch; the ball is
//! thrown differently in each, and the side view shows world 0.
//!
//! Usage:
//!   cargo run --example physics

use resin::jit::CpuJit;
use resin::physics::{Body, Joint, Scene, Shape, Simulation, WORLD};

const WORLDS: usize = 3;

fn main() {
    let mut scene = Scene::new();
    scene.substeps = 8;

    let ball = scene.add(Body {
        shape: Shape::Sphere { radius: 0.4 },
        position: [-3.0, 3.5, 0.0],
        restitution: 0.8,
        ..Body::default()
    });
    let bob = scene.add(Body {
        shape: Shape::Sphere { radius: 0.3 },
        position: [1.6, 4.5, 0.0],
        collides: false,
        ..Body::default()
    });
    scene.joints.push(Joint::Ball {
        a: WORLD,
        b: bob,
        anchor_a: [0.0, 4.5, 0.0], // hinge point in space
        anchor_b: [-1.6, 0.0, 0.0],
    });
    let brick = scene.add(Body {
        shape: Shape::Cuboid {
            half_extents: [0.5, 0.35, 0.4],
        },
        mass: 2.0,
        position: [3.0, 3.0, 0.0],
        orientation: [0.96, 0.0, 0.0, 0.28], // tilted so it tips over
        ..Body::default()
    });

    let mut sim = Simulation::new(CpuJit, scene, WORLDS);

    // Throw the ball differently in each world (state is [worlds, bodies, 3]).
    let bodies = sim.scene().bodies.len();
    for world in 0..WORLDS {
        sim.state_mut().vel.data_mut()[(world * bodies + ball) * 3] = world as f32;
    }

    for frame in 0..=180 {
        if frame % 20 == 0 {
            println!("t = {:.2}s", frame as f32 / 60.0);
            draw(&sim.positions(0), ball, bob, brick);
        }
        sim.step().expect("step");
    }

    println!("where the ball landed in each world (thrown at 0, 1, 2 m/s):");
    for world in 0..WORLDS {
        let p = sim.positions(world)[ball];
        println!("  world {world}: x = {:+.2}  y = {:.2}", p[0], p[1]);
    }
}

/// Side view of the x/y plane: 64 columns over x ∈ [-4, 4], 18 rows over
/// y ∈ [0, 5.2].
fn draw(positions: &[[f32; 3]], ball: usize, bob: usize, brick: usize) {
    const COLS: usize = 64;
    const ROWS: usize = 18;
    let mut grid = [[' '; COLS]; ROWS];

    let mut plot = |p: [f32; 3], glyph: char| {
        let col = ((p[0] + 4.0) / 8.0 * COLS as f32) as isize;
        let row = ROWS as isize - 1 - ((p[1] / 5.2) * ROWS as f32) as isize;
        if (0..COLS as isize).contains(&col) && (0..ROWS as isize).contains(&row) {
            grid[row as usize][col as usize] = glyph;
        }
    };
    plot(positions[ball], 'o');
    plot(positions[bob], '*');
    plot([0.0, 4.5, 0.0], '+'); // the pendulum's anchor point
    plot(positions[brick], '#');

    for row in grid {
        println!("|{}|", row.iter().collect::<String>());
    }
    println!("+{}+", "=".repeat(COLS));
}
