//! Compare resin's simulation of an MJCF model against a MuJoCo CPU
//! reference trajectory, step by step.
//!
//!   python3 scripts/mujoco_reference.py examples/mjcf/pendulum.xml 500 > /tmp/ref.txt
//!   cargo run --example mujoco_compare -- examples/mjcf/pendulum.xml /tmp/ref.txt
//!
//! Prints the position deviation per body over the trajectory. The two
//! engines discretize differently — MuJoCo solves constraints in joint
//! coordinates with soft contacts, in f64; resin projects positions in
//! maximal coordinates, in f32 — so trajectories agree closely but not
//! bitwise, and contact-heavy transients diverge fastest.

use resin::jit::CpuJit;
use resin::physics::{Simulation, mjcf};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(model_path), Some(dump_path)) = (args.next(), args.next()) else {
        eprintln!("usage: mujoco_compare <model.xml> <reference-dump.txt> [substeps] [iterations]");
        eprintln!("(generate the dump with scripts/mujoco_reference.py)");
        std::process::exit(2);
    };

    let mut loaded = mjcf::load_file(&model_path).expect("load MJCF");
    for note in &loaded.notes {
        eprintln!("note: {note}");
    }
    if let Some(substeps) = args.next() {
        loaded.scene.substeps = substeps.parse().expect("substeps");
    }
    if let Some(iterations) = args.next() {
        loaded.scene.iterations = iterations.parse().expect("iterations");
    }

    let dump = std::fs::read_to_string(&dump_path).expect("read reference dump");
    let mut lines = dump.lines();
    let header: Vec<&str> = lines
        .next()
        .expect("dump header")
        .split_whitespace()
        .collect();
    assert_eq!(header[0], "timestep", "not a reference dump");
    let timestep: f32 = header[1].parse().unwrap();
    let names: Vec<&str> = header[4..].to_vec();
    assert!(
        (timestep - loaded.scene.dt).abs() < 1e-9,
        "dump timestep {timestep} != model timestep {}",
        loaded.scene.dt
    );

    let rows: Vec<usize> = names.iter().map(|name| loaded.body_ids[*name]).collect();
    let mut sim = Simulation::new(CpuJit, loaded.scene, 1);

    let mut worst = vec![(0.0f32, 0usize); names.len()];
    let mut last = vec![0.0f32; names.len()];
    let mut steps = 0;
    for (step, line) in lines.enumerate() {
        sim.step().expect("step");
        steps = step + 1;
        let reference: Vec<f32> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let positions = sim.positions(0);
        for (which, &row) in rows.iter().enumerate() {
            // MuJoCo's z-up (x, y, z) is resin's y-up (x, z, -y).
            let mj = &reference[which * 7..which * 7 + 3];
            let want = [mj[0], mj[2], -mj[1]];
            let got = positions[row];
            let deviation = ((got[0] - want[0]).powi(2)
                + (got[1] - want[1]).powi(2)
                + (got[2] - want[2]).powi(2))
            .sqrt();
            if deviation > worst[which].0 {
                worst[which] = (deviation, step);
            }
            last[which] = deviation;
        }
    }

    println!("{steps} steps at dt = {timestep}");
    for (which, name) in names.iter().enumerate() {
        let (deviation, step) = worst[which];
        println!(
            "  {name}: worst position deviation {deviation:.6} m (step {step}), final {:.6} m",
            last[which]
        );
    }
}
