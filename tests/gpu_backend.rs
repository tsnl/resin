#![cfg(feature = "gpu")]
#[path = "support/toolchain.rs"]
mod config;

use resin::backend::glsl::{self, Stage};
use resin::toolchain::TempDir;
use resin_runtime::{
    ResinGpu, ResinMemory, ResinStatus, image_read_png, image_write_png, testing::lock_gpu,
};
use std::{path::Path, process::Command};

#[path = "support/shaders.rs"]
mod shaders;
mod support;

fn gpu() -> Option<ResinGpu> {
    match ResinGpu::create() {
        Ok(gpu) => Some(gpu),
        Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
            assert!(
                std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "a suitable Vulkan device is required"
            );
            eprintln!("skipping: no suitable Vulkan device");
            None
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    }
}

#[test]
fn typed_device_buffers_match_host_layout_and_preserve_bounds() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let module = support::module(
        r#"export { kernel, main };

        struct Data { marker: uint, wide: ulong, amount: float32 };
        struct Payload { tag: uint, data: Data, end: uint };
        struct Params { count: uint, values: Ptr<Payload>, tail: float32 };
        def at (values: Ptr<Payload>, index: uint) -> Ptr<Payload> = { (Span<Payload> { data = values, length = ulong(67) })(index) };
        def bump (p: Ptr<Payload>, index: uint) -> () = {
            var old = p.*;
            p.* := Payload {
                tag = old.tag + uint (1),
                data = Data { marker = index, wide = old.data.wide + ulong (4294967297), amount = old.data.amount + float32 (0.5) },
                end = old.end + uint (2)
            };
        };
        def kernel (index: uint, root: Ptr<Params>) -> () = {
            if (index < root.count) {
                var p = at(root.values, index);
                bump(p, index)
            } else { () }
        };
        def main () -> int = {
            var value = Payload { tag = uint (10), data = Data { marker = uint (99), wide = ulong (7), amount = float32 (1.25) }, end = uint (20) };
            var root = Params { count = uint (1), values = &value, tail = float32 (0.75) };
            kernel(uint (0), &root);
            kernel(uint (1), &root);
            if (value.tag == uint (11) && value.data.marker == uint (0) && value.data.wide == ulong (4294967304) && value.data.amount == float32 (1.75) && value.end == uint (22) && root.tail == float32 (0.75)) { 0 } else { 1 }
        };
    "#,
    );
    let c = resin::backend::c::emit(&module, "main").unwrap();
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("host-layout{}", std::env::consts::EXE_SUFFIX));
    let cc = std::env::var_os("CC").unwrap_or_else(|| resin::toolchain::DEFAULT_C_COMPILER.into());
    resin::toolchain::compile_c(&c, &executable, &config::c(&cc)).unwrap();
    assert!(Command::new(executable).status().unwrap().success());
    let glsl = glsl::emit(&module, "kernel", Stage::Compute).unwrap();
    let spv =
        resin::toolchain::compile_glsl(&glsl, Stage::Compute, &config::glsl(&compiler)).unwrap();
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Data {
        marker: u32,
        wide: u64,
        amount: f32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Payload {
        tag: u32,
        data: Data,
        end: u32,
    }
    #[repr(C)]
    struct Params {
        count: u32,
        values: u64,
        tail: f32,
    }
    const COUNT: usize = 67;
    let initial = Payload {
        tag: 10,
        data: Data {
            marker: 99,
            wide: 7,
            amount: 1.25,
        },
        end: 20,
    };
    assert_eq!((size_of::<Payload>(), align_of::<Payload>()), (40, 8));
    // Both address spaces use live, aligned allocations; reads follow synchronous submission.
    unsafe {
        let pipeline = gpu.create_compute_pipeline(&spv).unwrap();
        let values = gpu
            .malloc(
                (COUNT + 1) * size_of::<Payload>(),
                align_of::<Payload>(),
                ResinMemory::Default,
            )
            .unwrap();
        let root = gpu
            .malloc(
                size_of::<Params>(),
                align_of::<Params>(),
                ResinMemory::Default,
            )
            .unwrap();
        std::slice::from_raw_parts_mut(values.host_pointer().cast::<Payload>(), COUNT + 1)
            .fill(initial);
        root.host_pointer().cast::<Params>().write(Params {
            count: COUNT as u32,
            values: values.device_pointer(),
            tail: 0.75,
        });
        let mut commands = gpu.start_command_recording().unwrap();
        commands.set_pipeline(&pipeline).unwrap();
        commands
            .dispatch(root.device_pointer(), (COUNT as u32).div_ceil(64), 1, 1)
            .unwrap();
        gpu.submit(commands).unwrap();
        let values = std::slice::from_raw_parts(values.host_pointer().cast::<Payload>(), COUNT + 1);
        for (index, value) in values[..COUNT].iter().enumerate() {
            assert_eq!(
                *value,
                Payload {
                    tag: 11,
                    data: Data {
                        marker: index as u32,
                        wide: 4294967304,
                        amount: 1.75
                    },
                    end: 22
                }
            );
        }
        assert_eq!(values[COUNT], initial);
        assert_eq!((*root.host_pointer().cast::<Params>()).tail, 0.75);
    }
}

#[test]
fn particles_compute_then_render_from_the_same_buffer() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/particles.resin");
    let module = resin::ir::generate_program(&resin::ast::load(&source).unwrap()).unwrap();
    let compile = |stage: Stage| {
        let glsl = glsl::emit(&module, stage.entry(), stage).unwrap();
        resin::toolchain::compile_glsl(&glsl, stage, &config::glsl(&compiler)).unwrap()
    };
    let compute = compile(Stage::Compute);
    let vertex = compile(Stage::Vertex);
    let fragment = compile(Stage::Fragment);
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Particle {
        x: f32,
        y: f32,
        z: f32,
        vx: f32,
        vy: f32,
        vz: f32,
    }
    #[repr(C)]
    struct Params {
        count: u32,
        dt: f32,
        yaw_cos: f32,
        yaw_sin: f32,
        pitch_cos: f32,
        pitch_sin: f32,
        zoom: f32,
        aspect: f32,
        radius: f32,
        particles: u64,
        particle_length: u64,
    }
    const COUNT: usize = 1_000_000;
    let sentinel = Particle {
        x: 101.0,
        y: 102.0,
        z: 103.0,
        vx: 104.0,
        vy: 105.0,
        vz: 106.0,
    };
    assert_eq!(size_of::<Particle>(), 24);
    assert_eq!(size_of::<Params>(), 56);
    // The full draw catches accidental per-vertex searches that scale quadratically.
    // An extra workgroup exercises the count guard without touching the sentinel.
    unsafe {
        let compute = gpu.create_compute_pipeline(&compute).unwrap();
        let graphics = gpu.create_graphics_pipeline(&vertex, &fragment).unwrap();
        let particles = gpu
            .malloc(
                (COUNT + 1) * size_of::<Particle>(),
                align_of::<Particle>(),
                ResinMemory::Default,
            )
            .unwrap();
        let root = gpu
            .malloc(
                size_of::<Params>(),
                align_of::<Params>(),
                ResinMemory::Default,
            )
            .unwrap();
        let values =
            std::slice::from_raw_parts_mut(particles.host_pointer().cast::<Particle>(), COUNT + 1);
        for (index, p) in values[..COUNT].iter_mut().enumerate() {
            *p = Particle {
                x: (index % 1000) as f32 * 0.048 - 24.0,
                y: (index / 1000) as f32 * 0.060 - 30.0,
                z: (index % 97) as f32 * 0.5,
                vx: 1.0,
                vy: -2.0,
                vz: 3.0,
            };
        }
        values[COUNT] = sentinel;
        let first = values[0];
        let last = values[COUNT - 1];
        root.host_pointer().cast::<Params>().write(Params {
            count: COUNT as u32,
            dt: 0.005,
            yaw_cos: 1.0,
            yaw_sin: 0.0,
            pitch_cos: 1.0,
            pitch_sin: 0.0,
            zoom: 1.0,
            aspect: 1.0,
            radius: 0.006,
            particles: particles.device_pointer(),
            particle_length: COUNT as u64,
        });
        let mut image = gpu.create_image(256, 256).unwrap();
        let pixels = gpu.malloc(256 * 256 * 4, 4, ResinMemory::Readback).unwrap();
        for _ in 0..3 {
            let mut commands = gpu.start_command_recording().unwrap();
            commands.set_pipeline(&compute).unwrap();
            commands
                .dispatch(root.device_pointer(), (COUNT as u32).div_ceil(64) + 1, 1, 1)
                .unwrap();
            commands
                .begin_rendering(&mut image, [0.0, 0.0, 0.0, 1.0])
                .unwrap();
            commands.set_pipeline(&graphics).unwrap();
            commands
                .draw(root.device_pointer(), (COUNT * 24) as u32)
                .unwrap();
            commands.end_rendering().unwrap();
            commands.copy_image_to_buffer(&mut image, &pixels).unwrap();
            gpu.submit(commands).unwrap();
        }
        let values =
            std::slice::from_raw_parts(particles.host_pointer().cast::<Particle>(), COUNT + 1);
        assert_eq!(values[COUNT], sentinel);
        assert_ne!(values[0], first);
        assert_ne!(values[COUNT - 1], last);
        for p in &values[..COUNT] {
            assert!(
                [p.x, p.y, p.z, p.vx, p.vy, p.vz]
                    .iter()
                    .all(|v| v.is_finite())
            );
            assert!(p.x.abs() < 100.0 && p.y.abs() < 100.0 && p.z.abs() < 100.0);
        }
        let lit = pixels
            .host_bytes()
            .unwrap()
            .chunks_exact(4)
            .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
            .count();
        assert!(
            lit > 1000,
            "expected a visible particle cloud, got {lit} lit pixels"
        );

        // Magnify one sphere to verify the silhouette and per-fragment normal lighting.
        particles.host_pointer().cast::<Particle>().write(Particle {
            x: 0.0,
            y: 0.0,
            z: 25.0,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
        });
        (*root.host_pointer().cast::<Params>()).count = 1;
        (*root.host_pointer().cast::<Params>()).radius = 0.5;
        let mut commands = gpu.start_command_recording().unwrap();
        commands
            .begin_rendering(&mut image, [0.0, 0.0, 0.0, 1.0])
            .unwrap();
        commands.set_pipeline(&graphics).unwrap();
        commands.draw(root.device_pointer(), 24).unwrap();
        commands.end_rendering().unwrap();
        commands.copy_image_to_buffer(&mut image, &pixels).unwrap();
        gpu.submit(commands).unwrap();
        let bytes = pixels.host_bytes().unwrap();
        let blue = |x: usize, y: usize| bytes[(y * 256 + x) * 4 + 2];
        assert!(blue(128, 128) > 40, "sphere center should be lit");
        assert!(
            blue(110, 100) > blue(146, 156) + 15,
            "upper-left lighting should shade the curved surface"
        );
        assert_eq!(
            blue(185, 185),
            0,
            "sphere should not fill its bounding square"
        );
    }
}

#[test]
fn fragment_shaders_read_typed_root_parameters() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/triangle.resin");
    let mut ast = resin::ast::load(&source).unwrap();
    let file = &mut ast.modules.last_mut().unwrap().file;
    file.stmts.retain(|stmt| !matches!(&stmt.val, resin::ast::StmtKind::Function { name, .. } if name.val.as_ref() == "fragment"));
    file.stmts.extend(
        support::parse(
            "export { fragment }; @fragment_shader def fragment (color: Color, root: Ptr<Color>) -> Color = { root.* };",
        )
        .stmts,
    );
    let module = resin::ir::generate_program(&ast).unwrap();
    let compile = |stage: Stage| {
        let glsl = glsl::emit(&module, stage.entry(), stage).unwrap();
        resin::toolchain::compile_glsl(&glsl, stage, &config::glsl(&compiler)).unwrap()
    };
    let vertex = compile(Stage::Vertex);
    let fragment = compile(Stage::Fragment);
    // The mapped root is updated only between completed submissions.
    unsafe {
        let pipeline = gpu.create_graphics_pipeline(&vertex, &fragment).unwrap();
        let root = gpu.malloc(16, 4, ResinMemory::Default).unwrap();
        let mut image = gpu.create_image(64, 64).unwrap();
        let pixels = gpu.malloc(64 * 64 * 4, 4, ResinMemory::Readback).unwrap();
        for color in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]] {
            root.host_pointer().cast::<[f32; 4]>().write(color);
            let mut commands = gpu.start_command_recording().unwrap();
            commands
                .begin_rendering(&mut image, [0.0, 0.0, 0.0, 1.0])
                .unwrap();
            commands.set_pipeline(&pipeline).unwrap();
            commands.draw(root.device_pointer(), 3).unwrap();
            commands.end_rendering().unwrap();
            commands.copy_image_to_buffer(&mut image, &pixels).unwrap();
            gpu.submit(commands).unwrap();
            let pixels = pixels.host_bytes().unwrap();
            let center = (32 * 64 + 32) * 4;
            assert_eq!(
                &pixels[center..center + 4],
                color.map(|v| (v * 255.0) as u8)
            );
        }
    }
}

#[test]
fn shader_while_loops_execute_with_nested_and_zero_trip_iterations() {
    compute_values(
        "export { kernel }; struct PixelRoot { count: uint, pixels: Ptr<uint> }; def kernel(index: uint, root: Ptr<PixelRoot>) = { if (index < root.count) { var output = Span<uint> { data = root.pixels, length = 67L }; output(index).* := { var total = uint (0); var n = index; while (n > uint (0) && n <= index) { var j = uint (0); while (j < n) { total := total + uint (1); j := j + uint (1); }; n := n - uint (1); }; total }; }; };",
        |index| index * (index + 1) / 2,
    );
}

#[test]
fn shader_results_propagate_and_match_union_payloads_on_device() {
    compute_values(
        "export { kernel }; struct Zero {}; struct Odd { index: uint }; def checked(i: uint) -> Result<uint, Zero | Odd> = { if (i == uint(0)) { err(Zero {}) } else { if ((i & uint(1)) == uint(1)) { err(Odd { index = i }) } else { ok(i) } } }; def add(i: uint) -> Result<uint, _> = { var value = checked(i)?; ok(value + uint(10)) }; struct PixelRoot { count: uint, pixels: Ptr<uint> }; def kernel(i: uint, root: Ptr<PixelRoot>) = { if (i < root.count) { var output = Span<uint> { data = root.pixels, length = 67L }; output(i).* := { match (add(i)) { ok(value) => { value }, err(error) => { match (error) { Zero(zero) => { uint(0) }, Odd(odd) => { odd.index * uint(2) } } } } }; }; };",
        |index| {
            if index == 0 {
                0
            } else if index % 2 == 1 {
                index * 2
            } else {
                index + 10
            }
        },
    );
}

#[test]
fn shader_defer_preserves_values_and_runs_each_iteration() {
    compute_values(
        "export { kernel }; struct PixelRoot { count: uint, pixels: Ptr<uint> }; def kernel(i: uint, root: Ptr<PixelRoot>) = { if (i < root.count) { var output = Span<uint> { data = root.pixels, length = 67L }; output(i).* := { var n = uint(0); var total = uint(0); var saved = { defer { total := total * uint(2); }; while (n < i) { defer { total := total + n; }; n := n + uint(1); }; total }; saved + total }; }; };",
        |index| 3 * index * (index + 1) / 2,
    );
}

#[test]
fn shader_defer_delays_conditions_and_discards_expression_values() {
    compute_values(
        "export { kernel }; struct PixelRoot { count: uint, pixels: Ptr<uint> }; def kernel(i: uint, root: Ptr<PixelRoot>) = { if (i < root.count) { var output = Span<uint> { data = root.pixels, length = 67L }; output(i).* := { var n = uint(0); var total = uint(0); var saved = { defer if (n == i) { total := total * uint(2) } else { total := uint(999) }; defer while (n < i) { defer total := total + n; n := n + uint(1); }; total }; saved + total }; }; };",
        |index| index * (index + 1),
    );
}

#[test]
fn shader_defer_unwinds_errors_on_device() {
    compute_values(
        r#"export { kernel };
        struct Root { count: uint, pixels: Ptr<uint> };
        struct Odd { index: uint };
        def checked(i: uint) -> Result<uint, Odd> = {
            if ((i & uint(1)) == uint(1)) { err(Odd { index = i }) } else { ok(i) }
        };
        def work(i: uint, p: Ptr<uint>) -> Result<uint, _> = {
            defer p.* := p.* * uint(10) + uint(3);
            var n = {
                defer p.* := p.* * uint(10) + uint(2);
                var value = checked(i)?;
                defer p.* := uint(1);
                value
            };
            ok(n)
        };
        def kernel(i: uint, root: Ptr<Root>) = {
            if (i < root.count) {
                var p = (Span<uint> { data = root.pixels, length = ulong(67) })(i);
                p.* := uint(0);
                match (work(i, p)) {
                    ok(n) => { p.* := p.* + n; },
                    err(e) => { p.* := p.* + e.index; },
                };
                ()
            } else { () }
        };"#,
        |index| index + if index % 2 == 1 { 23 } else { 123 },
    );
}

#[test]
fn optional_unwrap_stops_shader_callers_on_none() {
    compute_values(
        r#"export { kernel };
        struct Root { count: uint, pixels: Ptr<uint> };
        def choose(i: uint) -> uint = { var value: uint | None; value := if ((i & 1I) == 0I) { i } else { None }; value! };
        def kernel(i: uint, root: Ptr<Root>) = {
            if (i < root.count) {
                var output = Span<uint> { data = root.pixels, length = 67L };
                output(i).* := 7I;
                var value = choose(i);
                output(i).* := value + 1I;
            };
        };"#,
        |index| if index % 2 == 0 { index + 1 } else { 7 },
    );
}

#[test]
fn none_elimination_preserves_shader_union_members() {
    compute_values(
        r#"export { kernel };
        struct Root { count: uint, pixels: Ptr<uint> };
        def choose(i: uint) -> uint | bool | None = {
            if ((i & 3I) == 0I) { None } else { if ((i & 3I) == 1I) { i } else { 1I == 1I } }
        };
        def read(i: uint) -> uint = {
            match (choose(i)!) { uint(n) => { n + 1I }, bool(b) => { if (b) { 42I } else { 0I } } }
        };
        def kernel(i: uint, root: Ptr<Root>) = {
            if (i < root.count) {
                var output = Span<uint> { data = root.pixels, length = 67L };
                output(i).* := 7I;
                output(i).* := read(i);
            };
        };"#,
        |index| match index % 4 {
            0 => 7,
            1 => index + 1,
            _ => 42,
        },
    );
}

#[test]
fn inherent_methods_execute_in_shader_helpers() {
    compute_values(
        r#"export { kernel };
        struct Root { count: uint, pixels: Ptr<uint> };
        struct Counter { value: uint };
        impl Counter {
            def new(value: uint) -> Counter = { Counter { value = value } };
            def add(self: Counter, n: uint) -> Counter = { Counter { value = self.value + n } };
            def read(self: Counter) -> uint = { self.value };
        }
        def kernel(id: uint, root: Ptr<Root>) = {
            if (id < root.count) {
                var counter = Counter.new(id);
                var incremented = counter.add(1I);
                var output = Span<uint> { data = root.pixels, length = 67L };
                output(id).* := incremented.read();
            };
        };
        "#,
        |index| index + 1,
    );
}

fn compute_values(source: &str, expected: fn(u32) -> u32) {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let module = support::module(source);
    let glsl = resin::backend::glsl::emit(&module, "kernel", resin::backend::glsl::Stage::Compute)
        .unwrap();
    let spv = resin::toolchain::compile_glsl(
        &glsl,
        resin::backend::glsl::Stage::Compute,
        &config::glsl(&compiler),
    )
    .unwrap();
    #[repr(C)]
    struct Root {
        count: u32,
        pixels: u64,
    }
    const COUNT: u32 = 67;
    // All resources share one GPU and remain live until synchronous submission completes.
    unsafe {
        let pipeline = gpu.create_compute_pipeline(&spv).unwrap();
        let pixels = gpu
            .malloc((COUNT as usize + 1) * 4, 4, ResinMemory::Default)
            .unwrap();
        let root = gpu
            .malloc(size_of::<Root>(), align_of::<Root>(), ResinMemory::Default)
            .unwrap();
        std::slice::from_raw_parts_mut(pixels.host_pointer().cast::<u32>(), COUNT as usize + 1)
            .fill(u32::MAX);
        root.host_pointer().cast::<Root>().write(Root {
            count: COUNT,
            pixels: pixels.device_pointer(),
        });
        let mut commands = gpu.start_command_recording().unwrap();
        commands.set_pipeline(&pipeline).unwrap();
        commands
            .dispatch(root.device_pointer(), COUNT.div_ceil(64), 1, 1)
            .unwrap();
        gpu.submit(commands).unwrap();
        let values =
            std::slice::from_raw_parts(pixels.host_pointer().cast::<u32>(), COUNT as usize + 1);
        for (index, &value) in values[..COUNT as usize].iter().enumerate() {
            assert_eq!(value, expected(index as u32), "invocation {index}");
        }
        assert_eq!(values[COUNT as usize], u32::MAX);
    }
}

#[test]
fn ordinary_resin_programs_render_and_write_pngs() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(gpu) = gpu() else { return };
    drop(gpu);
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    for name in ["gradient", "triangle"] {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join(format!("{name}.resin"));
        let executable = temp
            .path()
            .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
        let output = Command::new(env!("CARGO_BIN_EXE_resin"))
            .current_dir(temp.path())
            .arg(source)
            .arg("--glslc")
            .arg(&compiler)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
        assert!(!temp.path().join(format!("{name}.png")).exists());

        // The copy is standalone: no compiler or source files are needed to run it.
        let output = Command::new(executable)
            .current_dir(temp.path())
            .env("GLSLC", "/missing/glslc")
            .env("CC", "/missing/cc")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("wrote {name}.png\n")
        );
    }
    let image = image_read_png(temp.path().join("gradient.png"), 4).unwrap();
    assert_eq!((image.width, image.height), (256, 256));
    assert_eq!(image.pixels.len(), 256 * 256 * 4);
    for (i, pixel) in image.pixels.chunks_exact(4).enumerate() {
        assert_eq!(pixel, &[i as u8, (i >> 8) as u8, 64, 255], "pixel {i}");
    }
    let image = image_read_png(temp.path().join("triangle.png"), 4).unwrap();
    let reference = image_read_png(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/resin-runtime/tests/hello_triangle.png"
        ),
        4,
    )
    .unwrap();
    assert_eq!(
        (image.width, image.height),
        (reference.width, reference.height)
    );
    assert_eq!(image.pixels.len(), reference.pixels.len());
    for (i, (&actual, &expected)) in image.pixels.iter().zip(&reference.pixels).enumerate() {
        let tolerance = u8::from(i % 4 != 3);
        assert!(
            actual.abs_diff(expected) <= tolerance,
            "channel {i}: {actual} != {expected}"
        );
    }
}

#[test]
fn invalid_images_do_not_replace_existing_files() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let output = temp.path().join("existing.png");
    std::fs::write(&output, b"keep me").unwrap();
    assert!(image_write_png(&output, 2, 2, 4, &[]).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), b"keep me");
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
}

#[test]
fn array_indexing_and_helper_bounds_failures_stop_the_invocation() {
    compute_values(
        r#"export { kernel };
        def read(i: uint) -> uint = { var values = [uint(10), uint(20), uint(30)]; values(i).* };
        struct PixelRoot { count: uint, pixels: Ptr<uint> }; @compute_shader def kernel(i: uint, root: Ptr<PixelRoot>) = { if (i < root.count) { var output = Span<uint> { data = root.pixels, length = 67L }; output(i).* := { read(i) + uint(1) }; }; };"#,
        |i| if i < 3 { (i + 1) * 10 + 1 } else { u32::MAX },
    );
}

#[test]
fn span_write_bounds_failures_stop_callers_before_later_side_effects() {
    compute_values(
        r#"export { kernel };
        struct Root { count: uint, pixels: Ptr<uint> };
        def write(i: uint, pixels: Span<uint>) = { pixels(i).* := uint(42); };
        @compute_shader def kernel(i: uint, root: Ptr<Root>) = {
            var pixels = Span<uint> { data = root.pixels, length = ulong(3) };
            write(i, pixels);
            pixels(i).* := uint(43);
        };"#,
        |i| if i < 3 { 43 } else { u32::MAX },
    );
}

#[test]
fn numeric_suffixes_and_one_armed_if_execute_on_device() {
    compute_values(
        r#"export { kernel };
        struct PixelRoot { count: uint, pixels: Ptr<uint> }; @compute_shader def kernel(i: uint, root: Ptr<PixelRoot>) = { if (i < root.count) { var output = Span<uint> { data = root.pixels, length = 67L }; output(i).* := {
            var result = 0I;
            if ((i & 1I) == 0I) { result := i + 10I; };
            if (1.5f + 2.5f == 4f && 42L > 0L) { result := result + 1I; };
            result
        }; }; };"#,
        |i| if i & 1 == 0 { i + 11 } else { 1 },
    );
}

#[path = "support/interactions.rs"]
mod interactions;

fn execute_interaction(source: &str, expected: [u32; 2]) {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let m = support::module(source);
    let glsl = glsl::emit(&m, "kernel", Stage::Compute).unwrap();
    let spv =
        resin::toolchain::compile_glsl(&glsl, Stage::Compute, &config::glsl(&compiler)).unwrap();
    // Each fixture's root fits two uints. Read only after synchronous submission.
    unsafe {
        let pipeline = gpu.create_compute_pipeline(&spv).unwrap();
        let root = gpu.malloc(8, 4, ResinMemory::Default).unwrap();
        root.host_pointer().cast::<[u32; 2]>().write([0, 0]);
        let mut commands = gpu.start_command_recording().unwrap();
        commands.set_pipeline(&pipeline).unwrap();
        commands.dispatch(root.device_pointer(), 1, 1, 1).unwrap();
        gpu.submit(commands).unwrap();
        assert_eq!(
            root.host_pointer().cast::<[u32; 2]>().read(),
            expected,
            "{source}"
        );
    }
}

#[test]
fn interacting_features_execute_equivalently_on_gpu() {
    for (i, source) in interactions::variants().iter().enumerate() {
        let expected = match i / interactions::MARKERS.len() {
            0 => [7, 42],
            1 | 2 => [42, 0],
            3 => [921, 42],
            _ => unreachable!(),
        };
        execute_interaction(source, expected);
    }
}

#[test]
fn numeric_conversion_failures_stop_shader_helpers_before_stores() {
    for expression in [
        "uint(-1i)",
        "uint(-1.0f)",
        "uint(4294967296L)",
        "int(0.0f / 0.0f)",
        "int(1.0f / 0.0f)",
    ] {
        let source = format!(
            "export {{ kernel }}; def invalid() = {{ {expression}; }}; def kernel(i: uint, p: Ptr<uint>) = {{ if (i == 0I) {{ invalid(); p.* := 99I; }}; }};"
        );
        execute_interaction(&source, [0, 0]);
    }
}
