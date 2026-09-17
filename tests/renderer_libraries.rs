#[allow(dead_code)]
mod support;

#[test]
fn linear_algebra_and_extended_math() {
    let source = r#"export { main }; import { "$/linalg.resin", "$/math.resin" };
    fn main() {
        let camera = look_at(vec3(2,3,4),vec3(2,3,0),vec3(0,1,0));
        let origin = transform_point(inverse_rigid(camera),vec3(2,3,4));
        assert(length(origin) < 0.0001);
        let world = transform_point(camera,vec3(0,0,-2));
        assert(length(sub(world,vec3(2,3,2))) < 0.0001);
        assert(abs(pow(f32(2),f32(3))-8) < 0.0001);
        assert(abs(atan2(f32(1),f32(0))-1.5707963) < 0.0001);
        assert(floor(f32(-0.2)) == -1);
        assert(abs(exp(log(f32(3)))-3) < 0.0001);
        let projection = perspective(1,vec2(2,2),1,10);
        let clip = mul(projection,vec4(0,0,-1,1));
        assert(abs(clip.z) < 0.0001);
    }"#;
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn static_gltf_scene_tables_and_retained_source_owner() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/arris/scene.gltf");
    let scene = resin_runtime::GltfScene::load(&path).unwrap();
    assert_eq!(scene.materials().len(), 5);
    assert_eq!(scene.textures().len(), 1);
    assert_eq!(scene.pixels().len(), 8 * 8 * 4);
    assert_eq!(scene.vertices().len() % 3, 0);
    assert!(
        scene
            .vertices()
            .iter()
            .all(|v| v.position.iter().all(|n| n.is_finite()))
    );
    assert!(
        scene
            .vertices()
            .iter()
            .any(|v| v.material == 1 && v.position[0] < -1.5)
    );
    assert_eq!(scene.materials()[3].flags, 5);
    let source = r#"export { main }; import { "$/gltf.resin", "$/span.resin" };
        fn main() -> () | Err<_> {
            let scene = { let original = gltf_load("examples/assets/arris/scene.gltf".data)?; original:clone() };
            let mesh = scene:vertices(); let surfaces = scene:materials(); let images = scene:textures();
            assert(mesh.length > 1000 && mesh.length % 3 == 0);
            assert(surfaces.length == 5 && surfaces:at(3).flags == 5);
            assert(images:at(0).width == 8);
            assert(mesh:at(0).position.x == -4);
        }"#;
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn arris_example_compiles() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/arris.resin");
    let module =
        support::pipeline::host_entry(&path, "main").unwrap_or_else(|error| panic!("{error}"));
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    project.build_executable();
}

#[cfg(feature = "gpu")]
fn available_gpu(ray: bool) -> bool {
    use resin_runtime::{ResinGpu, ResinStatus, resin_gpu_supports_ray_tracing};
    let requirement = if ray {
        "RESIN_REQUIRE_RAY_TRACING"
    } else {
        "RESIN_REQUIRE_GPU"
    };
    let required = std::env::var(requirement).as_deref() == Ok("1");
    match ResinGpu::create() {
        Ok(gpu) => {
            let suitable = !ray || unsafe { resin_gpu_supports_ray_tracing(&gpu) } != 0;
            assert!(suitable || !required, "ray tracing pipelines required");
            suitable
        }
        Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
            assert!(!required, "GPU required");
            false
        }
        Err(error) => panic!("GPU initialization: {error:?}"),
    }
}

#[test]
#[cfg(feature = "gpu")]
fn float_visibility_depth_and_discard_preserve_nearest_surface() {
    let _lock = resin_runtime::testing::lock_gpu();
    if !available_gpu(false) {
        return;
    }
    let source = r#"export { main }; import { "$/gpu.resin", "$/graphics.resin", "$/span.resin", "$/shared.resin", "$/linalg.resin" };
    struct Parameters { depth: f32, value: f32, masked: u32 }
    @vertex_shader fn vertex(index: i32,root: Ptr<Parameters>) -> Vertex {
        let x: f32=if (index == 1) { 3 } else { -1 }; let y: f32=if (index == 2) { 3 } else { -1 };
        Vertex { position=Position { x=x,y=y,z=root.depth,w=1 },color=Color { r=(x+1)*0.5,g=0,b=0,a=1 } }
    }
    @fragment_shader fn fragment(input: Color,root: Ptr<Parameters>) -> Color | None {
        if (root.masked != 0 && input.r < 0.5) { None } else { Color { r=root.value,g=-2,b=16,a=1 } }
    }
    fn main() -> () | Err<_> {
        let gpu=gpu_new()?; let config=gpu:graphics_config(image_rgba32f,true);
        let pipeline=config:create_graphics_pipeline(vertex,fragment)?;
        let image=gpu:create_image(16,8,image_rgba32f)?; let depth=gpu:create_image(16,8,image_depth32)?;
        let pixels=gpu:alloc::<Vec4>(128)?; let depths=gpu:alloc::<f32>(128)?;
        let commands=gpu:start_command_recording()?; commands:begin_rendering(image,depth,0,0,0,0)?;
        // Draw closest masked geometry first; holes must not commit depth.
        commands:draw(pipeline,Parameters { depth=0.2,value=64,masked=1 },3)?;
        commands:draw(pipeline,Parameters { depth=0.8,value=4,masked=0 },3)?;
        // Middle layer must replace the far layer, but cannot overwrite near pixels.
        commands:draw(pipeline,Parameters { depth=0.5,value=8,masked=0 },3)?;
        commands:end_rendering()?; commands:copy_image_to_buffer(image,pixels)?; commands:copy_image_to_buffer(depth,depths)?; commands:submit()?;
        let host=arc_span_alloc(128,vec4(0,0,0,0))?; pixels:copy_to(host:get()); let colors=host:get();
        let depth_host=arc_span_alloc(128,f32(0))?; depths:copy_to(depth_host:get()); let distances=depth_host:get();
        let mut i: u64=0; while (i < 128) {
            let expected=if (i%16 < 8) { f32(8) } else { f32(64) }; let z=if (i%16 < 8) { f32(0.5) } else { f32(0.2) };
            assert(colors:at(i).x == expected && colors:at(i).y == -2 && colors:at(i).z == 16);
            assert(abs(distances:at(i)-z) < 0.00001); i=i+1;
        };
    }"#;
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&output.stdout).contains("Validation Error"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        !String::from_utf8_lossy(&output.stderr).contains("Validation Error"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[cfg(feature = "gpu")]
fn headless_gltf_hdri_render_is_finite_and_has_visible_geometry() {
    let _lock = resin_runtime::testing::lock_gpu();
    if !available_gpu(true) {
        return;
    }
    // Run the actual tutorial composition at a small size; do not duplicate its
    // integrator in a test-only renderer. Readback assertions are inserted at the
    // application's output boundary, where the guides and HDR buffers are ready.
    let source = include_str!("../examples/arris.resin")
        .replace("let width: u32 = 640;", "let width: u32 = 64;")
        .replace("let height: u32 = 480;", "let height: u32 = 48;")
        .replace("frame < 16", "frame < 3")
        .replace("frame == 15", "frame == 2")
        .replace(
            "save_images(width, height, pixels, filtered)?;",
            r#"
            let host = arc_span_alloc(count, vec4(0,0,0,0))?;
            positions:copy_to(host:get()); let guides = host:get();
            let mut visible: u32 = 0; let mut i: u64 = 0;
            while (i < count) { if (guides:at(i).w > 0) { visible=visible+1; }; i=i+1; };
            assert(visible > 500 && visible < 2800);
            save_images(width, height, pixels, filtered)?;
        "#,
        );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("render.resin");
    std::fs::write(&path, source).unwrap();
    let module =
        support::pipeline::host_entry(&path, "main").unwrap_or_else(|error| panic!("{error}"));
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let executable = project.build_executable();
    let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/arris");
    let output = std::process::Command::new(executable.path())
        .arg(assets.join("scene.gltf"))
        .arg(assets.join("studio.hdr"))
        .current_dir(directory.path())
        .output()
        .unwrap();
    assert_valid_output(&output);
    let image = resin_runtime::image_read_exr(directory.path().join("arris.exr")).unwrap();
    assert_eq!((image.width, image.height), (64, 48));
    assert!(
        image
            .pixels
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v < 10000.0)
    );
    assert!(image.pixels.chunks_exact(4).all(|rgba| rgba[3] == 1.0));
    assert!(
        image
            .pixels
            .chunks_exact(4)
            .map(|rgba| rgba[0] + rgba[1] + rgba[2])
            .sum::<f32>()
            > 100.0
    );
    assert!(directory.path().join("arris.png").metadata().unwrap().len() > 100);
}

#[cfg(feature = "gpu")]
fn assert_valid_output(output: &std::process::Output) {
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for bytes in [&output.stdout, &output.stderr] {
        assert!(
            !String::from_utf8_lossy(bytes).contains("Validation Error"),
            "{}",
            String::from_utf8_lossy(bytes)
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn standalone_svgf_reduces_noise_preserves_background_and_resets() {
    let _lock = resin_runtime::testing::lock_gpu();
    if !available_gpu(false) {
        return;
    }
    let source = r#"export { main }; import { "$/svgf.resin", "$/gpu.resin", "$/span.resin", "$/shared.resin", "$/linalg.resin" };
    fn main() -> () | Err<_> {
        let gpu=gpu_new()?; let colors=gpu:alloc::<Vec4>(512)?; let positions=gpu:alloc::<Vec4>(512)?; let normals=gpu:alloc::<Vec4>(512)?;
        let mut denoiser=svgf_create(gpu,32,16)?;
        let mut i: u64=0;
        while (i < 512) {
            let x=u32(i%32); let y=u32(i/32); let value=if ((x+y)%2 == 0) { f32(0.5) } else { f32(1.5) };
            let c=colors:at(i); c:store(vec4(value,value,value,1));
            let p=positions:at(i); p:store(vec4(2*(f32(x)+0.5)/32-1,2*(f32(y)+0.5)/16-1,0.5,if (x == 0) { f32(0) } else { f32(1) }));
            let n=normals:at(i); n:store(vec4(0,0,1,1)); i=i+1;
        };
        let result=denoiser:filter(colors,positions,normals,identity_transform())?;
        let host=arc_span_alloc(512,vec4(0,0,0,0))?; result:copy_to(host:get()); let values=host:get();
        let mut variance: f32=0; i=0;
        while (i < 512) {
            if (i%32 == 0) { assert(values:at(i).x == (if ((i/32)%2 == 0) { f32(0.5) } else { f32(1.5) })); }
            else { let error=values:at(i).x-1; variance=variance+error*error; };
            let c=colors:at(i); c:store(vec4(4,4,4,1)); i=i+1;
        };
        assert(variance/496 < 0.03);
        denoiser:reset(); let changed=denoiser:filter(colors,positions,normals,identity_transform())?;
        changed:copy_to(host:get()); i=0; while (i < 512) { assert(abs(values:at(i).x-4) < 0.001); i=i+1; };
        let history=denoiser:filter(colors,positions,normals,identity_transform())?;
        history:copy_to(host:get()); i=0; while (i < 512) { assert(abs(values:at(i).x-4) < 0.001); i=i+1; };
    }"#;
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for bytes in [&output.stdout, &output.stderr] {
        assert!(
            !String::from_utf8_lossy(bytes).contains("Validation Error"),
            "{}",
            String::from_utf8_lossy(bytes)
        );
    }
}

#[test]
fn importance_sampling_matches_pdfs_and_white_furnace_energy() {
    let source = r#"export { main }; import { "$/renderer/brdf.resin", "$/renderer/scene.resin", "$/renderer/environment.resin", "$/linalg.resin", "$/shared.resin", "$/span.resin" };
    fn main() -> () | Err<_> {
        let surface=Surface { position=vec3(0,0,0),geometric=vec3(0,0,1),normal=vec3(0,0,1),albedo=vec3(1,1,1),emission=vec3(0,0,0),metallic=0,roughness=0.5,alpha=1,cutoff=0.5,flags=0 };
        let mut state: u32=17; let mut sum=vec3(0,0,0); let mut i: u32=0;
        while (i < 20000) {
            let sample=bsdf_sample(surface,vec3(0,0,1),random(state),random(state),random(state));
            assert(sample.pdf >= 0 && sample.weight.x >= 0);
            assert(abs(sample.pdf-bsdf_pdf(surface,vec3(0,0,1),sample.direction)) < 0.0001);
            sum=add(sum,sample.weight); i=i+1;
        };
        // A unit-radiance furnace cannot gain energy; this single-scattering
        // diffuse/specular model should retain almost all of a white dielectric.
        assert(sum.x/20000 > 0.9 && sum.x/20000 < 1.03);
        let pixels=arc_span_alloc(8,vec4(2,3,4,1))?; let weights=arc_span_alloc(9,f32(0))?; let cdf=weights:get();
        i=0; while (i <= 8) { cdf:at_mut(u64(i))=f32(i)/8; i=i+1; };
        let env=ShaderEnvironment { pixels=pixels:get(),cdf=cdf,width=4,height=2,yaw=0.7,tint=vec3(1,1,1) };
        let mut direction_sum=vec3(0,0,0); i=0;
        while (i < 10000) {
            let sample=environment_sample(env,random(state),random(state),random(state));
            assert(abs(sample.pdf-0.07957747) < 0.00001);
            assert(abs(sample.pdf-environment_pdf(env,sample.direction)) < 0.00001);
            assert(abs(length(sample.direction)-1) < 0.00001);
            let value=environment_value(env,sample.direction); assert(value.x == 2 && value.y == 3 && value.z == 4);
            direction_sum=add(direction_sum,sample.direction); i=i+1;
        };
        assert(length(mul(direction_sum,0.0001)) < 0.03);
    }"#;
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[cfg(feature = "gpu")]
fn visibility_and_display_compose_without_ray_tracing() {
    let _lock = resin_runtime::testing::lock_gpu();
    if !available_gpu(false) {
        return;
    }
    let source = r#"export { main };
    import { "$/renderer.resin", "$/gltf.resin", "$/gpu.resin", "$/linalg.resin", "$/shared.resin", "$/span.resin" };
    fn main() -> () | Err<_> {
        let gpu=gpu_new()?;
        let asset=gltf_load("examples/assets/arris/scene.gltf".data)?;
        let scene=upload_scene(gpu,asset)?;
        let config=gpu:graphics_config(image_rgba32f,true);
        let raster=config:create_graphics_pipeline(primary_vertex,primary_fragment)?;
        let mapping=gpu:create_compute_pipeline(tone_map)?;
        let attachment=gpu:create_image(64,48,image_rgba32f)?;
        let depth=gpu:create_image(64,48,image_depth32)?;
        let visibility=gpu:alloc::<Vec4>(3072)?;
        let view=camera(look_at(vec3(3,2.2,4.5),vec3(0,0.65,0),vec3(0,1,0)),f32(64)/48);
        let projection=mul(perspective(view.focal_length,view.sensor_size,view.near,view.far),inverse_rigid(view.transform));
        let root=RasterRoot<DeviceScene> { scene=scene,view_projection=projection,eye=vec3(3,2.2,4.5) };
        // A caller supplies its own HDR input/output, with no camera or renderer.
        let linear=gpu:alloc::<Vec4>(1)?; let pixel=linear:at(0); pixel:store(vec4(1,3,0,7));
        let display=gpu:alloc::<u8>(4)?;
        let tone=ToneRoot<GpuSpan<Vec4>,GpuSpan<u8>> { input=linear:clone(),pixels=display:clone(),count=1,exposure=1,gamma=1 };
        let commands=gpu:start_command_recording()?;
        commands:begin_rendering(attachment,depth,0,0,0,0)?;
        commands:draw(raster,root,u32(scene.vertices.length))?;
        commands:end_rendering()?;
        commands:copy_image_to_buffer(attachment,visibility)?;
        commands:dispatch(mapping,tone,1,1,1)?;
        commands:submit()?;
        let host=arc_span_alloc(3072,vec4(0,0,0,0))?; visibility:copy_to(host:get()); let values=host:get();
        let mut visible: u32=0; let mut i: u64=0;
        while (i < 3072) { if (values:at(i).x > 0) { visible=visible+1; }; i=i+1; };
        assert(visible > 500 && visible < 2800);
        let bytes=arc_span_alloc(4,u8(0))?; display:copy_to(bytes:get()); let rgba=bytes:get();
        assert(rgba:at(0) == 128 && rgba:at(1) == 191 && rgba:at(2) == 0 && rgba:at(3) == 255);
        assert(pixel:load().y == 3 && pixel:load().w == 7);
    }"#;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("raster.resin");
    std::fs::write(&path, source).unwrap();
    let module =
        support::pipeline::host_entry(&path, "main").unwrap_or_else(|error| panic!("{error}"));
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    // Vertex, fragment, and display compute only: importing the public module
    // does not force the application to create any RT shaders or acceleration.
    assert_eq!(project.generated.shaders().len(), 3);
    assert_valid_output(&project.run());
}

#[test]
#[cfg(feature = "gpu")]
fn custom_ray_root_uses_caller_buffers_and_bounded_alpha_traversal() {
    let _lock = resin_runtime::testing::lock_gpu();
    if !available_gpu(true) {
        return;
    }
    let source = r#"export { main };
    import { "$/renderer.resin", "$/gltf.resin", "$/gpu.resin", "$/linalg.resin", "$/shared.resin", "$/span.resin" };
    struct Probe<S,O> { scene: S, distances: O }
    type ShaderProbe=Probe<ShaderScene,Span<f32>>;
    @miss_shader fn miss(initial: Hit,root: Ptr<ShaderProbe>) -> Hit { miss_hit() }
    @closest_hit_shader fn closest(initial: Hit,root: Ptr<ShaderProbe>) -> Hit { closest_hit() }
    @ray_generation_shader fn probe(index: u64,root: Ptr<ShaderProbe>) {
        let maximum: f32=if (index == 0) { 0.5 } else { if (index == 1) { 1.5 } else { 3 } };
        let minimum: f32=if (index == 3) { 2.1 } else { 0 };
        let mut state: u32=17;
        let hit=trace_surface(root.scene,vec3(0,0,0),vec3(0,0,1),minimum,maximum,state);
        root.distances:at_mut(index)=if (hit.triangle == ~u32(0)) { f32(-1) } else { hit.distance };
    }
    fn main() -> () | Err<_> {
        let gpu=gpu_new()?;
        // Construct two triangles without a GltfScene or file loader. Near is
        // fully MASK-rejected; far is opaque and must only occlude long rays.
        let vertices_host=arc_span_alloc(6,GltfVertex { position=vec3(0,0,0),normal=vec3(0,0,1),uv=vec2(0,0),tangent=vec4(1,0,0,1),color=vec4(1,1,1,1),material=0 })?;
        let mesh=vertices_host:get(); let mut i: u64=0;
        while (i < 6) {
            mesh:at_mut(i).position=vec3(if (i%3 == 0) { f32(-1) } else { if (i%3 == 1) { f32(1) } else { f32(0) } },if (i%3 == 2) { f32(1) } else { f32(-1) },f32(i/3)+1);
            mesh:at_mut(i).material=u32(i/3); i=i+1;
        };
        let materials_host=arc_span_alloc(2,GltfMaterial { albedo=vec4(1,1,1,1),emissive=vec3(0,0,0),metallic=0,roughness=1,albedo_texture=~u32(0),metallic_roughness_texture=~u32(0),emissive_texture=~u32(0),normal_texture=~u32(0),normal_scale=1,alpha_cutoff=0.5,flags=4 })?;
        let surfaces=materials_host:get(); surfaces:at_mut(0).albedo.w=0; surfaces:at_mut(0).flags=5;
        let scene=DeviceScene { vertices=upload(gpu,mesh)?,materials=upload(gpu,surfaces)?,textures=gpu:alloc::<GltfTexture>(1)?,pixels=gpu:alloc::<u8>(4)? };
        let acceleration=scene_acceleration(gpu,mesh)?;
        let tracing=acceleration:create_ray_tracing_pipeline(probe,miss,closest)?;
        let distances=gpu:alloc::<f32>(4)?;
        let root=Probe<DeviceScene,GpuSpan<f32>> { scene=scene,distances=distances:clone() };
        let commands=gpu:start_command_recording()?;
        commands:trace_rays(tracing,root,4,1,1)?; commands:submit()?;
        let host=arc_span_alloc(4,f32(0))?; distances:copy_to(host:get()); let values=host:get();
        assert(values:at(0) == -1 && values:at(1) == -1 && abs(values:at(2)-2) < 0.0001 && values:at(3) == -1);
    }"#;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("probe.resin");
    std::fs::write(&path, source).unwrap();
    let module =
        support::pipeline::host_entry(&path, "main").unwrap_or_else(|error| panic!("{error}"));
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    assert_eq!(project.generated.shaders().len(), 3);
    assert_valid_output(&project.run());
}
