//! Interactive 3DGS viewer: drag to orbit, scroll to zoom.
//!
//! The renderer is the composed resin graph (chunked, camera-as-parameter):
//! both programs compile once at startup, then every frame is just new
//! parameter bytes. Presentation is winit + softbuffer (a plain CPU blit of
//! the rendered image) — deliberately the thinnest possible layer, per the
//! GOAL.md stance that presentation is a helper *on top* of the core.
//!
//! ```sh
//! resin-viewer --ply data/hf/dylanebert-3dgs/luigi/luigi.ply --size 128
//! resin-viewer --offscreen 8      # headless: dump an orbit as PPM frames
//! ```

use std::num::NonZeroU32;
use std::rc::Rc;

use resin_gaussians::{chunked_renderer, gnomen_cloud, load_ply, Camera, CloudData, GaussianCloud};
use resin_jit::{ConcreteTensor, Jit};

/// A compiled camera → image function (backend erased).
type Renderer = Box<dyn Fn(&Camera) -> Vec<f32>>;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct Options {
    ply: Option<String>,
    backend: String,
    size: usize,
    stride: usize,
    chunk: usize,
    offscreen: Option<usize>,
    up_y: f32,
}

struct Orbit {
    center: [f32; 3],
    radius: f32,
    yaw: f32,
    pitch: f32,
    up: [f32; 3],
}

impl Orbit {
    fn camera(&self, size: usize) -> Camera {
        Camera::orbit(
            self.center,
            self.radius,
            self.yaw,
            self.pitch,
            self.up,
            50.0,
            size,
            size,
        )
    }
}

fn main() {
    let mut opts = Options {
        ply: None,
        backend: "auto".into(),
        size: 96,
        stride: 1,
        chunk: 512,
        offscreen: None,
        up_y: 1.0,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ply" => opts.ply = Some(args.next().expect("--ply <path>")),
            "--backend" => opts.backend = args.next().expect("--backend <cpu|wgpu|auto>"),
            "--size" => opts.size = args.next().unwrap().parse().expect("--size <pixels>"),
            "--stride" => opts.stride = args.next().unwrap().parse().expect("--stride <int>"),
            "--chunk" => opts.chunk = args.next().unwrap().parse().expect("--chunk <int>"),
            "--offscreen" => {
                opts.offscreen = Some(args.next().unwrap().parse().expect("--offscreen <frames>"))
            }
            // COLMAP-convention (y-down) checkpoints: pass --up-y -1.
            "--up-y" => opts.up_y = args.next().unwrap().parse().expect("--up-y <1|-1>"),
            other => panic!("unknown argument {other}"),
        }
    }

    let data = match &opts.ply {
        Some(path) => load_ply(path).expect("load ply"),
        None => gnomen_cloud(),
    };
    let data = if opts.stride > 1 {
        data.subsample(opts.stride)
    } else {
        data
    };
    println!("cloud: {} gaussians", data.count());

    // Frame the cloud: orbit around the bbox center at ~2x its extent.
    let (lo, hi) = data.bounds();
    let center = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let extent = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2))
        .sqrt()
        .max(0.2);
    let orbit = Orbit {
        center,
        radius: extent * 1.6,
        yaw: 20.0,
        pitch: -10.0,
        up: [0.0, opts.up_y, 0.0],
    };

    let render = build_renderer(&opts, &data);

    if let Some(frames) = opts.offscreen {
        run_offscreen(&opts, orbit, render, frames);
    } else {
        let event_loop = EventLoop::new().expect("winit event loop (need a display)");
        let mut app = App {
            size: opts.size,
            window: None,
            surface: None,
            render,
            orbit,
            dragging: false,
            last_cursor: (0.0, 0.0),
            frame: None,
        };
        event_loop.run_app(&mut app).expect("run app");
    }
}

/// Compile the chunked renderer on the chosen backend and box it as a
/// camera → image closure.
fn build_renderer(opts: &Options, data: &CloudData) -> Renderer {
    let n = data.count();
    let chunk = opts.chunk.min(n.max(2));
    match opts.backend.as_str() {
        #[cfg(feature = "wgpu")]
        "wgpu" => boxed_renderer(resin_jit::backends::wgpu::WgpuJit, data, opts.size, chunk),
        #[cfg(feature = "cpu")]
        "cpu" => boxed_renderer(resin_jit::backends::cpu::CpuJit, data, opts.size, chunk),
        "auto" => {
            #[cfg(feature = "wgpu")]
            if resin_jit::backends::wgpu::shared_context_available() {
                println!("backend: wgpu");
                return boxed_renderer(
                    resin_jit::backends::wgpu::WgpuJit,
                    data,
                    opts.size,
                    chunk,
                );
            }
            #[cfg(feature = "cpu")]
            {
                println!("backend: cpu (interpreter — expect low frame rates)");
                return boxed_renderer(resin_jit::backends::cpu::CpuJit, data, opts.size, chunk);
            }
            #[allow(unreachable_code)]
            {
                panic!("no backend available")
            }
        }
        other => panic!("backend {other} not available in this build"),
    }
}

fn boxed_renderer<J: Jit>(
    jit: J,
    data: &CloudData,
    size: usize,
    chunk: usize,
) -> Renderer {
    let n = data.count();
    let cloud = GaussianCloud {
        means: J::Tensor::from_f32(&[n, 3], &data.means_flat()),
        scales: J::Tensor::from_f32(&[n, 3], &data.scales_flat()),
        quats: J::Tensor::from_f32(&[n, 4], &data.quats_flat()),
        colors: J::Tensor::from_f32(&[n, 3], &data.colors_flat()),
        opacities: J::Tensor::from_f32(&[n], &data.opacities),
    };
    let renderer = chunked_renderer::<J>(jit, n, size, size, chunk);
    Box::new(move |camera: &Camera| renderer(&cloud, camera))
}

fn run_offscreen(
    opts: &Options,
    mut orbit: Orbit,
    render: Renderer,
    frames: usize,
) {
    for frame in 0..frames {
        orbit.yaw = 360.0 * frame as f32 / frames as f32;
        let t0 = std::time::Instant::now();
        let image = render(&orbit.camera(opts.size));
        let path = format!("viewer_frame_{frame:03}.ppm");
        write_ppm(&path, opts.size, opts.size, &image);
        println!(
            "{path}  ({:.1} ms)",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
}

struct App {
    size: usize,
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    render: Renderer,
    orbit: Orbit,
    dragging: bool,
    last_cursor: (f64, f64),
    frame: Option<Vec<f32>>,
}

impl App {
    fn rerender(&mut self) {
        self.frame = Some((self.render)(&self.orbit.camera(self.size)));
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("resin 3DGS viewer")
            .with_inner_size(LogicalSize::new(512.0, 512.0));
        let window = Rc::new(event_loop.create_window(attrs).expect("create window"));
        let context = softbuffer::Context::new(window.clone()).expect("softbuffer context");
        let surface =
            softbuffer::Surface::new(&context, window.clone()).expect("softbuffer surface");
        self.window = Some(window);
        self.surface = Some(surface);
        self.rerender();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.dragging = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);
                if self.dragging {
                    let (lx, ly) = self.last_cursor;
                    self.orbit.yaw += (x - lx) as f32 * 0.5;
                    self.orbit.pitch = (self.orbit.pitch + (y - ly) as f32 * 0.4)
                        .clamp(-85.0, 85.0);
                    self.rerender();
                }
                self.last_cursor = (x, y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.orbit.radius = (self.orbit.radius * (1.0 - amount * 0.1)).max(0.05);
                self.rerender();
            }
            WindowEvent::RedrawRequested => self.present(),
            WindowEvent::Resized(_) => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl App {
    /// Nearest-neighbor blit of the rendered image into the window buffer.
    fn present(&mut self) {
        let (Some(window), Some(surface), Some(frame)) =
            (&self.window, &mut self.surface, &self.frame)
        else {
            return;
        };
        let inner = window.inner_size();
        let (win_w, win_h) = (inner.width.max(1), inner.height.max(1));
        surface
            .resize(
                NonZeroU32::new(win_w).unwrap(),
                NonZeroU32::new(win_h).unwrap(),
            )
            .expect("resize surface");
        let mut buffer = surface.buffer_mut().expect("surface buffer");
        let side = self.size as u32;
        for y in 0..win_h {
            let sy = (y * side / win_h).min(side - 1) as usize;
            for x in 0..win_w {
                let sx = (x * side / win_w).min(side - 1) as usize;
                let at = (sy * self.size + sx) * 3;
                let quantize =
                    |v: f32| -> u32 { (v.clamp(0.0, 1.0) * 255.0).round() as u32 };
                let (r, g, b) = (
                    quantize(frame[at]),
                    quantize(frame[at + 1]),
                    quantize(frame[at + 2]),
                );
                buffer[(y * win_w + x) as usize] = (r << 16) | (g << 8) | b;
            }
        }
        buffer.present().expect("present");
    }
}

fn write_ppm(path: &str, width: usize, height: usize, rgb: &[f32]) {
    use std::io::Write;
    let mut out = Vec::with_capacity(rgb.len() + 32);
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    out.extend(
        rgb.iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8),
    );
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&out))
        .expect("write ppm");
}
