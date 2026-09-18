//! Static glTF import. Decode once, bake node transforms, and publish complete tables.
use crate::{GltfMaterial, GltfScene, GltfTexture, GltfVertex, ResinStatus};
use ::gltf::{buffer, image, mesh::Mode};
use std::{
    ffi::{CStr, c_char, c_void},
    path::Path,
    ptr,
};

pub(crate) fn load(path: &Path) -> Result<GltfScene, ResinStatus> {
    let (document, buffers, images) = ::gltf::import(path).map_err(|_| ResinStatus::IoError)?;
    // No extension is silently interpreted as its incomplete core fallback.
    if document.extensions_required().next().is_some() {
        return Err(ResinStatus::Unsupported);
    }
    let mut result = GltfScene {
        vertices: Vec::new(),
        materials: Vec::new(),
        textures: Vec::new(),
        pixels: Vec::new(),
    };
    for material in document.materials() {
        result.materials.push(decode_material(material)?);
    }
    result.materials.push(GltfMaterial {
        albedo: [1.; 4],
        emissive: [0.; 3],
        metallic: 1.,
        roughness: 1.,
        albedo_texture: u32::MAX,
        metallic_roughness_texture: u32::MAX,
        emissive_texture: u32::MAX,
        normal_texture: u32::MAX,
        normal_scale: 1.,
        alpha_cutoff: 0.5,
        flags: 0,
    });
    for texture in document.textures() {
        let decoded = &images[texture.source().index()];
        let sampler = texture.sampler();
        let first_byte = result.pixels.len() as u64;
        decode_pixels(decoded, &mut result.pixels)?;
        result.textures.push(GltfTexture {
            first_byte,
            width: decoded.width,
            height: decoded.height,
            wrap_s: sampler.wrap_s().as_gl_enum(),
            wrap_t: sampler.wrap_t().as_gl_enum(),
            nearest: u32::from(sampler.mag_filter() == Some(::gltf::texture::MagFilter::Nearest)),
            reserved: 0,
        });
    }
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or(ResinStatus::InvalidArgument)?;
    let identity = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    for node in scene.nodes() {
        append_node(node, identity, &buffers, &mut result, 0)?;
    }
    if result.vertices.is_empty() || result.vertices.len() > u32::MAX as usize {
        return Err(ResinStatus::InvalidArgument);
    }
    Ok(result)
}

fn texture_index(info: Option<::gltf::texture::Info<'_>>) -> Result<u32, ResinStatus> {
    match info {
        Some(info) if info.tex_coord() == 0 => Ok(info.texture().index() as u32),
        Some(_) => Err(ResinStatus::Unsupported),
        None => Ok(u32::MAX),
    }
}
fn decode_material(material: ::gltf::Material<'_>) -> Result<GltfMaterial, ResinStatus> {
    let pbr = material.pbr_metallic_roughness();
    let normal = material.normal_texture();
    if normal
        .as_ref()
        .is_some_and(|texture| texture.tex_coord() != 0)
    {
        return Err(ResinStatus::Unsupported);
    }
    Ok(GltfMaterial {
        albedo: pbr.base_color_factor(),
        emissive: material.emissive_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        albedo_texture: texture_index(pbr.base_color_texture())?,
        metallic_roughness_texture: texture_index(pbr.metallic_roughness_texture())?,
        emissive_texture: texture_index(material.emissive_texture())?,
        normal_texture: normal
            .as_ref()
            .map_or(u32::MAX, |texture| texture.texture().index() as u32),
        normal_scale: normal.map_or(1., |texture| texture.scale()),
        alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
        flags: match material.alpha_mode() {
            ::gltf::material::AlphaMode::Opaque => 0,
            ::gltf::material::AlphaMode::Mask => 1,
            ::gltf::material::AlphaMode::Blend => 2,
        } | if material.double_sided() { 4 } else { 0 },
    })
}

fn decode_pixels(image: &image::Data, output: &mut Vec<u8>) -> Result<(), ResinStatus> {
    use image::Format::*;
    let (channels, bytes) = match image.format {
        R8 => (1, 1),
        R8G8 => (2, 1),
        R8G8B8 => (3, 1),
        R8G8B8A8 => (4, 1),
        R16 => (1, 2),
        R16G16 => (2, 2),
        R16G16B16 => (3, 2),
        R16G16B16A16 => (4, 2),
        R32G32B32FLOAT => (3, 4),
        R32G32B32A32FLOAT => (4, 4),
    };
    let count = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(ResinStatus::InvalidArgument)?;
    if count.checked_mul(channels * bytes) != Some(image.pixels.len()) {
        return Err(ResinStatus::InvalidArgument);
    }
    output
        .try_reserve(count.checked_mul(4).ok_or(ResinStatus::OutOfMemory)?)
        .map_err(|_| ResinStatus::OutOfMemory)?;
    for pixel in image.pixels.chunks_exact(channels * bytes) {
        let mut rgba = [0, 0, 0, 255];
        for (channel, value) in rgba.iter_mut().enumerate().take(channels) {
            let start = channel * bytes;
            *value = match bytes {
                1 => pixel[start],
                2 => (u16::from_ne_bytes([pixel[start], pixel[start + 1]]) / 257) as u8,
                _ => (f32::from_ne_bytes(pixel[start..start + 4].try_into().unwrap()).clamp(0., 1.)
                    * 255.)
                    .round() as u8,
            };
        }
        if channels <= 2 {
            rgba[2] = rgba[0];
            let alpha = rgba[1];
            rgba[1] = rgba[0];
            if channels == 2 {
                rgba[3] = alpha;
            }
        }
        output.extend_from_slice(&rgba);
    }
    Ok(())
}

type Matrix = [[f32; 4]; 4]; // glTF column-major.
fn product(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|column| {
        std::array::from_fn(|row| (0..4).map(|k| a[k][row] * b[column][k]).sum())
    })
}
fn point(m: Matrix, p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
}
fn direction(m: Matrix, p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2])
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn normalized(a: [f32; 3]) -> [f32; 3] {
    let length = dot(a, a).sqrt();
    if length > 1e-12 {
        a.map(|x| x / length)
    } else {
        [0.; 3]
    }
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn normal(m: Matrix, n: [f32; 3]) -> [f32; 3] {
    let a = [m[0][0], m[0][1], m[0][2]];
    let b = [m[1][0], m[1][1], m[1][2]];
    let c = [m[2][0], m[2][1], m[2][2]];
    let x = cross(b, c);
    let y = cross(c, a);
    let z = cross(a, b);
    let sign = dot(a, x).signum();
    normalized(std::array::from_fn(|i| {
        (x[i] * n[0] + y[i] * n[1] + z[i] * n[2]) * sign
    }))
}
fn append_node(
    node: ::gltf::Node<'_>,
    parent: Matrix,
    buffers: &[buffer::Data],
    scene: &mut GltfScene,
    depth: usize,
) -> Result<(), ResinStatus> {
    if depth > 256 || node.skin().is_some() {
        return Err(ResinStatus::Unsupported);
    }
    let transform = product(parent, node.transform().matrix());
    if !transform.iter().flatten().all(|v| v.is_finite()) {
        return Err(ResinStatus::InvalidArgument);
    }
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            append_primitive(primitive, transform, buffers, scene)?;
        }
    }
    for child in node.children() {
        append_node(child, transform, buffers, scene, depth + 1)?;
    }
    Ok(())
}
fn append_primitive(
    primitive: ::gltf::Primitive<'_>,
    transform: Matrix,
    buffers: &[buffer::Data],
    scene: &mut GltfScene,
) -> Result<(), ResinStatus> {
    if primitive.morph_targets().next().is_some() {
        return Err(ResinStatus::Unsupported);
    }
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let positions: Vec<_> = reader
        .read_positions()
        .ok_or(ResinStatus::InvalidArgument)?
        .collect();
    let normals: Option<Vec<_>> = reader.read_normals().map(Iterator::collect);
    let uv: Option<Vec<_>> = reader
        .read_tex_coords(0)
        .map(|data| data.into_f32().collect());
    let tangents: Option<Vec<_>> = reader.read_tangents().map(Iterator::collect);
    let colors: Option<Vec<_>> = reader
        .read_colors(0)
        .map(|data| data.into_rgba_f32().collect());
    if [
        normals.as_ref().map(Vec::len),
        uv.as_ref().map(Vec::len),
        tangents.as_ref().map(Vec::len),
        colors.as_ref().map(Vec::len),
    ]
    .into_iter()
    .flatten()
    .any(|n| n != positions.len())
    {
        return Err(ResinStatus::InvalidArgument);
    }
    let indices: Vec<_> = reader
        .read_indices()
        .map(|data| data.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());
    if indices.iter().any(|&i| i as usize >= positions.len()) {
        return Err(ResinStatus::InvalidArgument);
    }
    let mut triangles = Vec::new();
    match primitive.mode() {
        Mode::Triangles if indices.len().is_multiple_of(3) => triangles = indices,
        Mode::TriangleStrip => {
            for i in 2..indices.len() {
                triangles.extend_from_slice(&if i % 2 == 0 {
                    [indices[i - 2], indices[i - 1], indices[i]]
                } else {
                    [indices[i - 1], indices[i - 2], indices[i]]
                });
            }
        }
        Mode::TriangleFan => {
            for i in 2..indices.len() {
                triangles.extend_from_slice(&[indices[0], indices[i - 1], indices[i]]);
            }
        }
        _ => return Err(ResinStatus::Unsupported),
    }
    let material = primitive
        .material()
        .index()
        .unwrap_or(scene.materials.len() - 1) as u32;
    let start = scene.vertices.len();
    for triangle in triangles.chunks_exact(3) {
        let face = normalized(cross(
            sub(
                positions[triangle[1] as usize],
                positions[triangle[0] as usize],
            ),
            sub(
                positions[triangle[2] as usize],
                positions[triangle[0] as usize],
            ),
        ));
        for &index in triangle {
            let i = index as usize;
            scene.vertices.push(GltfVertex {
                position: positions[i],
                normal: normals.as_ref().map_or(face, |n| n[i]),
                uv: uv.as_ref().map_or([0.; 2], |uv| uv[i]),
                tangent: tangents.as_ref().map_or([1., 0., 0., 1.], |t| t[i]),
                color: colors.as_ref().map_or([1.; 4], |c| c[i]),
                material,
            });
        }
    }
    let vertices = &mut scene.vertices[start..];
    if tangents.is_none() && uv.is_some() {
        mikktspace::generate_tangents(&mut Tangents(vertices));
    }
    let determinant = dot(
        [transform[0][0], transform[0][1], transform[0][2]],
        cross(
            [transform[1][0], transform[1][1], transform[1][2]],
            [transform[2][0], transform[2][1], transform[2][2]],
        ),
    );
    if determinant.abs() < 1e-12 {
        return Err(ResinStatus::InvalidArgument);
    }
    for vertex in vertices.iter_mut() {
        vertex.position = point(transform, vertex.position);
        vertex.normal = normal(transform, vertex.normal);
        let tangent = normalized(direction(
            transform,
            [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]],
        ));
        vertex.tangent = [
            tangent[0],
            tangent[1],
            tangent[2],
            vertex.tangent[3] * determinant.signum(),
        ];
        if !vertex
            .position
            .iter()
            .chain(&vertex.normal)
            .chain(&vertex.uv)
            .chain(&vertex.color)
            .chain(&vertex.tangent)
            .all(|v| v.is_finite())
        {
            return Err(ResinStatus::InvalidArgument);
        }
    }
    if determinant < 0. {
        for triangle in vertices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }
    Ok(())
}
struct Tangents<'a>(&'a mut [GltfVertex]);
impl mikktspace::Geometry for Tangents<'_> {
    fn num_faces(&self) -> usize {
        self.0.len() / 3
    }
    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }
    fn position(&self, f: usize, v: usize) -> [f32; 3] {
        self.0[f * 3 + v].position
    }
    fn normal(&self, f: usize, v: usize) -> [f32; 3] {
        self.0[f * 3 + v].normal
    }
    fn tex_coord(&self, f: usize, v: usize) -> [f32; 2] {
        self.0[f * 3 + v].uv
    }
    fn set_tangent_encoded(&mut self, t: [f32; 4], f: usize, v: usize) {
        self.0[f * 3 + v].tangent = t;
    }
}

/// # Safety
/// `path` is a NUL-terminated string and `output` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gltf_load(
    path: *const c_char,
    output: *mut *mut GltfScene,
) -> ResinStatus {
    if output.is_null() {
        return ResinStatus::InvalidArgument;
    }
    unsafe {
        *output = ptr::null_mut();
    }
    if path.is_null() {
        return ResinStatus::InvalidArgument;
    }
    let Ok(path) = (unsafe { CStr::from_ptr(path) }).to_str() else {
        return ResinStatus::InvalidArgument;
    };
    match load(Path::new(path)) {
        Ok(scene) => {
            unsafe {
                *output = Box::into_raw(Box::new(scene));
            }
            ResinStatus::Success
        }
        Err(error) => error,
    }
}
/// # Safety
/// `scene` is null or a live result of resin_gltf_load, released exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gltf_free(scene: *mut GltfScene) {
    if !scene.is_null() {
        drop(unsafe { Box::from_raw(scene) });
    }
}
/// Borrow an immutable table: 0 vertices, 1 materials, 2 textures, 3 pixel bytes.
/// # Safety
/// Scene is live and length is writable. The pointer expires with the scene.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gltf_table(
    scene: *const GltfScene,
    table: u32,
    length: *mut usize,
) -> *const c_void {
    if !length.is_null() {
        unsafe {
            *length = 0;
        }
    }
    let Some(scene) = (unsafe { scene.as_ref() }) else {
        return ptr::null();
    };
    let (data, count) = match table {
        0 => (scene.vertices.as_ptr().cast(), scene.vertices.len()),
        1 => (scene.materials.as_ptr().cast(), scene.materials.len()),
        2 => (scene.textures.as_ptr().cast(), scene.textures.len()),
        3 => (scene.pixels.as_ptr().cast(), scene.pixels.len()),
        _ => return ptr::null(),
    };
    if !length.is_null() {
        unsafe {
            *length = count;
        }
    }
    data
}

/// # Safety
/// Scene is live. The returned length counts table elements, not bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_gltf_count(scene: *const GltfScene, table: u32) -> usize {
    let mut length = 0;
    unsafe {
        resin_gltf_table(scene, table, &mut length);
    }
    length
}
