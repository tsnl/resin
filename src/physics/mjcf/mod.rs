//! Loading MuJoCo MJCF models into a [`Scene`].
//!
//! MJCF describes a robot as a tree of bodies, each posed relative to its
//! parent, with geoms (shapes), an optional joint connecting it to the
//! parent, and defaults inherited through named classes. This loader walks
//! that tree, accumulates world poses, and emits maximal-coordinate bodies
//! and joints for the solver:
//!
//! - `<freejoint/>` → an unconstrained body;
//! - `<joint type="hinge">` → [`Joint::Hinge`] (anchor and axis mapped from
//!   the child frame into the parent frame);
//! - `<joint type="ball">` → [`Joint::Ball`];
//! - no joint at all → a rigid weld: [`Joint::Ball`] plus two
//!   [`Joint::AxisAlign`]s pinning the relative orientation.
//!
//! MuJoCo worlds are z-up; this engine is y-up. All world-frame quantities
//! are rotated by −90° about x on the way in (`(x, y, z) → (x, z, −y)`);
//! body-local data (anchors, axes, geom sizes) is frame-relative and passes
//! through unchanged.
//!
//! The supported subset is what the solver can express: sphere and box
//! geoms, one joint per body, a ground plane at z = 0. Anything else —
//! meshes, capsules, slide joints, actuator dynamics, contact overrides —
//! fails with a clear message listing the unsupported feature, so a model
//! either loads faithfully or not at all. Notes about *harmlessly* ignored
//! attributes (damping, ranges, appearance) are collected in
//! [`Loaded::notes`] instead of failing.

#[cfg(test)]
mod parity;
pub mod xml;

use std::collections::HashMap;
use std::path::Path;

use crate::physics::scene::{Body, Joint, Motor, Scene, Shape, WORLD};
use xml::Element;

/// A loaded model: the scene plus the name → body-row map and actuator list
/// needed to drive and observe it.
pub struct Loaded {
    pub scene: Scene,
    /// MJCF body names to rows in the scene (and in `State` tensors).
    pub body_ids: HashMap<String, usize>,
    /// `<actuator><motor>` entries, in `scene.motors` order — the names and
    /// control ranges a controller needs on top of the physical model.
    pub actuators: Vec<Actuator>,
    /// Features that were ignored without changing the simulation contract.
    pub notes: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Actuator {
    pub name: String,
    pub joint: usize,
    pub gear: f32,
    pub ctrl_range: [f32; 2],
}

/// Load a model from a file, resolving `<include>` elements relative to it.
pub fn load_file(path: impl AsRef<Path>) -> Result<Loaded, String> {
    let root = parse_file(path.as_ref())?;
    build(&root)
}

/// Load a model from a string (no `<include>` resolution).
pub fn load_str(text: &str) -> Result<Loaded, String> {
    build(&xml::parse(text)?)
}

/// Parse a file to raw XML with `<include>`s spliced in — the structural
/// half of loading, usable on models the solver cannot simulate.
pub fn parse_file(path: &Path) -> Result<Element, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut root = xml::parse(&text)?;
    let dir = path.parent().unwrap_or(Path::new("."));
    splice_includes(&mut root, dir)?;
    Ok(root)
}

/// Replace each `<include file="…"/>` with the included document's children
/// (MuJoCo pastes the file's content where the include stood).
fn splice_includes(element: &mut Element, dir: &Path) -> Result<(), String> {
    let mut spliced = Vec::with_capacity(element.children.len());
    for child in element.children.drain(..) {
        if child.name == "include" {
            let file = child.attr("file").ok_or("include without file=")?;
            let included = parse_file(&dir.join(file))?;
            spliced.extend(included.children);
        } else {
            spliced.push(child);
        }
    }
    element.children = spliced;
    for child in &mut element.children {
        splice_includes(child, dir)?;
    }
    Ok(())
}

//
// Defaults: MJCF elements inherit attributes from named <default> classes.
//

/// `classes["go2"]["joint"]` holds the default joint attributes of class
/// "go2". Nested classes are flattened at parse time: a child class starts
/// from its parent's resolved attributes.
type Defaults = HashMap<String, HashMap<String, Vec<(String, String)>>>;

fn collect_defaults(root: &Element) -> Defaults {
    let mut classes = Defaults::new();
    for defaults in root.children_named("default") {
        walk_defaults(defaults, "main", &mut classes);
    }
    classes
}

fn walk_defaults(element: &Element, parent: &str, classes: &mut Defaults) {
    let class = element.attr("class").unwrap_or(parent).to_string();
    let mut resolved = classes.get(parent).cloned().unwrap_or_default();
    for child in &element.children {
        if child.name != "default" {
            let slot = resolved.entry(child.name.clone()).or_default();
            for (key, value) in &child.attrs {
                slot.retain(|(k, _)| k != key);
                slot.push((key.clone(), value.clone()));
            }
        }
    }
    classes.insert(class.clone(), resolved);
    for child in element.children_named("default") {
        walk_defaults(child, &class, classes);
    }
}

/// Attribute lookup order: the element itself, then its class (or the
/// enclosing body tree's childclass), then the global "main" class.
fn lookup<'a>(
    element: &'a Element,
    childclass: &str,
    defaults: &'a Defaults,
    attr: &str,
) -> Option<&'a str> {
    if let Some(value) = element.attr(attr) {
        return Some(value);
    }
    let class = element.attr("class").unwrap_or(childclass);
    for name in [class, "main"] {
        if let Some(value) = defaults
            .get(name)
            .and_then(|kinds| kinds.get(&element.name))
            .and_then(|attrs| attrs.iter().find(|(key, _)| key == attr))
        {
            return Some(&value.1);
        }
    }
    None
}

//
// Building the scene.
//

struct Builder {
    scene: Scene,
    body_ids: HashMap<String, usize>,
    joint_names: HashMap<String, usize>,
    notes: Vec<String>,
    defaults: Defaults,
    /// Angles are degrees unless `<compiler angle="radian">`.
    radians: bool,
}

fn build(root: &Element) -> Result<Loaded, String> {
    if root.name != "mujoco" {
        return Err(format!("expected <mujoco> root, found <{}>", root.name));
    }
    let mut builder = Builder {
        scene: Scene::new(),
        body_ids: HashMap::new(),
        joint_names: HashMap::new(),
        notes: Vec::new(),
        defaults: collect_defaults(root),
        radians: false,
    };
    builder.scene.ground = false;

    for compiler in root.children_named("compiler") {
        match compiler.attr("angle") {
            Some("radian") => builder.radians = true,
            Some("degree") | None => builder.radians = false,
            Some(other) => return Err(format!("unsupported compiler angle={other}")),
        }
        if compiler.attr("coordinate") == Some("global") {
            return Err("global coordinates are unsupported".into());
        }
    }
    for option in root.children_named("option") {
        if let Some(timestep) = option.attr("timestep") {
            builder.scene.dt = parse_num(timestep)?;
        }
        if let Some(gravity) = option.attr("gravity") {
            builder.scene.gravity = to_y_up(parse_vec3(gravity)?);
        }
    }
    // One MJCF timestep = one frame here; the default 4 internal substeps
    // keep XPBD's O(h) discretization error well under MuJoCo's at the same
    // dt (halve/double substeps and the deviation follows suit).

    for worldbody in root.children_named("worldbody") {
        for geom in worldbody.children_named("geom") {
            builder.world_geom(geom)?;
        }
        for body in worldbody.children_named("body") {
            builder.body(body, WORLD, Pose::identity(), "main")?;
        }
    }

    let mut actuators = Vec::new();
    for section in root.children_named("actuator") {
        for motor in &section.children {
            let actuator = builder.actuator(motor)?;
            builder.scene.motors.push(Motor {
                joint: actuator.joint,
                gear: actuator.gear,
            });
            actuators.push(actuator);
        }
    }

    for section in ["tendon", "equality", "sensor", "contact"] {
        if root.child(section).is_some() {
            return Err(format!("<{section}> is unsupported"));
        }
    }

    Ok(Loaded {
        scene: builder.scene,
        body_ids: builder.body_ids,
        actuators,
        notes: builder.notes,
    })
}

/// A world pose: rotation quaternion `(w, x, y, z)` and translation.
#[derive(Clone, Copy)]
struct Pose {
    quat: [f32; 4],
    pos: [f32; 3],
}

impl Pose {
    fn identity() -> Pose {
        Pose {
            quat: [1.0, 0.0, 0.0, 0.0],
            pos: [0.0; 3],
        }
    }

    /// This pose followed by a child pose given in this pose's frame.
    fn then(&self, child: Pose) -> Pose {
        Pose {
            quat: quat_mul(self.quat, child.quat),
            pos: add3(self.pos, rotate(self.quat, child.pos)),
        }
    }

    fn point_to_world(&self, local: [f32; 3]) -> [f32; 3] {
        add3(self.pos, rotate(self.quat, local))
    }
}

impl Builder {
    /// Recursively add a `<body>` and its subtree. Poses accumulate in
    /// MuJoCo's z-up world frame and convert to y-up only when written into
    /// the scene, so all local math stays in MuJoCo's own frames.
    /// `parent_pose` is the parent's pose in the MuJoCo world; `parent` is
    /// the row to joint against.
    fn body(
        &mut self,
        element: &Element,
        parent: usize,
        parent_pose: Pose,
        childclass: &str,
    ) -> Result<(), String> {
        let childclass = element.attr("childclass").unwrap_or(childclass).to_string();
        let local = Pose {
            pos: parse_vec3(element.attr("pos").unwrap_or("0 0 0"))?,
            quat: self.orientation(element)?,
        };
        let pose = parent_pose.then(local);

        let (shape, mass) = self.shape_and_mass(element, &childclass)?;
        let row = self.scene.add(Body {
            shape,
            mass,
            position: to_y_up(pose.pos),
            orientation: quat_mul(Y_UP, pose.quat),
            ..Body::default()
        });
        let name = element
            .attr("name")
            .map(str::to_string)
            .unwrap_or_else(|| format!("body{row}"));
        self.body_ids.insert(name, row);

        self.joint(element, parent, row, local, pose, &childclass)?;

        for child in element.children_named("body") {
            self.body(child, row, pose, &childclass)?;
        }
        Ok(())
    }

    /// Emit the joint between `child` and `parent` — a point/axis pair
    /// expressed in each side's local frame. For a body parented to the
    /// static world, "parent local" is the engine's world frame, so MuJoCo
    /// world coordinates convert to y-up on that side only.
    fn joint(
        &mut self,
        element: &Element,
        parent: usize,
        child: usize,
        local: Pose,
        pose: Pose,
        childclass: &str,
    ) -> Result<(), String> {
        let in_parent_point = |p: [f32; 3]| {
            if parent == WORLD {
                to_y_up(pose.point_to_world(p))
            } else {
                local.point_to_world(p)
            }
        };
        let in_parent_axis = |v: [f32; 3]| {
            if parent == WORLD {
                to_y_up(rotate(pose.quat, v))
            } else {
                rotate(local.quat, v)
            }
        };

        let mut joints = element.children_named("joint").collect::<Vec<_>>();
        joints.extend(element.children_named("freejoint"));
        if joints.len() > 1 {
            return Err("multiple joints on one body are unsupported".into());
        }

        let Some(joint) = joints.first() else {
            // No joint: welded to the parent. A ball joint pins the child's
            // origin; two axis alignments pin its relative orientation.
            self.scene.joints.push(Joint::Ball {
                a: parent,
                b: child,
                anchor_a: in_parent_point([0.0; 3]),
                anchor_b: [0.0; 3],
            });
            for axis in [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
                self.scene.joints.push(Joint::AxisAlign {
                    a: parent,
                    b: child,
                    axis_a: in_parent_axis(axis),
                    axis_b: axis,
                });
            }
            return Ok(());
        };

        if let Some(name) = joint.attr("name") {
            self.joint_names
                .insert(name.to_string(), self.scene.joints.len());
        }
        for ignored in ["range", "damping", "armature", "frictionloss", "stiffness"] {
            if lookup(joint, childclass, &self.defaults, ignored).is_some() {
                self.note(format!("joint {ignored} is ignored"));
            }
        }

        let kind = if joint.name == "freejoint" {
            "free"
        } else {
            lookup(joint, childclass, &self.defaults, "type").unwrap_or("hinge")
        };
        let anchor = parse_vec3(joint.attr("pos").unwrap_or("0 0 0"))?;
        match kind {
            "free" => {} // unconstrained: exactly our default
            "ball" => self.scene.joints.push(Joint::Ball {
                a: parent,
                b: child,
                anchor_a: in_parent_point(anchor),
                anchor_b: anchor,
            }),
            "hinge" => {
                let axis = parse_vec3(
                    lookup(joint, childclass, &self.defaults, "axis").unwrap_or("0 0 1"),
                )?;
                self.scene.joints.push(Joint::Hinge {
                    a: parent,
                    b: child,
                    anchor_a: in_parent_point(anchor),
                    anchor_b: anchor,
                    axis_a: in_parent_axis(axis),
                    axis_b: axis,
                });
            }
            other => return Err(format!("joint type {other:?} is unsupported")),
        }
        Ok(())
    }

    /// Body orientation from `quat` or `euler` (intrinsic x-y-z, matching
    /// MuJoCo's default `eulerseq`).
    fn orientation(&self, element: &Element) -> Result<[f32; 4], String> {
        match (element.attr("quat"), element.attr("euler")) {
            (None, None) => Ok([1.0, 0.0, 0.0, 0.0]),
            (Some(_), Some(_)) => Err("body has both quat and euler".into()),
            (Some(quat), None) => {
                let q: [f32; 4] = parse_floats(quat)?
                    .try_into()
                    .map_err(|_| format!("expected 4 numbers in quat {quat:?}"))?;
                Ok(normalize4(q))
            }
            (None, Some(euler)) => {
                let scale = if self.radians {
                    1.0
                } else {
                    std::f32::consts::PI / 180.0
                };
                let [rx, ry, rz] = parse_vec3(euler)?.map(|a| a * scale);
                let qx = [(rx / 2.0).cos(), (rx / 2.0).sin(), 0.0, 0.0];
                let qy = [(ry / 2.0).cos(), 0.0, (ry / 2.0).sin(), 0.0];
                let qz = [(rz / 2.0).cos(), 0.0, 0.0, (rz / 2.0).sin()];
                Ok(quat_mul(quat_mul(qx, qy), qz))
            }
        }
    }

    /// The body's collision shape and mass. One primitive geom per body;
    /// `<inertial mass>` overrides the geom-derived mass (its diagonal
    /// inertia is still recomputed from the shape).
    fn shape_and_mass(
        &mut self,
        element: &Element,
        childclass: &str,
    ) -> Result<(Shape, f32), String> {
        let geoms: Vec<&Element> = element.children_named("geom").collect();
        if geoms.len() > 1 {
            return Err("multiple geoms on one body are unsupported".into());
        }
        let Some(geom) = geoms.first() else {
            return Err("body without a geom is unsupported".into());
        };

        if parse_vec3(lookup(geom, childclass, &self.defaults, "pos").unwrap_or("0 0 0"))?
            != [0.0; 3]
        {
            return Err("geom pos offsets are unsupported".into());
        }
        let sizes = parse_floats(lookup(geom, childclass, &self.defaults, "size").unwrap_or(""))?;
        let shape = match lookup(geom, childclass, &self.defaults, "type").unwrap_or("sphere") {
            "sphere" => Shape::Sphere {
                radius: *sizes.first().ok_or("sphere needs a size")?,
            },
            "box" => Shape::Cuboid {
                half_extents: sizes
                    .clone()
                    .try_into()
                    .map_err(|_| "box needs 3 sizes".to_string())?,
            },
            other => return Err(format!("geom type {other:?} is unsupported")),
        };

        let volume = match shape {
            Shape::Sphere { radius } => 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3),
            Shape::Cuboid {
                half_extents: [x, y, z],
            } => 8.0 * x * y * z,
        };
        let geom_mass = match lookup(geom, childclass, &self.defaults, "mass") {
            Some(mass) => parse_num(mass)?,
            None => {
                let density = parse_num(
                    lookup(geom, childclass, &self.defaults, "density").unwrap_or("1000"),
                )?;
                density * volume
            }
        };
        let mass = match element.child("inertial") {
            Some(inertial) => {
                self.note("inertial diaginertia is recomputed from the geom".into());
                parse_num(inertial.attr("mass").ok_or("inertial without mass")?)?
            }
            None => geom_mass,
        };
        Ok((shape, mass))
    }

    /// Geoms directly in `<worldbody>`: only the ground plane at z = 0.
    fn world_geom(&mut self, geom: &Element) -> Result<(), String> {
        if lookup(geom, "main", &self.defaults, "type") != Some("plane") {
            return Err("only a plane geom is supported directly in worldbody".into());
        }
        let pos = parse_vec3(geom.attr("pos").unwrap_or("0 0 0"))?;
        if pos[2] != 0.0 || geom.attr("quat").is_some() || geom.attr("euler").is_some() {
            return Err("only the plane z = 0 with default orientation is supported".into());
        }
        self.scene.ground = true;
        Ok(())
    }

    fn actuator(&mut self, motor: &Element) -> Result<Actuator, String> {
        if motor.name != "motor" {
            return Err(format!(
                "actuator <{}> is unsupported (only <motor>)",
                motor.name
            ));
        }
        let joint_name = motor.attr("joint").ok_or("motor without joint=")?;
        let joint = *self
            .joint_names
            .get(joint_name)
            .ok_or_else(|| format!("motor targets unknown joint {joint_name:?}"))?;
        if !matches!(self.scene.joints[joint], Joint::Hinge { .. }) {
            return Err(format!(
                "motor on non-hinge joint {joint_name:?} is unsupported"
            ));
        }
        let gear = parse_num(
            motor
                .attr("gear")
                .unwrap_or("1")
                .split_whitespace()
                .next()
                .unwrap_or("1"),
        )?;
        let ctrl_range = match motor.attr("ctrlrange") {
            Some(range) => {
                let values = parse_floats(range)?;
                [values[0], values[1]]
            }
            None => [-1.0, 1.0],
        };
        Ok(Actuator {
            name: motor.attr("name").unwrap_or(joint_name).to_string(),
            joint,
            gear,
            ctrl_range,
        })
    }

    fn note(&mut self, message: String) {
        if !self.notes.contains(&message) {
            self.notes.push(message);
        }
    }
}

const Y_UP: [f32; 4] = [
    std::f32::consts::FRAC_1_SQRT_2,
    -std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    0.0,
];

fn to_y_up(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[2], -v[1]]
}

fn parse_num(text: &str) -> Result<f32, String> {
    text.trim()
        .parse()
        .map_err(|_| format!("expected a number, got {text:?}"))
}

fn parse_vec3(text: &str) -> Result<[f32; 3], String> {
    let values = parse_floats(text)?;
    values
        .try_into()
        .map_err(|_| format!("expected 3 numbers, got {text:?}"))
}

fn parse_floats(text: &str) -> Result<Vec<f32>, String> {
    text.split_whitespace()
        .map(|field| field.parse().map_err(|_| format!("bad number in {text:?}")))
        .collect()
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn normalize4(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    q.map(|c| c / len)
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn loads_a_falling_sphere_with_ground() {
        let loaded = load_str(
            r#"<mujoco>
                 <option timestep="0.004" gravity="0 0 -10"/>
                 <worldbody>
                   <geom type="plane" size="10 10 1"/>
                   <body name="ball" pos="1 2 3">
                     <freejoint/>
                     <geom type="sphere" size="0.5" mass="2"/>
                   </body>
                 </worldbody>
               </mujoco>"#,
        )
        .unwrap();

        let scene = &loaded.scene;
        assert!(scene.ground);
        assert_eq!(scene.dt, 0.004);
        assert!(
            close3(scene.gravity, [0.0, -10.0, 0.0]),
            "{:?}",
            scene.gravity
        );
        assert!(scene.joints.is_empty(), "freejoint adds no constraint");

        let ball = loaded.body_ids["ball"];
        let body = &scene.bodies[ball];
        // MuJoCo (1, 2, 3) is y-up (1, 3, -2).
        assert!(
            close3(body.position, [1.0, 3.0, -2.0]),
            "{:?}",
            body.position
        );
        assert_eq!(body.mass, 2.0);
        assert!(matches!(body.shape, Shape::Sphere { radius } if radius == 0.5));
    }

    #[test]
    fn maps_a_hinge_pendulum_into_parent_and_child_frames() {
        // A rod hanging from the world, hinged at its top end, like
        // MuJoCo's classic pendulum: the joint sits 0.3 above the body
        // frame, on the world z-axis.
        let loaded = load_str(
            r#"<mujoco>
                 <worldbody>
                   <body name="rod" pos="0 0 0.7">
                     <joint type="hinge" pos="0 0 0.3" axis="0 1 0"/>
                     <geom type="box" size="0.05 0.05 0.3"/>
                   </body>
                 </worldbody>
               </mujoco>"#,
        )
        .unwrap();

        assert_eq!(loaded.scene.joints.len(), 1);
        let Joint::Hinge {
            a,
            b,
            anchor_a,
            anchor_b,
            axis_a,
            axis_b,
        } = loaded.scene.joints[0]
        else {
            panic!("expected a hinge");
        };
        assert_eq!(a, WORLD);
        assert_eq!(b, loaded.body_ids["rod"]);
        // Child-local anchor and axis stay in MuJoCo's frame…
        assert!(close3(anchor_b, [0.0, 0.0, 0.3]));
        assert!(close3(axis_b, [0.0, 1.0, 0.0]));
        // …the world side converts to y-up: (0, 0, 1.0) → (0, 1.0, 0).
        assert!(close3(anchor_a, [0.0, 1.0, 0.0]), "{anchor_a:?}");
        assert!(close3(axis_a, [0.0, 0.0, -1.0]), "{axis_a:?}");
    }

    #[test]
    fn jointless_bodies_become_welds() {
        let loaded = load_str(
            r#"<mujoco>
                 <worldbody>
                   <body name="base" pos="0 0 1">
                     <freejoint/>
                     <geom type="box" size="0.2 0.2 0.2"/>
                     <body name="bump" pos="0 0 0.3">
                       <geom type="sphere" size="0.1"/>
                     </body>
                   </body>
                 </worldbody>
               </mujoco>"#,
        )
        .unwrap();
        // Weld = ball + two axis alignments.
        assert_eq!(loaded.scene.joints.len(), 3);
        assert!(matches!(loaded.scene.joints[0], Joint::Ball { .. }));
        assert!(matches!(loaded.scene.joints[1], Joint::AxisAlign { .. }));
        assert!(matches!(loaded.scene.joints[2], Joint::AxisAlign { .. }));
    }

    #[test]
    fn defaults_resolve_through_class_chains() {
        let loaded = load_str(
            r#"<mujoco>
                 <default>
                   <geom density="500"/>
                   <default class="small">
                     <geom type="sphere" size="0.1"/>
                   </default>
                 </default>
                 <worldbody>
                   <body name="b" pos="0 0 1">
                     <freejoint/>
                     <geom class="small"/>
                   </body>
                 </worldbody>
               </mujoco>"#,
        )
        .unwrap();
        let body = &loaded.scene.bodies[loaded.body_ids["b"]];
        assert!(matches!(body.shape, Shape::Sphere { radius } if radius == 0.1));
        // density 500 comes from the parent class.
        let expected = 500.0 * 4.0 / 3.0 * std::f32::consts::PI * 0.1f32.powi(3);
        assert!((body.mass - expected).abs() < 1e-5, "mass {}", body.mass);
    }

    #[test]
    fn unsupported_features_fail_with_clear_messages() {
        let slide = r#"<mujoco><worldbody><body><joint type="slide"/>
            <geom type="sphere" size="0.1"/></body></worldbody></mujoco>"#;
        assert!(load_str(slide).err().unwrap().contains("slide"));

        let mesh = r#"<mujoco><worldbody><body><freejoint/>
            <geom type="mesh" mesh="m"/></body></worldbody></mujoco>"#;
        assert!(load_str(mesh).err().unwrap().contains("mesh"));
    }

    /// Every model in the MuJoCo Menagerie must either load or fail with a
    /// clean diagnostic — never a panic. (Skips when the submodule is not
    /// checked out.)
    #[test]
    fn menagerie_models_parse_or_fail_cleanly() {
        let menagerie = Path::new("third_party/mujoco_menagerie");
        if !menagerie.is_dir() {
            eprintln!("skip: mujoco_menagerie submodule not checked out");
            return;
        }
        let (mut parsed, mut loaded_count) = (0, 0);
        for entry in std::fs::read_dir(menagerie).unwrap() {
            let dir = entry.unwrap().path();
            if !dir.is_dir() {
                continue;
            }
            for file in std::fs::read_dir(&dir).unwrap() {
                let path = file.unwrap().path();
                if path.extension().is_none_or(|e| e != "xml")
                    || path.file_name().is_some_and(|n| {
                        // Keyframe-only and scene files include the model files.
                        n.to_string_lossy().ends_with("keyframes.xml")
                    })
                {
                    continue;
                }
                // The structural half must always succeed…
                let root =
                    parse_file(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                assert_eq!(root.name, "mujoco", "{}", path.display());
                parsed += 1;
                // …and full loading either works or explains itself.
                if build(&root).is_ok() {
                    loaded_count += 1;
                }
            }
        }
        assert!(parsed > 100, "only parsed {parsed} menagerie files");
        eprintln!("parsed {parsed} menagerie XMLs structurally; {loaded_count} fully loadable");
    }
}
