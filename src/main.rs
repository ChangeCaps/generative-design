use bevy::prelude::*;
use std::ops::Range;

mod index_table;

pub struct Point {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Point {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl From<(i32, i32, i32)> for Point {
    fn from(tuple: (i32, i32, i32)) -> Self {
        Point::new(tuple.0, tuple.1, tuple.2)
    }
}

pub struct DensityField {
    samples: Vec<f32>,

    x_bound: Range<i32>,
    y_bound: Range<i32>,
    z_bound: Range<i32>,

    x_size: usize,
    y_size: usize,
    z_size: usize,

    sample_density: f32,
}

impl<T: Into<Point>> std::ops::Index<T> for DensityField {
    type Output = f32;

    fn index(&self, point: T) -> &Self::Output {
        let point = point.into();
        let x = (point.x - self.x_bound.start) as usize;
        let y = (point.y - self.y_bound.start) as usize;
        let z = (point.z - self.z_bound.start) as usize;

        &self.samples[y * self.x_size * self.z_size + x * self.z_size + z]
    }
}

impl<T: Into<Point>> std::ops::IndexMut<T> for DensityField {
    fn index_mut(&mut self, point: T) -> &mut Self::Output {
        let point = point.into();
        let x = (point.x - self.x_bound.start) as usize;
        let y = (point.y - self.y_bound.start) as usize;
        let z = (point.z - self.z_bound.start) as usize;

        &mut self.samples[y * self.x_size * self.z_size + x * self.z_size + z]
    }
}

impl DensityField {
    pub fn empty(
        x_bound: Range<i32>,
        y_bound: Range<i32>,
        z_bound: Range<i32>,
        sample_density: f32,
    ) -> Self {
        let x_size = (x_bound.end - x_bound.start) as usize;
        let y_size = (y_bound.end - y_bound.start) as usize;
        let z_size = (z_bound.end - z_bound.start) as usize;

        DensityField {
            samples: vec![0.0; x_size * y_size * z_size],

            x_bound,
            y_bound,
            z_bound,

            x_size,
            y_size,
            z_size,

            sample_density,
        }
    }

    pub fn set_density(&mut self, setter: fn(Vec3) -> f32) -> &mut Self {
        for x in self.x_bound.clone() {
            for y in self.y_bound.clone() {
                for z in self.z_bound.clone() {
                    let pos = Vec3::new(x as f32, y as f32, z as f32) * self.sample_density;

                    self[(x, y, z)] = setter(pos);
                }
            }
        }

        self
    }

    pub fn generate_mesh(&self, iso_level: f32) -> Mesh {
        let start_instant = std::time::Instant::now();

        fn vert_interpret(v0: Vec3, v1: Vec3, d0: f32, d1: f32, iso_level: f32) -> Vec3 {
            if (iso_level - d0).abs() < 0.0001 {
                return v0;
            }
            if (iso_level - d1).abs() < 0.0001 {
                return v1;
            }
            if (d0 - d1).abs() < 0.0001 {
                return v0;
            }

            let mu = (iso_level - d0) / (d1 - d0);
            v0 + (v1 - v0) * mu
        }

        let mut mesh = Mesh::new(bevy::render::pipeline::PrimitiveTopology::TriangleList);

        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut num_normals: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for x in self.x_bound.start..self.x_bound.end - 1 {
            for y in self.y_bound.start..self.y_bound.end - 1 {
                for z in self.z_bound.start..self.z_bound.end - 1 {
                    let grid_points = [
                        (x, y, z),
                        (x + 1, y, z),
                        (x + 1, y + 1, z),
                        (x, y + 1, z),
                        (x, y, z + 1),
                        (x + 1, y, z + 1),
                        (x + 1, y + 1, z + 1),
                        (x, y + 1, z + 1),
                    ];

                    let grid_vals = [
                        self[grid_points[0]],
                        self[grid_points[1]],
                        self[grid_points[2]],
                        self[grid_points[3]],
                        self[grid_points[4]],
                        self[grid_points[5]],
                        self[grid_points[6]],
                        self[grid_points[7]],
                    ];

                    let mut cube_index = 0;

                    if grid_vals[0] < iso_level {
                        cube_index |= 1
                    };
                    if grid_vals[1] < iso_level {
                        cube_index |= 2
                    };
                    if grid_vals[2] < iso_level {
                        cube_index |= 4
                    };
                    if grid_vals[3] < iso_level {
                        cube_index |= 8
                    };
                    if grid_vals[4] < iso_level {
                        cube_index |= 16
                    };
                    if grid_vals[5] < iso_level {
                        cube_index |= 32
                    };
                    if grid_vals[6] < iso_level {
                        cube_index |= 64
                    };
                    if grid_vals[7] < iso_level {
                        cube_index |= 128
                    };

                    if index_table::EDGE_TABLE[cube_index] == 0 {
                        continue;
                    }

                    let mut inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

                    macro_rules! edge {
                        ($i0:expr, $i1:expr, $n:expr) => {
                            if index_table::EDGE_TABLE[cube_index] & (1 << $n) != 0 {
                                let v0 = Vec3::new(
                                    grid_points[$i0].0 as f32,
                                    grid_points[$i0].1 as f32,
                                    grid_points[$i0].2 as f32,
                                );
                                let v1 = Vec3::new(
                                    grid_points[$i1].0 as f32,
                                    grid_points[$i1].1 as f32,
                                    grid_points[$i1].2 as f32,
                                );

                                inds[$n] = vertices.len();

                                let vert = vert_interpret(
                                    v0 * self.sample_density,
                                    v1 * self.sample_density,
                                    grid_vals[$i0],
                                    grid_vals[$i1],
                                    iso_level,
                                );

                                if let Some((index, _)) = vertices
                                    .iter()
                                    .enumerate()
                                    .find(|(_, v)| (vert - Vec3::from(**v)).length() < self.sample_density / 50.0)
                                {
                                    inds[$n] = index;
                                } else {
                                    vertices.push((vert).into());
                                    normals.push([0.0, 0.0, 0.0]);
                                    num_normals.push(0.0);
                                }
                            }
                        };
                    }

                    edge!(0, 1, 0);
                    edge!(1, 2, 1);
                    edge!(2, 3, 2);
                    edge!(3, 0, 3);

                    edge!(4, 5, 4);
                    edge!(5, 6, 5);
                    edge!(6, 7, 6);
                    edge!(7, 4, 7);

                    edge!(0, 4, 8);
                    edge!(1, 5, 9);
                    edge!(2, 6, 10);
                    edge!(3, 7, 11);

                    for mut i in 0..5 {
                        i *= 3;

                        if index_table::TRI_TABLE[cube_index][i] == -1
                            || index_table::TRI_TABLE[cube_index][i + 1] == -1
                            || index_table::TRI_TABLE[cube_index][i + 2] == -1
                        {
                            break;
                        }

                        let i0 = inds[index_table::TRI_TABLE[cube_index][i] as usize];
                        let i1 = inds[index_table::TRI_TABLE[cube_index][i + 1] as usize];
                        let i2 = inds[index_table::TRI_TABLE[cube_index][i + 2] as usize];

                        if i0 == i1 || i0 == i2 || i1 == i2 {
                            continue;
                        }

                        indices.push(i0 as u32);
                        indices.push(i1 as u32);
                        indices.push(i2 as u32);
                    }
                }
            }
        }

        for mut i in 0..indices.len() / 3 {
            i *= 3;

            let i0 = indices[i] as usize;
            let i1 = indices[i + 1] as usize;
            let i2 = indices[i + 2] as usize;

            let mut n0 = Vec3::from(normals[i0]);
            let mut n1 = Vec3::from(normals[i1]);
            let mut n2 = Vec3::from(normals[i2]);

            let v0 = Vec3::from(vertices[i0]);
            let v1 = Vec3::from(vertices[i1]);
            let v2 = Vec3::from(vertices[i2]);

            let area = (v1 - v0).cross(v2 - v0).length() / 2.0;
            let mut normal = (v0 - v1).cross(v0 - v2);

            if normal.length() > 0.0 {
                normal = normal.normalize() * (1.0 / area);
            } else {
                println!(
                    "normal between {}, {}, {}, is length 0, index: {}",
                    v0, v1, v2, i
                );
            }

            n0 += normal;
            n1 += normal;
            n2 += normal;

            normals[i0] = n0.into();
            normals[i1] = n1.into();
            normals[i2] = n2.into();

            num_normals[i0] += 1.0;
            num_normals[i1] += 1.0;
            num_normals[i2] += 1.0;
        }

        for (normal, num) in normals.iter_mut().zip(num_normals) {
            let n = Vec3::from(*normal);
            *normal = (n / num).normalize().into();
        }

        println!("len_verts: {}", vertices.len());

        mesh.set_indices(Some(bevy::render::mesh::Indices::U32(indices)));
        mesh.set_attribute(
            Mesh::ATTRIBUTE_POSITION,
            bevy::render::mesh::VertexAttributeValues::Float3(vertices),
        );
        mesh.set_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            bevy::render::mesh::VertexAttributeValues::Float3(normals),
        );

        println!("cube marching took: {:?}", std::time::Instant::now() - start_instant);

        mesh
    }

    fn mesh_set_system(
        mut meshes: ResMut<Assets<Mesh>>,
        query: Query<(&Handle<Mesh>, Changed<DensityField>)>,
    ) {
        for (mesh, density_field) in query.iter() {
            let mesh = meshes.get_mut(mesh).unwrap();
            *mesh = density_field.generate_mesh(0.5);
        }
    }
}

fn rotation(time: Res<Time>, mut query: Query<(&mut Transform, &Handle<Mesh>)>) {
    for (mut transform, _) in query.iter_mut() {
        transform.rotation = Quat::from_rotation_y(time.seconds_since_startup as f32);
    }
}

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup.system())
        .add_system(DensityField::mesh_set_system.system())
        .add_system(rotation.system())
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut density_field = DensityField::empty(-100..100, -100..100, -100..100, 0.025);
    density_field.set_density(|pos| {
        sdf::sphere(pos + Vec3::new(-0.25, 0.0, 0.0), 1.0)
            .max(sdf::sphere(pos + Vec3::new(0.25, 0.0, 0.0), 1.0))
            .max(sdf::rect(pos + Vec3::new(0.0, -0.25, 0.0), Vec3::new(0.5, 0.5, 0.5)))
            .min(1.0)
            .max(0.0)
    });

    commands
        .spawn(Camera3dComponents {
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)),
            ..Default::default()
        })
        .spawn(LightComponents {
            transform: Transform::from_translation(Vec3::new(5.0, 5.0, 0.0)),
            ..Default::default()
        })
        .spawn(PbrComponents {
            mesh: meshes.add(shape::Icosphere::default().into()),
            material: standard_materials.add(StandardMaterial {
                albedo: Color::rgb(0.5, 0.5, 0.5),
                albedo_texture: None,
                shaded: true,
            }),
            ..Default::default()
        })
        .with(density_field);
}

mod sdf {
    use super::*;

    pub fn sphere(pos: Vec3, radius: f32) -> f32 {
        radius - pos.length()
    }

    pub fn rect(pos: Vec3, size: Vec3) -> f32 {
        if (pos.abs() - size / 2.0).max(Vec3::new(0.0, 0.0, 0.0)).length() > 0.0 {
            0.0
        } else {
            1.0
        }
    }
}
