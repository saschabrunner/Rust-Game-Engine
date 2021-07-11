use nalgebra::{Matrix4, Point3, Vector3};

use gfx_hal::window::Extent2D;

pub struct CameraDesc {
    pub position: Point3<f32>,
    pub up: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,

    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for CameraDesc {
    fn default() -> CameraDesc {
        CameraDesc {
            position: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, -1.0, 0.0),
            yaw: 0.0,
            pitch: 0.0,

            fov: 0.9, // around 50° vertical fov
            near: 0.1,
            far: 100.0,
        }
    }
}

pub struct Camera {
    position: Point3<f32>,
    up: Vector3<f32>,
    forward: Vector3<f32>,
    right: Vector3<f32>,

    yaw: f32,
    pitch: f32,

    fov: f32,
    near: f32,
    far: f32,

    is_valid: bool,
    last_view: Matrix4<f32>,
    last_projection: Matrix4<f32>,

    // todo: move below into separate input/game module
    sensitivity: f32,
    speed: f32,
    invert_vertical: bool,

    surface_extent: Extent2D,
}

impl Camera {
    pub fn new(desc: &CameraDesc, surface_extent: Extent2D) -> Self {
        let mut camera = Camera {
            position: desc.position,
            up: desc.up,
            forward: Vector3::zeros(),
            right: Vector3::zeros(),

            yaw: desc.yaw,
            pitch: desc.pitch,

            fov: desc.fov,
            near: desc.near,
            far: desc.far,

            is_valid: false,
            last_view: Matrix4::zeros(),
            last_projection: Matrix4::zeros(),

            sensitivity: 0.002,
            speed: 0.25,
            invert_vertical: false,

            surface_extent,
        };

        camera.update();
        camera
    }

    pub fn get_view(self: &mut Self) -> Matrix4<f32> {
        if !self.is_valid {
            self.update();
        }

        self.last_view
    }

    pub fn get_projection(&mut self) -> Matrix4<f32> {
        if !self.is_valid {
            self.update();
        }

        self.last_projection
    }

    pub fn rotate(&mut self, x: f32, y: f32) {
        // x positive -> mouse moved to the right
        // y positive -> mouse moved up
        // invert y here if requested
        let real_y = if self.invert_vertical { -y } else { y };

        self.yaw += x * self.sensitivity;
        self.pitch += real_y * self.sensitivity;

        // constrain vertical rotation
        const MAX_PITCH: f32 = std::f32::consts::PI / 2.01; //slightly less than 90 degrees
        if self.pitch > MAX_PITCH {
            self.pitch = MAX_PITCH;
        } else if self.pitch < -MAX_PITCH {
            self.pitch = -MAX_PITCH;
        }

        self.is_valid = false;
    }

    pub fn translate(&mut self, x: f32, y: f32) {
        // todo: delta time
        // x positive -> move to right
        // y positive -> move forward
        self.position += (x * self.speed * self.right) + (y * self.speed * self.forward);
        self.is_valid = false;
    }

    pub fn set_surface_extent(&mut self, surface_extent: Extent2D) {
        self.surface_extent = surface_extent;
        self.is_valid = false;
    }

    fn update(&mut self) {
        // create a vector that points the camera forward
        // pitch and yaw of zero point to +z
        self.forward = Vector3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        );

        // pitch and yaw of zero point to +x
        self.right = self.forward.cross(&self.up).normalize();

        // todo: switch to a more suitable coordinate system for nalgebra to avoid these copies here
        let pos_temp = Point3::new(self.position.x, self.position.y, -self.position.z);
        let forward_temp = Vector3::new(self.forward.x, self.forward.y, -self.forward.z);
        let up_temp = Vector3::new(self.up.x, -self.up.y, self.up.z);

        self.last_view =
            Matrix4::<f32>::look_at_rh(&pos_temp, &(pos_temp + forward_temp), &up_temp);

        self.last_projection = Matrix4::<f32>::new_perspective(
            self.surface_extent.width as f32 / self.surface_extent.height as f32,
            self.fov,
            self.near,
            self.far,
        );

        self.is_valid = true;
    }
}
