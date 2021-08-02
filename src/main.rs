mod camera;
mod device;
mod image;
mod memory;
mod render;
mod world;

use render::Renderer;

use gfx_hal::{adapter::Adapter, window::Extent2D, Instance};

use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
};

pub const APPLICATION_NAME: &'static str = "Rust Game Engine";
pub const DEFAULT_WINDOW_SIZE: [u32; 2] = [512, 512];

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let logical_window_size: LogicalSize<u32> = DEFAULT_WINDOW_SIZE.into();

    let window = winit::window::WindowBuilder::new()
        .with_title(APPLICATION_NAME)
        .with_inner_size(logical_window_size)
        .build(&event_loop)
        .expect("Failed to create window");

    window.set_cursor_grab(true).unwrap();
    window.set_cursor_visible(false);

    let instance = backend::Instance::create(APPLICATION_NAME, 1).expect("Backend not supported");

    // surface creation is unsafe because raw window handle might be invalid pointer
    let surface = unsafe {
        instance
            .create_surface(&window)
            .expect("Failed to create surface for window")
    };

    let adapter: Adapter<backend::Backend> = find_adapter(&instance);

    let mut renderer = Renderer::new(instance, surface, adapter);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(dimensions) => {
                renderer.recreate_swapchain(Extent2D {
                    width: dimensions.width,
                    height: dimensions.height,
                });
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                renderer.recreate_swapchain(Extent2D {
                    width: new_inner_size.width,
                    height: new_inner_size.height,
                });
            }
            WindowEvent::KeyboardInput { input, .. } => match input {
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(key),
                    ..
                } => match key {
                    // todo: build key tracking input system around it? DeviceEvent::Key instead maybe?
                    VirtualKeyCode::W => renderer.camera().translate(0.0, 1.0),
                    VirtualKeyCode::A => renderer.camera().translate(-1.0, 0.0),
                    VirtualKeyCode::S => renderer.camera().translate(0.0, -1.0),
                    VirtualKeyCode::D => renderer.camera().translate(1.0, 0.0),
                    VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                    _ => (),
                },
                _ => (),
            },
            _ => (),
        },
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::MouseMotion { delta } => {
                renderer.camera().rotate(delta.0 as f32, delta.1 as f32)
            }
            _ => (),
        },
        Event::MainEventsCleared => window.request_redraw(),
        Event::RedrawRequested(_) => {
            renderer.render();
        }
        _ => (),
    });
}

// todo: this might be needed again for proper dpi handling
// fn get_window_sizes<T>(event_loop: &EventLoop<T>) -> (LogicalSize<u32>, PhysicalSize<u32>) {
//     // get either the primary monitor's dpi or the
//     // monitor with the highest dpi (on Wayland for example)
//     let primary_monitor = event_loop.primary_monitor();
//     let dpi = match primary_monitor {
//         Some(primary_monitor) => primary_monitor.scale_factor(),
//         None => event_loop
//             .available_monitors()
//             .map(|monitor| monitor.scale_factor())
//             .max_by(|a, b| a.partial_cmp(b).unwrap())
//             .unwrap_or(0.0),
//     };

//     let logical: LogicalSize<u32> = DEFAULT_WINDOW_SIZE.into();
//     let physical: PhysicalSize<u32> = logical.to_physical(dpi);

//     (logical, physical)
// }

fn find_adapter<B: gfx_hal::Backend>(instance: &B::Instance) -> Adapter<B> {
    // multiple adapters might exist, like an internal and a dedicated GPU
    // an algorithm could be implemented here to select a different GPU
    instance.enumerate_adapters().remove(0)
}
