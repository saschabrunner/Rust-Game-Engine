// this file will build the shaders at compile time and output spir-v binaries
use shaderc::ShaderKind;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // run script only if a shader changed
    println!("cargo:rerun-if-changed=data/shaders");

    // compiled shaders end up here
    std::fs::create_dir_all("target/data/shaders")?;

    let mut shader_compiler = shaderc::Compiler::new().unwrap();

    for entry in std::fs::read_dir("data/shaders")? {
        let entry = entry?;

        if entry.file_type()?.is_file() {
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string(); // to_string at end to get a clone

            let shader_kind =
                path.extension()
                    .and_then(|ext| match ext.to_string_lossy().as_ref() {
                        "vert" => Some(ShaderKind::Vertex),
                        "frag" => Some(ShaderKind::Fragment),
                        _ => None,
                    });

            if let Some(shader_kind) = shader_kind {
                let source = std::fs::read_to_string(&path)?;
                let compiled_shader = shader_compiler
                    .compile_into_spirv(&source, shader_kind, &file_name, "main", None)?
                    .as_binary_u8()
                    .to_vec();

                let out_path = format!("target/data/shaders/{}.spv", file_name);

                std::fs::write(&out_path, &compiled_shader)?;
            }
        }
    }

    Ok(())
}
