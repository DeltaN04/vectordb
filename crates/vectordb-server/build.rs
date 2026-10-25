fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = "../../proto/vectordb.proto";
    tonic_prost_build::configure()
        .build_server(true)
        .compile_protos(&[proto], &["../../proto"])?;
    Ok(())
}
