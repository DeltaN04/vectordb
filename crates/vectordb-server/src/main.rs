use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use vectordb_server::{serve_grpc, serve_http, AppState};

#[derive(Parser)]
#[command(name = "vectordb-server")]
struct Args {
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,
    #[arg(long)]
    data_dir: Option<PathBuf>,
    #[arg(long)]
    http_addr: Option<String>,
    #[arg(long)]
    grpc_addr: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct Config {
    data_dir: Option<String>,
    http_addr: Option<String>,
    grpc_addr: Option<String>,
    api_key: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let mut cfg = Config::default();
    if args.config.exists() {
        let s = std::fs::read_to_string(&args.config)?;
        cfg = toml::from_str(&s)?;
    }
    if let Ok(v) = std::env::var("VECTORDB_DATA_DIR") {
        cfg.data_dir = Some(v);
    }
    if let Ok(v) = std::env::var("VECTORDB_HTTP_ADDR") {
        cfg.http_addr = Some(v);
    }
    if let Ok(v) = std::env::var("VECTORDB_GRPC_ADDR") {
        cfg.grpc_addr = Some(v);
    }
    if let Ok(v) = std::env::var("VECTORDB_API_KEY") {
        cfg.api_key = Some(v);
    }

    let data_dir = args
        .data_dir
        .unwrap_or_else(|| PathBuf::from(cfg.data_dir.clone().unwrap_or_else(|| "./data".into())));
    let http_addr: SocketAddr = args
        .http_addr
        .clone()
        .or(cfg.http_addr)
        .unwrap_or_else(|| "0.0.0.0:8080".into())
        .parse()?;
    let grpc_addr: SocketAddr = args
        .grpc_addr
        .clone()
        .or(cfg.grpc_addr)
        .unwrap_or_else(|| "0.0.0.0:50051".into())
        .parse()?;

    std::fs::create_dir_all(&data_dir)?;
    let st = AppState {
        data_dir,
        collections: std::sync::Arc::new(
            tokio::sync::RwLock::new(std::collections::HashMap::new()),
        ),
        api_key: cfg.api_key,
    };

    let http_st = st.clone();
    let grpc_st = st.clone();
    let http_task = tokio::spawn(async move {
        if let Err(e) = serve_http(http_st, http_addr).await {
            tracing::error!("http error: {}", e);
        }
    });
    let grpc_task = tokio::spawn(async move {
        if let Err(e) = serve_grpc(grpc_st, grpc_addr).await {
            tracing::error!("grpc error: {}", e);
        }
    });

    tokio::signal::ctrl_c().await?;
    http_task.abort();
    grpc_task.abort();
    Ok(())
}
