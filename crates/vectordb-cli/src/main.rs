use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vectordb_core::collection::Collection;
use vectordb_core::distance::Metric;
use vectordb_core::index::HnswParams;
use vectordb_core::storage::FsyncPolicy;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Create {
        path: PathBuf,
        name: String,
        dim: usize,
    },
    Stats {
        path: PathBuf,
        name: String,
        dim: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Create { path, name, dim } => {
            let _ = Collection::create(
                &path,
                name,
                dim,
                Metric::L2,
                HnswParams::default(),
                FsyncPolicy::None,
            )?;
            println!("created");
        }
        Cmd::Stats { path, name, dim } => {
            let c = Collection::create(
                &path,
                name,
                dim,
                Metric::L2,
                HnswParams::default(),
                FsyncPolicy::None,
            )?;
            println!("{}", c.stats());
        }
    }
    Ok(())
}
