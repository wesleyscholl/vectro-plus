use clap::{Parser, Subcommand};
use std::path::Path;
use vectro_cli::compress_stream;

#[derive(Parser)]
#[command(name = "vectro")]
#[command(about = "Vectro+ — Rust embedding compressor & search tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compress {
        input: String,
        output: String,
        #[arg(long, default_value_t = false)]
        /// Produce a quantized streaming dataset (per-dimension min/max -> u8).
        /// This reduces size and speeds up search at the cost of some accuracy.
        /// Use for large datasets where memory/storage is constrained.
        /// Default: false
        quantize: bool,
    },
    /// Run library benchmarks (uses the `vectro_plus` bench harness).
    /// Streams benchmark output and shows a spinner while running.
    Bench {},
    Search {
        query: String,
        #[arg(short, long, default_value_t = 10)]
        top_k: usize,
        /// Path to dataset (bincode). If omitted, uses built-in toy dataset.
        #[arg(long)]
        dataset: Option<String>,
    },
    Serve {
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress { input, output, quantize } => {
            let _ = crate::compress_stream(&input, &output, quantize)?;
        }
        Commands::Bench {} => {
            // Run cargo bench for vectro_plus and stream output. Show a spinner while running.
            use indicatif::{ProgressBar, ProgressStyle};
            use std::process::{Command, Stdio};

            let pb = ProgressBar::new_spinner();
            pb.set_style(ProgressStyle::with_template("{spinner} {msg}").unwrap());
            pb.enable_steady_tick(std::time::Duration::from_millis(80));
            pb.set_message("running benches...");

            let mut cmd = Command::new("cargo");
            cmd.arg("bench").arg("-p").arg("vectro_plus").stdout(Stdio::piped()).stderr(Stdio::piped());

            let mut child = cmd.spawn().expect("failed to spawn cargo bench");
            if let Some(mut out) = child.stdout.take() {
                let mut buf = Vec::new();
                use std::io::Read;
                let _ = out.read_to_end(&mut buf);
                pb.finish_and_clear();
                print!("{}", String::from_utf8_lossy(&buf));
            }
            let status = child.wait().expect("bench wait failed");
            if !status.success() {
                eprintln!("bench failed: {:?}", status);
            }
        }
        Commands::Search { query, top_k, dataset } => {
            let vec: Vec<f32> = query
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect();

            // Load dataset if provided. If not provided and ./dataset.bin exists, use it.
            let embeddings = if let Some(path) = dataset {
                match vectro_plus::EmbeddingDataset::load(&path) {
                    Ok(ds) => ds.embeddings,
                    Err(e) => {
                        eprintln!("failed to load dataset {}: {}", path, e);
                        vec![
                            vectro_plus::Embedding::new("one", vec![1.0, 0.0]),
                            vectro_plus::Embedding::new("two", vec![0.0, 1.0]),
                            vectro_plus::Embedding::new("three", vec![0.707, 0.707]),
                        ]
                    }
                }
            } else if Path::new("./dataset.bin").exists() {
                match vectro_plus::EmbeddingDataset::load("./dataset.bin") {
                    Ok(ds) => ds.embeddings,
                    Err(e) => {
                        eprintln!("failed to load ./dataset.bin: {}", e);
                        vec![
                            vectro_plus::Embedding::new("one", vec![1.0, 0.0]),
                            vectro_plus::Embedding::new("two", vec![0.0, 1.0]),
                            vectro_plus::Embedding::new("three", vec![0.707, 0.707]),
                        ]
                    }
                }
            } else {
                vec![
                    vectro_plus::Embedding::new("one", vec![1.0, 0.0]),
                    vectro_plus::Embedding::new("two", vec![0.0, 1.0]),
                    vectro_plus::Embedding::new("three", vec![0.707, 0.707]),
                ]
            };

            // Build SearchIndex for faster repeated queries
            let idx = vectro_plus::search::SearchIndex::from_dataset(&embeddings);
            let results = idx.top_k(&vec, top_k);
            for (i, (id, score)) in results.into_iter().enumerate() {
                println!("{}. {} -> {:.6}", i + 1, id, score);
            }
        }
        Commands::Serve { port } => {
            println!("serve on port {} (not implemented)", port);
        }
    }

    Ok(())
}
