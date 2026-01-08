use clap::Parser;
use std::io::{self, Write};
use yt_clipper_rust::{check_dependencies, full_process};

mod server;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Run in web server mode
    #[arg(long)]
    server: bool,

    /// Port to run server on
    #[arg(long, default_value_t = 3000)]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Check dependencies (ffmpeg, yt-dlp)
    if let Err(e) = check_dependencies() {
        eprintln!("Error checking dependencies: {}", e);
        std::process::exit(1);
    }

    if args.server {
        server::start_server(args.port).await;
        return Ok(());
    }

    // CLI Mode
    print!("Enter YouTube link: ");
    io::stdout().flush()?;

    let mut link = String::new();
    io::stdin().read_line(&mut link)?;
    let link = link.trim();

    if link.is_empty() {
        println!("Invalid input.");
        return Ok(());
    }

    match full_process(link, "clips").await {
        Ok(files) => {
            println!(
                "Finished processing. {} clip(s) successfully saved to 'clips'.",
                files.len()
            );
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
