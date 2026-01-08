use clap::Parser;
use std::io::{self, Write};
use yt_clipper_rust::{
    check_dependencies, full_process, update_ytdlp,
    CropMode, ProcessOptions, SubtitleConfig, WhisperModel,
};

mod server;

#[derive(Parser, Debug)]
#[command(name = "yt-clipper-rust")]
#[command(version, about = "YouTube Heatmap Clipper - Generate viral vertical clips from YouTube videos")]
#[command(long_about = "Automatically extract high-engagement moments from YouTube videos using heatmap data and convert them to vertical format for social media.")]
struct Args {
    /// Run in web server mode
    #[arg(long)]
    server: bool,

    /// Port to run server on
    #[arg(long, default_value_t = 3000)]
    port: u16,

    /// YouTube URL (optional, will prompt if not provided)
    #[arg(short, long)]
    url: Option<String>,

    /// Crop mode: default, split-left, split-right
    #[arg(short, long, default_value = "default")]
    crop: String,

    /// Enable auto subtitle using Faster-Whisper
    #[arg(short, long)]
    subtitle: bool,

    /// Whisper model size: tiny, base, small, medium, large
    #[arg(long, default_value = "small")]
    model: String,

    /// Subtitle language code (e.g., id, en, ja)
    #[arg(long, default_value = "id")]
    language: String,

    /// Output directory for clips
    #[arg(short, long, default_value = "clips")]
    output: String,

    /// Update yt-dlp before processing
    #[arg(long)]
    update: bool,

    /// Run in interactive mode (prompts for all options)
    #[arg(short, long)]
    interactive: bool,
}

fn prompt_crop_mode() -> CropMode {
    println!("\n=== Crop Mode ===");
    println!("1. Default (center crop)");
    println!("2. Split Left (top: center, bottom: bottom-left facecam)");
    println!("3. Split Right (top: center, bottom: bottom-right facecam)");

    loop {
        print!("\nSelect crop mode (1-3): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            continue;
        }

        if let Some(mode) = CropMode::from_input(input.trim()) {
            println!("Selected: {}", mode.description());
            return mode;
        }
        println!("Invalid choice. Please enter 1, 2, or 3.");
    }
}

fn prompt_subtitle() -> (bool, WhisperModel) {
    println!("\n=== Auto Subtitle ===");
    println!("Available models:");
    println!("  - tiny   ({})", WhisperModel::Tiny.size_display());
    println!("  - base   ({})", WhisperModel::Base.size_display());
    println!("  - small  ({}) [recommended]", WhisperModel::Small.size_display());
    println!("  - medium ({})", WhisperModel::Medium.size_display());
    println!("  - large  ({})", WhisperModel::Large.size_display());

    print!("\nEnable subtitle? (y/n or model name): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return (false, WhisperModel::Small);
    }

    let input_trimmed = input.trim().to_lowercase();

    // Check if user typed a model name directly
    if let Some(model) = WhisperModel::from_input(&input_trimmed) {
        println!("Subtitle enabled with model: {} ({})", model, model.size_display());
        return (true, model);
    }

    // Check for y/yes
    let enabled = matches!(input_trimmed.as_str(), "y" | "yes");

    if !enabled {
        println!("Subtitle disabled.");
        return (false, WhisperModel::Small);
    }

    print!("Select model (tiny/base/small/medium/large) [small]: ");
    io::stdout().flush().unwrap();

    let mut model_input = String::new();
    if io::stdin().read_line(&mut model_input).is_err() || model_input.trim().is_empty() {
        println!("Using default model: small");
        return (true, WhisperModel::Small);
    }

    let model = WhisperModel::from_input(model_input.trim()).unwrap_or(WhisperModel::Small);
    println!("Subtitle enabled with model: {} ({})", model, model.size_display());
    (true, model)
}

fn prompt_url() -> String {
    print!("\nEnter YouTube URL: ");
    io::stdout().flush().unwrap();

    let mut link = String::new();
    io::stdin().read_line(&mut link).unwrap();
    link.trim().to_string()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Check dependencies (ffmpeg, yt-dlp)
    if let Err(e) = check_dependencies() {
        eprintln!("Error checking dependencies: {}", e);
        std::process::exit(1);
    }

    // Update yt-dlp if requested
    if args.update {
        let _ = update_ytdlp();
    }

    // Server mode
    if args.server {
        server::start_server(args.port).await;
        return Ok(());
    }

    // Determine options - interactive or from args
    let (crop_mode, subtitle_enabled, whisper_model, language, url) = if args.interactive {
        // Interactive mode
        let crop_mode = prompt_crop_mode();
        let (subtitle_enabled, whisper_model) = prompt_subtitle();
        let url = prompt_url();
        (crop_mode, subtitle_enabled, whisper_model, args.language.clone(), url)
    } else {
        // Use command line arguments
        let crop_mode = CropMode::from_input(&args.crop).unwrap_or(CropMode::Default);
        let whisper_model = WhisperModel::from_input(&args.model).unwrap_or(WhisperModel::Small);

        let url = if let Some(u) = args.url {
            u
        } else {
            // Prompt for URL if not provided
            prompt_url()
        };

        (crop_mode, args.subtitle, whisper_model, args.language.clone(), url)
    };

    if url.is_empty() {
        println!("Invalid input. No URL provided.");
        return Ok(());
    }

    // Build process options (SubtitleConfig::new auto-detects backend)
    let subtitle_config = SubtitleConfig::new(
        subtitle_enabled,
        whisper_model,
        &language,
    );

    let options = ProcessOptions::new(crop_mode, subtitle_config, &args.output);

    println!("\n=== Processing ===");
    println!("URL: {}", url);
    println!("Crop mode: {}", crop_mode.description());
    println!("Subtitle: {}", if options.subtitle.enabled {
        format!("enabled ({}, {})", options.subtitle.model, options.subtitle.language)
    } else {
        "disabled".to_string()
    });
    println!("Output: {}", args.output);
    println!();

    match full_process(&url, &options).await {
        Ok(files) => {
            println!(
                "\nFinished processing. {} clip(s) successfully saved to '{}'.",
                files.len(),
                args.output
            );
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
