use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};
use yt_clipper_rust::{
    full_process, subtitle::check_python_available, CropMode, ProcessOptions, SubtitleConfig,
    WhisperModel,
};
use std::net::SocketAddr;

#[derive(Deserialize)]
pub struct ProcessRequest {
    url: String,
    #[serde(default)]
    crop_mode: Option<String>,
    #[serde(default)]
    subtitle: Option<bool>,
    #[serde(default)]
    whisper_model: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    output_dir: Option<String>,
    #[serde(default)]
    gpu: Option<bool>,
}

#[derive(Serialize)]
struct ProcessResponse {
    message: String,
    files: Vec<String>,
    options: ProcessOptionsResponse,
}

#[derive(Serialize)]
struct ProcessOptionsResponse {
    crop_mode: String,
    subtitle_enabled: bool,
    whisper_model: Option<String>,
    language: Option<String>,
    gpu: bool,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

async fn process_handler(Json(payload): Json<ProcessRequest>) -> impl IntoResponse {
    // Parse crop mode
    let crop_mode = payload
        .crop_mode
        .as_deref()
        .and_then(CropMode::from_input)
        .unwrap_or(CropMode::Default);

    // Parse whisper model
    let whisper_model = payload
        .whisper_model
        .as_deref()
        .and_then(WhisperModel::from_input)
        .unwrap_or(WhisperModel::Small);

    // Check subtitle availability
    let subtitle_enabled = payload.subtitle.unwrap_or(false) && check_python_available();

    // Language
    let language = payload.language.clone().unwrap_or_else(|| "id".to_string());

    // Output directory
    let output_dir = payload.output_dir.clone().unwrap_or_else(|| "clips".to_string());

    // GPU acceleration
    let use_gpu = payload.gpu.unwrap_or(false);

    // Build options
    let subtitle_config = SubtitleConfig::new(subtitle_enabled, whisper_model, &language);
    let options = ProcessOptions::new(crop_mode, subtitle_config, &output_dir)
        .with_gpu(use_gpu);

    // Process video
    match full_process(&payload.url, &options).await {
        Ok(files) => {
            let response = ProcessResponse {
                message: "Processing complete".to_string(),
                files,
                options: ProcessOptionsResponse {
                    crop_mode: crop_mode.to_string(),
                    subtitle_enabled,
                    whisper_model: if subtitle_enabled {
                        Some(whisper_model.to_string())
                    } else {
                        None
                    },
                    language: if subtitle_enabled {
                        Some(language)
                    } else {
                        None
                    },
                    gpu: use_gpu,
                },
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "features": {
            "crop_modes": ["default", "split-left", "split-right"],
            "subtitle": check_python_available(),
            "whisper_models": ["tiny", "base", "small", "medium", "large"],
            "gpu": true
        }
    }))
}

pub async fn start_server(port: u16) {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/api/process", post(process_handler))
        .route("/api/health", axum::routing::get(health_handler))
        .nest_service("/clips", ServeDir::new("clips"))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("Server running on http://{}", addr);
    println!("\nAvailable endpoints:");
    println!("  POST /api/process - Process YouTube video");
    println!("  GET  /api/health  - Health check");
    println!("  GET  /clips/*     - Serve generated clips");
    println!("\nExample request:");
    println!(r#"  curl -X POST http://localhost:{}/api/process \"#, port);
    println!(r#"    -H "Content-Type: application/json" \"#);
    println!(r#"    -d '{{"url": "https://youtube.com/watch?v=VIDEO_ID", "crop_mode": "default", "subtitle": false, "gpu": true}}'"#);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
