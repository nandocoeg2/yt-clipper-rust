use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::{
    cors::CorsLayer,
    services::ServeDir,
    trace::TraceLayer,
};
use yt_clipper_rust::{full_process};
use std::net::SocketAddr;

#[derive(Deserialize)]
struct ProcessRequest {
    url: String,
}

#[derive(Serialize)]
struct ProcessResponse {
    message: String,
    files: Vec<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

async fn process_handler(Json(payload): Json<ProcessRequest>) -> impl IntoResponse {
    // Hardcoded output directory for web mode for now, or per-request
    let output_dir = "clips";
    
    match full_process(&payload.url, output_dir).await {
        Ok(files) => {
            (StatusCode::OK, Json(ProcessResponse {
                message: "Processing complete".to_string(),
                files,
            })).into_response()
        },
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
                error: e.to_string(),
            })).into_response()
        }
    }
}

pub async fn start_server(port: u16) {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/api/process", post(process_handler))
        .nest_service("/clips", ServeDir::new("clips"))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("Server running on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
