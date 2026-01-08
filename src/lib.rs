use anyhow::{anyhow, Result};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::process::Command;
use url::Url;

pub const MIN_SCORE: f64 = 0.40;
pub const MAX_DURATION: f64 = 60.0;
pub const PADDING: f64 = 10.0; // Extra seconds added before and after

#[derive(Debug, Clone, Deserialize)]
pub struct HeatmapSegment {
    pub start: f64,
    pub duration: f64,
    pub score: f64,
}

/// Extract the YouTube video ID from a given URL.
pub fn extract_video_id(url: &str) -> Option<String> {
    let parsed = Url::parse(url).ok()?;
    let host = parsed.host_str()?;

    if host == "youtu.be" || host == "www.youtu.be" {
        return Some(parsed.path().trim_start_matches('/').to_string());
    }

    if host == "youtube.com" || host == "www.youtube.com" {
        if parsed.path() == "/watch" {
            let pairs = parsed.query_pairs();
            for (key, value) in pairs {
                if key == "v" {
                    return Some(value.into_owned());
                }
            }
        }
        if parsed.path().starts_with("/shorts/") {
            let parts: Vec<&str> = parsed.path().split('/').collect();
            if parts.len() >= 3 {
                return Some(parts[2].to_string());
            }
        }
    }

    None
}

/// Fetch and parse YouTube 'Most Replayed' heatmap data.
pub async fn fetch_heatmap(video_id: &str) -> Result<Vec<HeatmapSegment>> {
    let url = format!("https://www.youtube.com/watch?v={}", video_id);
    let client = reqwest::Client::new();
    let res = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0")
        .send()
        .await?
        .text()
        .await?;

    let re = Regex::new(r#""markers":\s*(\[.*?\])\s*,\s*"?markersMetadata"?"#)?;
    let caps = re
        .captures(&res)
        .ok_or_else(|| anyhow!("No heatmap markers found"))?;
    let json_text = caps.get(1).unwrap().as_str().replace("\\\"", "\"");

    let markers: Vec<Value> = serde_json::from_str(&json_text)?;

    let mut results = Vec::new();

    for marker in markers {
        let data = if let Some(renderer) = marker.get("heatMarkerRenderer") {
            renderer
        } else {
            &marker
        };

        // Helper to parse potential string or number values
        let parse_val = |v: &Value| -> Option<f64> {
            match v {
                Value::Number(n) => n.as_f64(),
                Value::String(s) => s.parse().ok(),
                _ => None,
            }
        };

        if let (Some(start_val), Some(duration_val), Some(score_val)) = (
            data.get("startMillis"),
            data.get("durationMillis"),
            data.get("intensityScoreNormalized"),
        ) {
            let score = parse_val(score_val).unwrap_or(0.0);

            if score >= MIN_SCORE {
                let start_millis = parse_val(start_val).unwrap_or(0.0);
                let duration_millis = parse_val(duration_val).unwrap_or(0.0);

                let start = start_millis / 1000.0;
                let duration = duration_millis / 1000.0;

                results.push(HeatmapSegment {
                    start,
                    duration: duration.min(MAX_DURATION),
                    score,
                });
            }
        }
    }

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Retrieve the total duration of a YouTube video in seconds using yt-dlp.
pub fn get_duration(video_id: &str) -> Result<u64> {
    let output = Command::new("yt-dlp") // Assumes yt-dlp is in PATH, similar to python script expectations but we should probably ensure it or genericize 'sys.executable' logic if needed, but 'yt-dlp' in path is standard expectation.
        .arg("--get-duration")
        .arg(format!("https://youtu.be/{}", video_id))
        .output()?;

    if !output.status.success() {
        return Err(anyhow!("yt-dlp failed to get duration"));
    }

    let stdout = String::from_utf8(output.stdout)?;
    let time_str = stdout.trim();

    // Format usually hh:mm:ss or mm:ss
    let parts: Vec<&str> = time_str.split(':').collect();

    let duration = match parts.len() {
        2 => {
            let m: u64 = parts[0].parse().unwrap_or(0);
            let s: u64 = parts[1].parse().unwrap_or(0);
            m * 60 + s
        }
        3 => {
            let h: u64 = parts[0].parse().unwrap_or(0);
            let m: u64 = parts[1].parse().unwrap_or(0);
            let s: u64 = parts[2].parse().unwrap_or(0);
            h * 3600 + m * 60 + s
        }
        _ => parts[0].parse().unwrap_or(0),
    };

    Ok(duration)
}

/// Download, crop, and export a single vertical clip based on a heatmap segment.
pub fn process_clip(
    video_id: &str,
    segment: &HeatmapSegment,
    index: usize,
    total_duration: u64,
    output_dir: &str,
) -> Result<bool> {
    let start_original = segment.start;
    let end_original = segment.start + segment.duration;

    let start = (start_original - PADDING).max(0.0);
    let end = (end_original + PADDING).min(total_duration as f64);

    if end - start < 3.0 {
        return Ok(false);
    }

    let temp_file = format!("temp_{}.mp4", index);
    let output_path = std::path::Path::new(output_dir).join(format!("clip_{}.mp4", index));
    let output_file = output_path.to_string_lossy().to_string();

    println!(
        "[Clip {}] Processing segment ({}s - {}s, padding {}s)",
        index, start as u64, end as u64, PADDING
    );

    // 1. Download segment
    let status = Command::new("yt-dlp")
        // .arg(format!("--force-ipv4")) // Python had this, strictly necessary? maybe.
        .args(&["--force-ipv4", "--quiet", "--no-warnings"])
        .arg("--downloader")
        .arg("ffmpeg")
        .arg("--downloader-args")
        .arg(format!(
            "ffmpeg_i:-ss {} -to {} -hide_banner -loglevel error",
            start, end
        ))
        .arg("-f")
        .arg("bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best")
        .arg("-o")
        .arg(&temp_file)
        .arg(format!("https://youtu.be/{}", video_id))
        .status()?;

    if !status.success() {
        println!("Failed to download video segment.");
        return Ok(false);
    }

    if !std::path::Path::new(&temp_file).exists() {
        println!("Failed to download video segment (file missing).");
        return Ok(false);
    }

    // 2. Convert/Crop
    let convert_status = Command::new("ffmpeg")
        .args(&["-y", "-hide_banner", "-loglevel", "error"])
        .arg("-i")
        .arg(&temp_file)
        .args(&["-vf", "scale=-2:1280,crop=720:1280"])
        .args(&["-c:v", "libx264", "-preset", "ultrafast", "-crf", "26"])
        .args(&["-c:a", "aac", "-b:a", "128k"])
        .arg(&output_file)
        .status()?;

    // 3. Cleanup temp file
    let _ = std::fs::remove_file(&temp_file);

    if convert_status.success() {
        println!("Clip successfully generated: {}", output_file);
        Ok(true)
    } else {
        println!("Failed to convert clip.");
        Ok(false)
    }
}

pub async fn full_process(video_url: &str, output_dir: &str) -> Result<Vec<String>> {
    let video_id = extract_video_id(video_url).ok_or_else(|| anyhow!("Invalid URL"))?;

    println!("Fetching heatmap for {}", video_id);
    let segments = fetch_heatmap(&video_id).await?;

    if segments.is_empty() {
        return Err(anyhow!("No high-engagement segments found"));
    }

    println!("Found {} segments. getting duration...", segments.len());
    let duration = get_duration(&video_id)?;

    fs::create_dir_all(output_dir)?;

    let mut generated_files = Vec::new();
    let max_clips = 10;
    let mut success_count = 0;

    for segment in segments {
        if success_count >= max_clips {
            break;
        }

        let index = success_count + 1;
        // Note: process_clip is blocking; for a high-performance server, wrapping in spawn_blocking would be better.
        // But for this tool, direct call is acceptable.
        if let Ok(true) = process_clip(&video_id, &segment, index, duration, output_dir) {
            generated_files.push(format!("clip_{}.mp4", index));
            success_count += 1;
        }
    }

    Ok(generated_files)
}

pub fn check_dependencies() -> Result<()> {
    if which::which("ffmpeg").is_err() {
        return Err(anyhow!(
            "FFmpeg not found. Please install FFmpeg and ensure it is in PATH."
        ));
    }

    if which::which("yt-dlp").is_err() {
        return Err(anyhow!("yt-dlp not found. Please install it and ensure it is in PATH.\nDownload: https://github.com/yt-dlp/yt-dlp/releases"));
    }

    Ok(())
}
