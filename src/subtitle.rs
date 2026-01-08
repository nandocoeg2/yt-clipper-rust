use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::process::Command;
use strum::{Display, EnumString};

/// Available Whisper model sizes
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize, Display, EnumString)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum WhisperModel {
    Tiny,
    Base,
    #[default]
    Small,
    Medium,
    Large,
}

impl WhisperModel {
    /// Get approximate model size for display
    pub fn size_display(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "~75 MB",
            WhisperModel::Base => "~142 MB",
            WhisperModel::Small => "~466 MB",
            WhisperModel::Medium => "~1.5 GB",
            WhisperModel::Large => "~2.9 GB",
        }
    }

    /// Get the ggml model filename for whisper.cpp
    pub fn ggml_filename(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "ggml-tiny.bin",
            WhisperModel::Base => "ggml-base.bin",
            WhisperModel::Small => "ggml-small.bin",
            WhisperModel::Medium => "ggml-medium.bin",
            WhisperModel::Large => "ggml-large.bin",
        }
    }

    /// Get HuggingFace download URL for the model
    pub fn download_url(&self) -> String {
        format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            self.ggml_filename()
        )
    }

    /// Parse from user input
    pub fn from_input(input: &str) -> Option<Self> {
        match input.trim().to_lowercase().as_str() {
            "tiny" => Some(WhisperModel::Tiny),
            "base" => Some(WhisperModel::Base),
            "small" => Some(WhisperModel::Small),
            "medium" => Some(WhisperModel::Medium),
            "large" | "large-v1" | "large-v2" | "large-v3" => Some(WhisperModel::Large),
            _ => None,
        }
    }
}

/// Subtitle backend to use
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SubtitleBackend {
    /// whisper.cpp binary (recommended, no Python needed)
    #[default]
    WhisperCpp,
    /// Python faster-whisper (requires Python 3.11/3.12)
    FasterWhisper,
}

/// Subtitle configuration
#[derive(Debug, Clone)]
pub struct SubtitleConfig {
    pub enabled: bool,
    pub model: WhisperModel,
    pub language: String,
    pub backend: SubtitleBackend,
}

impl Default for SubtitleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: WhisperModel::Small,
            language: "id".to_string(),
            backend: SubtitleBackend::WhisperCpp,
        }
    }
}

impl SubtitleConfig {
    pub fn new(enabled: bool, model: WhisperModel, language: &str) -> Self {
        // Auto-detect best backend
        let backend = if check_whisper_cpp_available() {
            SubtitleBackend::WhisperCpp
        } else if check_python_available() {
            SubtitleBackend::FasterWhisper
        } else {
            SubtitleBackend::WhisperCpp // Default, will show error later
        };

        Self {
            enabled,
            model,
            language: language.to_string(),
            backend,
        }
    }

    pub fn with_backend(mut self, backend: SubtitleBackend) -> Self {
        self.backend = backend;
        self
    }
}

/// Get the whisper.cpp models directory
pub fn get_whisper_cpp_models_dir() -> std::path::PathBuf {
    // Check common locations
    if let Some(home) = dirs::home_dir() {
        let whisper_dir = home.join(".cache").join("whisper.cpp");
        if whisper_dir.exists() {
            return whisper_dir;
        }
        // Create if doesn't exist
        let _ = fs::create_dir_all(&whisper_dir);
        return whisper_dir;
    }

    // Fallback to current directory
    std::path::PathBuf::from("models")
}

/// Check if whisper.cpp binary is available
pub fn check_whisper_cpp_available() -> bool {
    // Try common binary names (whisper-cli is from scoop)
    let binary_names = ["whisper-cli", "whisper", "whisper-cpp", "main"];

    for name in binary_names {
        if which::which(name).is_ok() {
            return true;
        }
    }

    // Check in current directory
    let current_dir_binaries = [
        "whisper-cli.exe",
        "whisper.exe",
        "whisper-cpp.exe",
        "main.exe",
        "whisper-cli",
        "whisper",
        "main",
    ];
    for name in current_dir_binaries {
        if std::path::Path::new(name).exists() {
            return true;
        }
    }

    false
}

/// Get the whisper.cpp binary path
fn get_whisper_cpp_binary() -> Option<String> {
    let binary_names = ["whisper-cli", "whisper", "whisper-cpp", "main"];

    for name in binary_names {
        if which::which(name).is_ok() {
            return Some(name.to_string());
        }
    }

    // Check in current directory
    let current_dir_binaries = [
        "whisper-cli.exe",
        "whisper.exe",
        "whisper-cpp.exe",
        "main.exe",
        "whisper-cli",
        "whisper",
        "main",
    ];
    for name in current_dir_binaries {
        if std::path::Path::new(name).exists() {
            return Some(format!("./{}", name));
        }
    }

    None
}

/// Check if the whisper.cpp model exists
pub fn check_whisper_model_exists(model: WhisperModel) -> bool {
    let models_dir = get_whisper_cpp_models_dir();
    let model_path = models_dir.join(model.ggml_filename());
    model_path.exists()
}

/// Download whisper.cpp model using curl or powershell
pub fn download_whisper_model(model: WhisperModel) -> Result<std::path::PathBuf> {
    let models_dir = get_whisper_cpp_models_dir();
    fs::create_dir_all(&models_dir)?;

    let model_path = models_dir.join(model.ggml_filename());

    if model_path.exists() {
        println!("  Model already exists: {}", model_path.display());
        return Ok(model_path);
    }

    let url = model.download_url();
    println!(
        "  Downloading {} model ({})...",
        model,
        model.size_display()
    );
    println!("  URL: {}", url);
    println!("  Destination: {}", model_path.display());

    // Try curl first
    let status = Command::new("curl")
        .args(["-L", "-o"])
        .arg(&model_path)
        .arg(&url)
        .args(["--progress-bar"])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("  Model downloaded successfully!");
            return Ok(model_path);
        }
        _ => {
            // Try PowerShell on Windows
            #[cfg(target_os = "windows")]
            {
                println!("  Trying PowerShell download...");
                let ps_command = format!(
                    "Invoke-WebRequest -Uri '{}' -OutFile '{}'",
                    url,
                    model_path.display()
                );
                let status = Command::new("powershell")
                    .args(["-Command", &ps_command])
                    .status();

                if let Ok(s) = status {
                    if s.success() {
                        println!("  Model downloaded successfully!");
                        return Ok(model_path);
                    }
                }
            }
        }
    }

    Err(anyhow!(
        "Failed to download model. Please download manually from:\n  {}\n  and save to: {}",
        url,
        model_path.display()
    ))
}

/// Extract audio from video using FFmpeg (required for whisper.cpp)
fn extract_audio(video_file: &str, audio_file: &str) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-i", video_file])
        .args(["-ar", "16000"]) // 16kHz sample rate required by Whisper
        .args(["-ac", "1"]) // Mono
        .args(["-c:a", "pcm_s16le"]) // 16-bit PCM
        .arg(audio_file)
        .status()?;

    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("Failed to extract audio from video"))
    }
}

/// Word with timestamp from whisper
#[derive(Debug, Clone)]
struct TimedWord {
    text: String,
    start: f64,
    end: f64,
}

/// Parse whisper.cpp JSON output to get word-level timestamps
fn parse_whisper_json(json_file: &str) -> Result<Vec<TimedWord>> {
    let content = fs::read_to_string(json_file)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let mut words = Vec::new();

    // Try parsing the full JSON format first (from -ojf / --output-json-full)
    if let Some(transcription) = json.get("transcription").and_then(|t| t.as_array()) {
        for segment in transcription {
            // Check for tokens array (word-level data)
            if let Some(tokens) = segment.get("tokens").and_then(|t| t.as_array()) {
                for token in tokens {
                    if let (Some(text), Some(t0), Some(t1)) = (
                        token.get("text").and_then(|t| t.as_str()),
                        token.get("t0").and_then(|t| t.as_i64()),
                        token.get("t1").and_then(|t| t.as_i64()),
                    ) {
                        let text = text.trim();
                        // Skip empty, special tokens, and tokens starting with [
                        if !text.is_empty() && !text.starts_with('[') && !text.starts_with('<') {
                            words.push(TimedWord {
                                text: text.to_string(),
                                start: t0 as f64 / 100.0, // centiseconds to seconds
                                end: t1 as f64 / 100.0,
                            });
                        }
                    }
                    // Alternative format with offsets
                    else if let (Some(text), Some(start), Some(end)) = (
                        token.get("text").and_then(|t| t.as_str()),
                        token
                            .get("offsets")
                            .and_then(|o| o.get("from"))
                            .and_then(|f| f.as_i64()),
                        token
                            .get("offsets")
                            .and_then(|o| o.get("to"))
                            .and_then(|t| t.as_i64()),
                    ) {
                        let text = text.trim();
                        if !text.is_empty() && !text.starts_with('[') && !text.starts_with('<') {
                            words.push(TimedWord {
                                text: text.to_string(),
                                start: start as f64 / 1000.0,
                                end: end as f64 / 1000.0,
                            });
                        }
                    }
                }
            }
            // Fallback: use segment-level timestamps
            else if let (Some(text), Some(t0), Some(t1)) = (
                segment.get("text").and_then(|t| t.as_str()),
                segment
                    .get("timestamps")
                    .and_then(|ts| ts.get("from"))
                    .and_then(|f| f.as_str()),
                segment
                    .get("timestamps")
                    .and_then(|ts| ts.get("to"))
                    .and_then(|t| t.as_str()),
            ) {
                // Parse timestamp format "00:00:01,234"
                fn parse_ts(s: &str) -> Option<f64> {
                    let s = s.replace(',', ".");
                    let parts: Vec<&str> = s.split(':').collect();
                    if parts.len() == 3 {
                        let h: f64 = parts[0].parse().ok()?;
                        let m: f64 = parts[1].parse().ok()?;
                        let s: f64 = parts[2].parse().ok()?;
                        Some(h * 3600.0 + m * 60.0 + s)
                    } else {
                        None
                    }
                }

                if let (Some(start), Some(end)) = (parse_ts(t0), parse_ts(t1)) {
                    // Split segment text into words with estimated timing
                    let segment_words: Vec<&str> = text.split_whitespace().collect();
                    let duration = end - start;
                    let word_duration = duration / segment_words.len().max(1) as f64;

                    for (i, word_text) in segment_words.iter().enumerate() {
                        let word_text = word_text.trim();
                        if !word_text.is_empty() && !word_text.starts_with('[') {
                            words.push(TimedWord {
                                text: word_text.to_string(),
                                start: start + (i as f64 * word_duration),
                                end: start + ((i + 1) as f64 * word_duration),
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(words)
}

/// Format time for ASS format (h:mm:ss.cc)
fn format_ass_time(seconds: f64) -> String {
    let h = (seconds / 3600.0) as u32;
    let m = ((seconds % 3600.0) / 60.0) as u32;
    let s = (seconds % 60.0) as u32;
    let cs = ((seconds % 1.0) * 100.0) as u32;
    format!("{}:{:02}:{:02}.{:02}", h, m, s, cs)
}

/// Generate ASS subtitle with word-by-word highlight animation (TikTok/CapCut style)
fn generate_ass_with_word_highlight(words: &[TimedWord], output_file: &str) -> Result<()> {
    let mut file = fs::File::create(output_file)?;

    // ASS Header with styles optimized for word-by-word animation
    // Using transform effects for pop animation
    let header = r#"[Script Info]
Title: Word Highlight Subtitles
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,0,2,20,20,80,1
Style: Active,Arial Black,58,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,0,2,20,20,80,1
Style: Inactive,Arial Black,48,&H80FFFFFF,&H000000FF,&H00000000,&H40000000,1,0,0,0,100,100,0,0,1,3,0,2,20,20,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"#;

    file.write_all(header.as_bytes())?;

    // Group words into short phrases (2-4 words) for better readability
    let mut phrases: Vec<Vec<&TimedWord>> = Vec::new();
    let mut current_phrase: Vec<&TimedWord> = Vec::new();
    let max_words_per_phrase = 3;
    let max_chars_per_phrase = 20;
    let mut current_chars = 0;

    for word in words {
        current_phrase.push(word);
        current_chars += word.text.len() + 1;

        // Check for natural breaks (punctuation) or length limits
        let has_punctuation = word.text.ends_with('.')
            || word.text.ends_with(',')
            || word.text.ends_with('?')
            || word.text.ends_with('!');

        if current_phrase.len() >= max_words_per_phrase
            || current_chars >= max_chars_per_phrase
            || has_punctuation
        {
            phrases.push(current_phrase);
            current_phrase = Vec::new();
            current_chars = 0;
        }
    }
    if !current_phrase.is_empty() {
        phrases.push(current_phrase);
    }

    // Generate animated dialogue for each phrase
    for phrase_words in &phrases {
        if phrase_words.is_empty() {
            continue;
        }

        let phrase_start = phrase_words.first().unwrap().start;
        let phrase_end = phrase_words.last().unwrap().end + 0.5;

        // For each word in the phrase, create highlight animation
        for (word_idx, word) in phrase_words.iter().enumerate() {
            let word_start = word.start;
            let word_end = word.end;

            // Build the text with current word highlighted
            let mut text = String::new();

            for (i, w) in phrase_words.iter().enumerate() {
                if i == word_idx {
                    // Active word: Yellow, larger, with pop animation
                    // \t = transform over time, \fscx\fscy = scale
                    text.push_str(&format!(
                        "{{\\c&H00FFFF&\\fscx110\\fscy110\\t(0,50,\\fscx100\\fscy100)}}{}{{\\r}}",
                        w.text
                    ));
                } else if i < word_idx {
                    // Previous words: dimmer white
                    text.push_str(&format!("{{\\c&HCCCCCC&\\fscx95\\fscy95}}{}", w.text));
                } else {
                    // Future words: very dim
                    text.push_str(&format!("{{\\c&H666666&\\fscx90\\fscy90}}{}", w.text));
                }

                if i < phrase_words.len() - 1 {
                    text.push(' ');
                }
            }

            // Write dialogue line for this word's active period
            let dialogue = format!(
                "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
                format_ass_time(word_start),
                format_ass_time(word_end.max(word_start + 0.1)),
                text
            );
            file.write_all(dialogue.as_bytes())?;
        }

        // Show complete phrase briefly after all words are spoken
        let mut final_text = String::new();
        for (i, w) in phrase_words.iter().enumerate() {
            final_text.push_str(&format!("{{\\c&HFFFFFF&\\fscx100\\fscy100}}{}", w.text));
            if i < phrase_words.len() - 1 {
                final_text.push(' ');
            }
        }

        let last_word_end = phrase_words.last().unwrap().end;
        if phrase_end > last_word_end {
            let dialogue = format!(
                "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
                format_ass_time(last_word_end),
                format_ass_time(phrase_end),
                final_text
            );
            file.write_all(dialogue.as_bytes())?;
        }
    }

    Ok(())
}

/// Generate simple ASS (fallback when word-level timing not available)
fn generate_simple_ass(srt_file: &str, output_ass: &str) -> Result<()> {
    let srt_content = fs::read_to_string(srt_file)?;
    let mut file = fs::File::create(output_ass)?;

    // ASS Header - bold, large, with box effect
    let header = r#"[Script Info]
Title: Subtitles
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,38,&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,1,0,0,0,100,100,0,0,4,0,3,2,20,20,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"#;

    file.write_all(header.as_bytes())?;

    // Parse SRT and convert to ASS
    let mut lines_iter = srt_content.lines().peekable();

    while let Some(line) = lines_iter.next() {
        // Skip sequence number
        if line.trim().parse::<u32>().is_ok() {
            // Next line should be timestamp
            if let Some(timestamp_line) = lines_iter.next() {
                if let Some((start, end)) = parse_srt_timestamp(timestamp_line) {
                    // Collect text lines until empty line
                    let mut text_parts = Vec::new();
                    while let Some(text_line) = lines_iter.peek() {
                        if text_line.trim().is_empty() {
                            lines_iter.next();
                            break;
                        }
                        text_parts.push(lines_iter.next().unwrap().to_string());
                    }
                    let text = text_parts.join("\\N");

                    // Write dialogue
                    let dialogue = format!(
                        "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
                        format_ass_time(start),
                        format_ass_time(end),
                        text
                    );
                    file.write_all(dialogue.as_bytes())?;
                }
            }
        }
    }

    Ok(())
}

/// Parse SRT timestamp line "00:00:01,000 --> 00:00:02,500"
fn parse_srt_timestamp(line: &str) -> Option<(f64, f64)> {
    let parts: Vec<&str> = line.split(" --> ").collect();
    if parts.len() != 2 {
        return None;
    }

    fn parse_time(s: &str) -> Option<f64> {
        let s = s.trim().replace(',', ".");
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 3 {
            return None;
        }
        let h: f64 = parts[0].parse().ok()?;
        let m: f64 = parts[1].parse().ok()?;
        let s: f64 = parts[2].parse().ok()?;
        Some(h * 3600.0 + m * 60.0 + s)
    }

    Some((parse_time(parts[0])?, parse_time(parts[1])?))
}

/// Generate subtitle using whisper.cpp with word-level timestamps
fn generate_subtitle_whisper_cpp(
    video_file: &str,
    output_sub: &str,
    config: &SubtitleConfig,
) -> Result<()> {
    let binary = get_whisper_cpp_binary()
        .ok_or_else(|| anyhow!("whisper.cpp binary not found. Please install it."))?;

    // Check/download model
    if !check_whisper_model_exists(config.model) {
        println!("  Model not found. Downloading...");
        download_whisper_model(config.model)?;
    }

    let model_path = get_whisper_cpp_models_dir().join(config.model.ggml_filename());

    // Extract audio first (whisper.cpp works with audio files)
    let audio_file = format!("{}.wav", video_file.trim_end_matches(".mp4"));
    println!("  Extracting audio...");
    extract_audio(video_file, &audio_file)?;

    let output_base = output_sub
        .trim_end_matches(".ass")
        .trim_end_matches(".srt");

    println!(
        "  Transcribing with whisper.cpp ({}) - word-level...",
        config.model
    );

    // Use --output-json-full for detailed word timestamps
    // Use --split-on-word for word-level splitting
    // Use --max-len 1 for very short segments
    let output = Command::new(&binary)
        .args(["-m", &model_path.to_string_lossy()])
        .args(["-f", &audio_file])
        .args(["-l", &config.language])
        .args(["--output-json-full"]) // Full JSON with token timestamps
        .args(["--split-on-word"]) // Split on word boundaries
        .args(["--max-len", "1"]) // Very short segments for precise timing
        .args(["-of", output_base])
        .output()?;

    let json_file = format!("{}.json", output_base);
    let ass_file = format!("{}.ass", output_base);

    if output.status.success() && std::path::Path::new(&json_file).exists() {
        // Parse JSON and generate word-highlight ASS
        println!("  Generating word-by-word highlight subtitles...");
        match parse_whisper_json(&json_file) {
            Ok(words) if !words.is_empty() => {
                println!("  Found {} words with timestamps", words.len());
                generate_ass_with_word_highlight(&words, &ass_file)?;
                let _ = fs::remove_file(&json_file);
                let _ = fs::remove_file(&audio_file);

                // Rename to expected output
                if ass_file != output_sub {
                    fs::rename(&ass_file, output_sub)?;
                }
                println!("  Word-highlight subtitles generated!");
                return Ok(());
            }
            Ok(_) => {
                println!("  No words found in JSON, falling back...");
            }
            Err(e) => {
                println!("  Word-level parsing failed: {}, falling back...", e);
            }
        }
        let _ = fs::remove_file(&json_file);
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("  JSON generation failed: {}", stderr);
    }

    // Fallback: generate SRT and convert to styled ASS
    println!("  Falling back to standard subtitles...");

    let output = Command::new(&binary)
        .args(["-m", &model_path.to_string_lossy()])
        .args(["-f", &audio_file])
        .args(["-l", &config.language])
        .args(["--output-srt"])
        .args(["-of", output_base])
        .output()?;

    // Clean up audio file
    let _ = fs::remove_file(&audio_file);

    if output.status.success() {
        let srt_file = format!("{}.srt", output_base);
        if std::path::Path::new(&srt_file).exists() {
            generate_simple_ass(&srt_file, output_sub)?;
            let _ = fs::remove_file(&srt_file);
            println!("  Styled subtitles generated!");
            Ok(())
        } else {
            Err(anyhow!("SRT file not created"))
        }
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(anyhow!("whisper.cpp failed: {}", stderr))
    }
}

/// Check if faster-whisper Python package is available
pub fn check_faster_whisper_available() -> bool {
    let output = Command::new("python")
        .args(["-c", "import faster_whisper"])
        .output();

    match output {
        Ok(o) => o.status.success(),
        Err(_) => {
            let output = Command::new("python3")
                .args(["-c", "import faster_whisper"])
                .output();
            matches!(output, Ok(o) if o.status.success())
        }
    }
}

/// Check if Python is available
pub fn check_python_available() -> bool {
    Command::new("python")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
        || Command::new("python3")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

/// Get the Python executable that works
fn get_python_executable() -> &'static str {
    if Command::new("python").arg("--version").output().is_ok() {
        "python"
    } else {
        "python3"
    }
}

/// Install faster-whisper if not available
pub fn install_faster_whisper() -> Result<()> {
    println!("  Installing faster-whisper...");
    let python = get_python_executable();

    let status = Command::new(python)
        .args(["-m", "pip", "install", "faster-whisper"])
        .status()?;

    if status.success() {
        println!("  faster-whisper installed successfully.");
        Ok(())
    } else {
        Err(anyhow!("Failed to install faster-whisper"))
    }
}

/// Generate subtitle using faster-whisper (Python)
fn generate_subtitle_faster_whisper(
    video_file: &str,
    output_srt: &str,
    config: &SubtitleConfig,
) -> Result<()> {
    if !check_faster_whisper_available() {
        println!("  faster-whisper not found. Installing...");
        install_faster_whisper()?;
    }

    let python = get_python_executable();
    let model_name = config.model.to_string();
    let language = &config.language;

    let python_script = format!(
        r#"
import sys
from faster_whisper import WhisperModel

video_file = "{video_file}"
output_srt = "{output_srt}"
model_name = "{model_name}"
language = "{language}"

print(f"Loading Whisper model '{{model_name}}'...")
model = WhisperModel(model_name, device="cpu", compute_type="int8")

print("Transcribing audio...")
segments, info = model.transcribe(video_file, language=language)

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{{hours:02d}}:{{minutes:02d}}:{{secs:02d}},{{millis:03d}}"

print("Generating subtitle file...")
with open(output_srt, "w", encoding="utf-8") as f:
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()
        f.write(f"{{i}}\n")
        f.write(f"{{start_time}} --> {{end_time}}\n")
        f.write(f"{{text}}\n\n")

print("Subtitle generated successfully.")
"#,
        video_file = video_file.replace('\\', "\\\\").replace('"', "\\\""),
        output_srt = output_srt.replace('\\', "\\\\").replace('"', "\\\""),
        model_name = model_name,
        language = language,
    );

    println!(
        "  Generating subtitle with faster-whisper ({})...",
        model_name
    );

    let output = Command::new(python).args(["-c", &python_script]).output()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            println!("  {}", line);
        }
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(anyhow!("Failed to generate subtitle: {}", stderr))
    }
}

/// Generate subtitle using the configured backend
pub fn generate_subtitle(
    video_file: &str,
    output_srt: &str,
    config: &SubtitleConfig,
) -> Result<()> {
    if !config.enabled {
        return Ok(());
    }

    match config.backend {
        SubtitleBackend::WhisperCpp => {
            generate_subtitle_whisper_cpp(video_file, output_srt, config)
        }
        SubtitleBackend::FasterWhisper => {
            generate_subtitle_faster_whisper(video_file, output_srt, config)
        }
    }
}

/// Burn subtitle onto video using FFmpeg
pub fn burn_subtitle(video_file: &str, sub_file: &str, output_file: &str) -> Result<()> {
    let abs_sub_path = std::path::Path::new(sub_file)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(sub_file));

    // FFmpeg subtitle filter needs special escaping on Windows
    let subtitle_path = abs_sub_path
        .to_string_lossy()
        .replace('\\', "/")
        .replace(':', "\\:");

    // Detect if it's ASS or SRT based on extension
    let is_ass = sub_file.ends_with(".ass");

    let subtitle_filter = if is_ass {
        // For ASS files, use ass filter (preserves styling including karaoke effects)
        format!("ass='{}'", subtitle_path)
    } else {
        // For SRT files, use subtitles filter with styling
        format!(
            "subtitles='{}':force_style='FontName=Arial Black,FontSize=42,Bold=1,\
            PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,\
            BorderStyle=1,Outline=3,Shadow=2,MarginV=120'",
            subtitle_path
        )
    };

    println!("  Burning subtitle to video...");

    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-i", video_file])
        .args(["-vf", &subtitle_filter])
        .args(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "26"])
        .args(["-c:a", "copy"])
        .arg(output_file)
        .status()?;

    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("Failed to burn subtitle to video"))
    }
}

/// Process subtitle for a video clip
pub fn process_subtitle(
    cropped_file: &str,
    output_file: &str,
    config: &SubtitleConfig,
    index: usize,
) -> Result<String> {
    if !config.enabled {
        fs::rename(cropped_file, output_file)?;
        return Ok(output_file.to_string());
    }

    // Use ASS for whisper.cpp (word-by-word), SRT for faster-whisper
    let sub_ext = match config.backend {
        SubtitleBackend::WhisperCpp => "ass",
        SubtitleBackend::FasterWhisper => "srt",
    };
    let sub_file = format!("temp_{}.{}", index, sub_ext);

    match generate_subtitle(cropped_file, &sub_file, config) {
        Ok(_) => match burn_subtitle(cropped_file, &sub_file, output_file) {
            Ok(_) => {
                let _ = fs::remove_file(cropped_file);
                let _ = fs::remove_file(&sub_file);
                Ok(output_file.to_string())
            }
            Err(e) => {
                println!(
                    "  Failed to burn subtitle: {}. Using video without subtitle.",
                    e
                );
                let _ = fs::remove_file(&sub_file);
                fs::rename(cropped_file, output_file)?;
                Ok(output_file.to_string())
            }
        },
        Err(e) => {
            println!(
                "  Failed to generate subtitle: {}. Continuing without subtitle.",
                e
            );
            fs::rename(cropped_file, output_file)?;
            Ok(output_file.to_string())
        }
    }
}

/// Print subtitle backend status
pub fn print_subtitle_status() {
    println!("\n=== Subtitle Backend Status ===");

    if check_whisper_cpp_available() {
        println!("  [OK] whisper.cpp: Available");
        if let Some(binary) = get_whisper_cpp_binary() {
            println!("       Binary: {}", binary);
        }
    } else {
        println!("  [--] whisper.cpp: Not found");
        println!("       Download from: https://github.com/ggerganov/whisper.cpp/releases");
    }

    if check_python_available() {
        if check_faster_whisper_available() {
            println!("  [OK] faster-whisper: Available");
        } else {
            println!("  [--] faster-whisper: Not installed (run: pip install faster-whisper)");
        }
    } else {
        println!("  [--] Python: Not found");
    }

    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_model_from_input() {
        assert_eq!(WhisperModel::from_input("tiny"), Some(WhisperModel::Tiny));
        assert_eq!(WhisperModel::from_input("small"), Some(WhisperModel::Small));
        assert_eq!(WhisperModel::from_input("large"), Some(WhisperModel::Large));
        assert_eq!(WhisperModel::from_input("invalid"), None);
    }

    #[test]
    fn test_ggml_filename() {
        assert_eq!(WhisperModel::Small.ggml_filename(), "ggml-small.bin");
        assert_eq!(WhisperModel::Large.ggml_filename(), "ggml-large.bin");
    }

    #[test]
    fn test_subtitle_config_default() {
        let config = SubtitleConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.model, WhisperModel::Small);
        assert_eq!(config.language, "id");
    }
}
