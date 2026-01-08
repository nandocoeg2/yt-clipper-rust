use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// Height for top section (center content) in split mode
pub const TOP_HEIGHT: u32 = 960;

/// Height for bottom section (facecam) in split mode
pub const BOTTOM_HEIGHT: u32 = 350;

/// Output video dimensions
pub const OUTPUT_WIDTH: u32 = 720;
pub const OUTPUT_HEIGHT: u32 = 1280;

/// Crop mode for video processing
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize, Display, EnumString)]
#[strum(serialize_all = "kebab-case")]
#[serde(rename_all = "kebab-case")]
pub enum CropMode {
    /// Standard center crop - takes center portion of video
    #[default]
    Default,
    /// Split crop: top = center content, bottom = bottom-left corner (facecam)
    SplitLeft,
    /// Split crop: top = center content, bottom = bottom-right corner (facecam)
    SplitRight,
}

impl CropMode {
    /// Get the FFmpeg video filter string for this crop mode
    pub fn get_ffmpeg_filter(&self) -> String {
        match self {
            CropMode::Default => {
                // Scale to cover 720x1280 (maintains aspect ratio, ensures both dimensions are >= target)
                // Then center crop to exactly 720x1280
                format!(
                    "scale={}:{}:force_original_aspect_ratio=increase,crop={}:{}",
                    OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_HEIGHT
                )
            }
            CropMode::SplitLeft => {
                // Split crop: top = center of video, bottom = bottom-left corner (facecam)
                //
                // Strategy:
                // 1. Scale video to fixed height (1280) to ensure we have enough pixels
                // 2. Split the SCALED video (before any cropping)
                // 3. Crop center region for top section (720x960)
                // 4. Crop bottom-left corner for facecam (720x350)
                // 5. Stack vertically
                //
                // For a 16:9 video scaled to height 1280:
                //   - Width becomes ~2276
                //   - Top crop: center of video (x=(2276-720)/2, y=(1280-960)/2)
                //   - Bottom crop: bottom-left (x=0, y=1280-350=930)
                format!(
                    "scale=-2:{}[scaled];\
                    [scaled]split=2[s1][s2];\
                    [s1]crop={}:{}:(iw-{})/2:(ih-{})/2[top];\
                    [s2]crop={}:{}:0:ih-{}[bottom];\
                    [top][bottom]vstack=inputs=2[out]",
                    OUTPUT_HEIGHT,  // Scale to height 1280
                    OUTPUT_WIDTH, TOP_HEIGHT, OUTPUT_WIDTH, TOP_HEIGHT,  // Center crop 720x960
                    OUTPUT_WIDTH, BOTTOM_HEIGHT, BOTTOM_HEIGHT  // Bottom-left crop 720x350
                )
            }
            CropMode::SplitRight => {
                // Split crop: top = center of video, bottom = bottom-right corner (facecam)
                //
                // Same as SplitLeft but facecam from bottom-right instead
                format!(
                    "scale=-2:{}[scaled];\
                    [scaled]split=2[s1][s2];\
                    [s1]crop={}:{}:(iw-{})/2:(ih-{})/2[top];\
                    [s2]crop={}:{}:iw-{}:ih-{}[bottom];\
                    [top][bottom]vstack=inputs=2[out]",
                    OUTPUT_HEIGHT,  // Scale to height 1280
                    OUTPUT_WIDTH, TOP_HEIGHT, OUTPUT_WIDTH, TOP_HEIGHT,  // Center crop 720x960
                    OUTPUT_WIDTH, BOTTOM_HEIGHT, OUTPUT_WIDTH, BOTTOM_HEIGHT  // Bottom-right crop 720x350
                )
            }
        }
    }

    /// Check if this mode uses complex filter (requires -filter_complex instead of -vf)
    pub fn is_complex_filter(&self) -> bool {
        matches!(self, CropMode::SplitLeft | CropMode::SplitRight)
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            CropMode::Default => "Default (center crop)",
            CropMode::SplitLeft => "Split (top: center, bottom: bottom-left facecam)",
            CropMode::SplitRight => "Split (top: center, bottom: bottom-right facecam)",
        }
    }

    /// Parse from user input (1, 2, 3 or string names)
    pub fn from_input(input: &str) -> Option<Self> {
        match input.trim().to_lowercase().as_str() {
            "1" | "default" => Some(CropMode::Default),
            "2" | "split-left" | "split_left" | "splitleft" => Some(CropMode::SplitLeft),
            "3" | "split-right" | "split_right" | "splitright" => Some(CropMode::SplitRight),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_mode_from_input() {
        assert_eq!(CropMode::from_input("1"), Some(CropMode::Default));
        assert_eq!(CropMode::from_input("2"), Some(CropMode::SplitLeft));
        assert_eq!(CropMode::from_input("3"), Some(CropMode::SplitRight));
        assert_eq!(CropMode::from_input("default"), Some(CropMode::Default));
        assert_eq!(CropMode::from_input("split-left"), Some(CropMode::SplitLeft));
        assert_eq!(CropMode::from_input("invalid"), None);
    }

    #[test]
    fn test_is_complex_filter() {
        assert!(!CropMode::Default.is_complex_filter());
        assert!(CropMode::SplitLeft.is_complex_filter());
        assert!(CropMode::SplitRight.is_complex_filter());
    }
}
