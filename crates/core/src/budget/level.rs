//! Map budget_score → R0..R4 → model tier → ranker difficulty. See SPEC-v3 §5.

use crate::budget::dims::MAX_BUDGET;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    R0,
    R1,
    R2,
    R3,
    R4,
}

/// Upper thresholds (exclusive) separating R0|R1|R2|R3; R4 is the open top (SPEC-v3 §5.1).
const THRESHOLDS: [f64; 4] = [4.0, 8.0, 12.0, 17.0];

impl Level {
    pub fn index(self) -> usize {
        match self {
            Level::R0 => 0,
            Level::R1 => 1,
            Level::R2 => 2,
            Level::R3 => 3,
            Level::R4 => 4,
        }
    }

    pub fn from_index(i: usize) -> Level {
        match i {
            0 => Level::R0,
            1 => Level::R1,
            2 => Level::R2,
            3 => Level::R3,
            _ => Level::R4,
        }
    }

    pub fn label(self) -> &'static str {
        ["R0", "R1", "R2", "R3", "R4"][self.index()]
    }

    pub fn tier(self) -> &'static str {
        ["tiny", "small", "medium", "strong", "best"][self.index()]
    }

    /// Step up/down, clamped to R0..R4.
    pub fn shift(self, delta: i32) -> Level {
        Level::from_index((self.index() as i32 + delta).clamp(0, 4) as usize)
    }

    /// The higher of two levels.
    pub fn max_with(self, other: Level) -> Level {
        Level::from_index(self.index().max(other.index()))
    }
}

/// Bucket a budget_score into a level.
pub fn level_of(score: f64) -> Level {
    if score < THRESHOLDS[0] {
        Level::R0
    } else if score < THRESHOLDS[1] {
        Level::R1
    } else if score < THRESHOLDS[2] {
        Level::R2
    } else if score < THRESHOLDS[3] {
        Level::R3
    } else {
        Level::R4
    }
}

/// `[lower, upper)` budget bounds of a level (R4 upper = MAX_BUDGET).
pub fn bounds(level: Level) -> (f64, f64) {
    match level {
        Level::R0 => (0.0, THRESHOLDS[0]),
        Level::R1 => (THRESHOLDS[0], THRESHOLDS[1]),
        Level::R2 => (THRESHOLDS[1], THRESHOLDS[2]),
        Level::R3 => (THRESHOLDS[2], THRESHOLDS[3]),
        Level::R4 => (THRESHOLDS[3], MAX_BUDGET),
    }
}

/// Raw estimator difficulty for the ranker: budget_score normalized to [0,1]
/// (SPEC-v3 §5.3). Monotonic in budget_score.
pub fn raw_difficulty(score: f64) -> f64 {
    (score / MAX_BUDGET).clamp(0.0, 1.0)
}

/// Difficulty floor implied by a (possibly escalated) level's lower bound.
pub fn level_floor_difficulty(level: Level) -> f64 {
    (bounds(level).0 / MAX_BUDGET).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_bucket_correctly() {
        assert_eq!(level_of(3.9), Level::R0);
        assert_eq!(level_of(4.0), Level::R1);
        assert_eq!(level_of(11.9), Level::R2);
        assert_eq!(level_of(12.0), Level::R3);
        assert_eq!(level_of(16.9), Level::R3);
        assert_eq!(level_of(17.0), Level::R4);
        assert_eq!(level_of(99.0), Level::R4);
    }

    #[test]
    fn tier_and_label_track_index() {
        assert_eq!(Level::R0.tier(), "tiny");
        assert_eq!(Level::R3.tier(), "strong");
        assert_eq!(Level::R4.label(), "R4");
    }

    #[test]
    fn shift_and_max_clamp() {
        assert_eq!(Level::R0.shift(-1), Level::R0);
        assert_eq!(Level::R4.shift(1), Level::R4);
        assert_eq!(Level::R1.shift(2), Level::R3);
        assert_eq!(Level::R1.max_with(Level::R3), Level::R3);
    }

    #[test]
    fn difficulty_is_in_unit_interval_and_monotonic() {
        assert!((raw_difficulty(0.0) - 0.0).abs() < 1e-9);
        assert!((raw_difficulty(MAX_BUDGET) - 1.0).abs() < 1e-9);
        assert!(raw_difficulty(20.0) > raw_difficulty(5.0));
        assert!(level_floor_difficulty(Level::R3) > level_floor_difficulty(Level::R1));
    }
}
