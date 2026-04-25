use std::time::Duration;

pub const MIN_LEVEL: u8 = 1;
pub const MAX_LEVEL: u8 = 15;

pub fn fall_speed_seconds(level: u8) -> f32 {
    let level = level.clamp(MIN_LEVEL, MAX_LEVEL) as f32;
    (0.8 - ((level - 1.0) * 0.007)).powf(level - 1.0)
}

pub fn soft_drop_speed_seconds(level: u8) -> f32 {
    fall_speed_seconds(level) / 20.0
}

pub fn fall_duration(level: u8) -> Duration {
    Duration::from_secs_f32(fall_speed_seconds(level))
}

pub fn soft_drop_duration(level: u8) -> Duration {
    Duration::from_secs_f32(soft_drop_speed_seconds(level))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 0.001,
            "{actual} should be near {expected}"
        );
    }

    #[test]
    fn fall_speed_matches_guideline_reference_values() {
        assert_near(fall_speed_seconds(1), 1.000);
        assert_near(fall_speed_seconds(8), 0.135);
        assert_near(fall_speed_seconds(15), 0.007);
    }

    #[test]
    fn soft_drop_is_twenty_times_normal_fall_speed() {
        assert_near(soft_drop_speed_seconds(1), fall_speed_seconds(1) / 20.0);
        assert_near(soft_drop_speed_seconds(15), fall_speed_seconds(15) / 20.0);
    }

    #[test]
    fn level_is_clamped_to_guideline_range() {
        assert_eq!(fall_speed_seconds(0), fall_speed_seconds(MIN_LEVEL));
        assert_eq!(fall_speed_seconds(16), fall_speed_seconds(MAX_LEVEL));
    }
}
