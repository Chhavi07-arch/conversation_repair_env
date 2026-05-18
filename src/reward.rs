// src/reward.rs
//
// Pure functions to compute rewards for each step.
// Not involved in the bug; provides reward values based on action and result.

use log::info;
use crate::steps::{StepAction, StepResult};

/// Compute a reward for an EXTRACT step.
///
/// # Arguments
/// * `facts_count` - Number of facts extracted in this step.
///
/// # Returns
/// A floating-point reward value. Base penalty of -0.5 plus 0.1 per fact.
pub fn compute_extract_reward(facts_count: u32) -> f64 {
    let reward: f64 = -0.5 + 0.1 * f64::from(facts_count);
    info!("EXTRACT reward: {:.2} (facts={})", reward, facts_count);
    reward
}

/// Compute a reward for an ALIGN step.
///
/// # Arguments
/// * `conflict_alignments_count` - Number of conflict alignments resolved.
///
/// # Returns
/// A floating-point reward value. Base penalty of -0.1 if no alignments,
/// otherwise +0.3 per alignment up to a maximum of 0.5.
pub fn compute_align_reward(conflict_alignments_count: u32) -> f64 {
    let reward: f64 = if conflict_alignments_count == 0 {
        -0.1
    } else {
        0.3 * f64::from(conflict_alignments_count).min(0.5)
    };
    info!("ALIGN reward: {:.2} (alignments={})", reward, conflict_alignments_count);
    reward
}

/// Compute a reward for a RESOLVE step.
///
/// # Returns
/// A fixed positive reward of 1.0 for successful resolution.
pub fn compute_resolve_reward() -> f64 {
    let reward: f64 = 1.0;
    info!("RESOLVE reward: {:.2}", reward);
    reward
}

/// Dispatch reward computation based on the current step action and its result.
///
/// # Arguments
/// * `action` - The `StepAction` of the current step.
/// * `result` - The `StepResult` containing counts and state.
///
/// # Returns
/// The computed reward for this step.
pub fn compute_reward(action: &StepAction, result: &StepResult) -> f64 {
    match action {
        StepAction::EXTRACT => compute_extract_reward(result.facts_count),
        StepAction::ALIGN => compute_align_reward(result.conflict_alignments_count),
        StepAction::RESOLVE => compute_resolve_reward(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steps::{StepAction, StepResult};

    #[test]
    fn test_extract_reward() {
        let reward = compute_extract_reward(3);
        assert!((reward + 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_align_reward_positive() {
        let reward = compute_align_reward(1);
        assert!((reward - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_align_reward_zero() {
        let reward = compute_align_reward(0);
        assert!((reward + 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_resolve_reward() {
        let reward = compute_resolve_reward();
        assert!((reward - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dispatch_extract() {
        let result = StepResult {
            facts_count: 3,
            conflict_alignments_count: 0,
            reward: 0.0,
            done: false,
            error: None,
        };
        let reward = compute_reward(&StepAction::EXTRACT, &result);
        assert!((reward + 0.2).abs() < 1e-6);
    }
}