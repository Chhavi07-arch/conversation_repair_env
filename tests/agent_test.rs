// tests/agent_test.rs
// Replace `your_crate_name` with the actual crate name from Cargo.toml.
use your_crate_name::agent::Agent;
use your_crate_name::steps::{StepAction, StepResult};

/// Tests that `handle_align` sets `done = true` when all conflicts are resolved.
#[cfg(test)]
mod agent_handle_align_tests {
    use super::*;

    /// Verify that with zero conflict alignments, the agent marks the step as done.
    #[test]
    fn test_handle_align_all_conflicts_resolved() {
        let mut agent = Agent::new();
        let mut step_result = StepResult {
            facts_count: 3,
            conflict_alignments_count: 0,
            reward: 0.0,
            done: false,
            error: None,
        };

        agent.handle_align(&mut step_result);

        assert!(
            step_result.done,
            "Expected done to be true when conflict_alignments_count == 0"
        );
    }

    /// Verify that with remaining conflicts, the step is not done.
    #[test]
    fn test_handle_align_conflicts_remaining() {
        let mut agent = Agent::new();
        let mut step_result = StepResult {
            facts_count: 3,
            conflict_alignments_count: 2,
            reward: 0.0,
            done: false,
            error: None,
        };

        agent.handle_align(&mut step_result);

        assert!(
            !step_result.done,
            "Expected done to remain false when conflicts remain"
        );
    }

    /// Verify that `handle_align` does not modify other fields incorrectly.
    #[test]
    fn test_handle_align_preserves_fields() {
        let mut agent = Agent::new();
        let mut step_result = StepResult {
            facts_count: 5,
            conflict_alignments_count: 0,
            reward: 0.5,
            done: false,
            error: None,
        };

        agent.handle_align(&mut step_result);

        assert_eq!(step_result.facts_count, 5);
        assert_eq!(step_result.conflict_alignments_count, 0);
        assert_eq!(step_result.reward, 0.5);
        assert!(step_result.error.is_none());
    }

    /// Integration: verify that `step()` with ALIGN action and zero conflicts leads to done==true.
    #[test]
    fn test_step_align_resolves() {
        let mut agent = Agent::new();
        let step_result = agent.step(StepAction::Align { conflicts_remaining: 0 });

        assert!(step_result.done, "step() with Align(0) should mark done");
    }
}