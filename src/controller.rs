//! Controller module responsible for sequencing episode steps.
//!
//! The [`Controller`] struct decides the next [`StepAction`] based on the
//! current [`StepResult`]. It relies on the `done` flag to determine when
//! to transition to the final RESOLVE action.

use crate::steps::{StepAction, StepResult};
use log::{info, warn};

/// Tracks the action sequence and decides the next step.
///
/// The controller alternates between `EXTRACT` and `ALIGN` until
/// `StepResult::done` becomes true, at which point it returns `RESOLVE`.
pub struct Controller {
    /// The last action that was dispatched, used to enforce the alternating schedule.
    last_action: Option<StepAction>,
}

impl Controller {
    /// Creates a new controller with no prior action history.
    pub fn new() -> Self {
        Self { last_action: None }
    }

    /// Determines the next `StepAction` based on the current `StepResult`.
    ///
    /// # Arguments
    ///
    /// * `step_result` - The result of the last executed step.
    ///
    /// # Returns
    ///
    /// * `StepAction::RESOLVE` if [`StepResult::done`] is `true`.
    /// * Otherwise, the next action in the alternating `EXTRACT` → `ALIGN` cycle.
    ///
    /// # Logging
    ///
    /// Logs the chosen action and any errors present in the step result.
    pub fn next_action(&mut self, step_result: &StepResult) -> StepAction {
        if let Some(ref error) = step_result.error {
            warn!("Step encountered error: {}", error);
        }

        if step_result.done {
            info!("Episode done – proceeding to RESOLVE step");
            return StepAction::RESOLVE;
        }

        let next = self.determine_next();
        info!("Next action determined: {:?}", next);

        self.last_action = Some(next);
        next
    }

    /// Determines the next action based on the stored `last_action` state.
    fn determine_next(&self) -> StepAction {
        match self.last_action {
            None | Some(StepAction::ALIGN) => StepAction::EXTRACT,
            Some(StepAction::EXTRACT) => StepAction::ALIGN,
            // Should not occur under normal operation; fall back to EXTRACT.
            Some(StepAction::RESOLVE) => {
                warn!("Unexpected RESOLVE as last action; resetting to EXTRACT");
                StepAction::EXTRACT
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steps::StepResult;

    #[test]
    fn initial_action_is_extract() {
        let mut controller = Controller::new();
        let result = StepResult {
            facts_count: 0,
            conflict_alignments_count: 0,
            reward: 0.0,
            done: false,
            error: None,
        };
        assert_eq!(controller.next_action(&result), StepAction::EXTRACT);
    }

    #[test]
    fn alternates_extract_align() {
        let mut controller = Controller::new();
        let result = StepResult {
            facts_count: 0,
            conflict_alignments_count: 0,
            reward: 0.0,
            done: false,
            error: None,
        };

        // First call -> EXTRACT
        assert_eq!(controller.next_action(&result), StepAction::EXTRACT);
        // Second call -> ALIGN
        assert_eq!(controller.next_action(&result), StepAction::ALIGN);
        // Third call -> EXTRACT again
        assert_eq!(controller.next_action(&result), StepAction::EXTRACT);
    }

    #[test]
    fn resolves_when_done() {
        let mut controller = Controller::new();
        let result = StepResult {
            facts_count: 0,
            conflict_alignments_count: 0,
            reward: 0.0,
            done: true,
            error: None,
        };
        assert_eq!(controller.next_action(&result), StepAction::RESOLVE);
    }
}