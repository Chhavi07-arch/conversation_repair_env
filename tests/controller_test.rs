//! Integration tests for `Controller::next_action()`.
//!
//! These tests verify the controller's decision logic when fed simulated
//! `StepResult` objects. The tests ensure that after an ALIGN step with
//! `done = true`, the next action is `StepAction::RESOLVE`, and that the
//! loop‑breaking condition (ALIGN not done) correctly leads to another
//! EXTRACT action.
//!
//! Assumes the crate is named `aigon_agent`. If the actual crate name
//! differs, replace `aigon_agent` below.

use aigon_agent::controller::Controller;
use aigon_agent::steps::{StepAction, StepResult};

/// Helper to create a `StepResult` with a given `done` flag.
fn step_result_done(done: bool) -> StepResult {
    StepResult {
        facts_count: 3,
        conflict_alignments_count: 0,
        reward: 0.0,
        done,
        error: None,
    }
}

/// Verifies that after an ALIGN step where `done = true` (all conflicts
/// resolved), the next action is `RESOLVE`.
#[test]
fn align_done_triggers_resolve() {
    let controller = Controller::new();
    let result = step_result_done(true);
    let action = controller.next_action(&result);
    assert_eq!(
        action,
        StepAction::RESOLVE,
        "Expected RESOLVE after ALIGN with done=true, got {:?}",
        action
    );
    println!("[TEST] align_done_triggers_resolve: action={:?}", action);
}

/// Verifies that after an ALIGN step where `done = false` (still conflicts
/// remain), the controller returns `EXTRACT` – i.e. the loop continues.
#[test]
fn align_not_done_loops_to_extract() {
    let controller = Controller::new();
    let result = step_result_done(false);
    let action = controller.next_action(&result);
    assert_eq!(
        action,
        StepAction::EXTRACT,
        "Expected EXTRACT after ALIGN with done=false, got {:?}",
        action
    );
    println!(
        "[TEST] align_not_done_loops_to_extract: action={:?}",
        action
    );
}

/// Verifies that after an EXTRACT step (represented here as a default
/// non‑done result) the controller also returns `EXTRACT`.
/// This is a basic sanity check to ensure the controller does not jump
/// to RESOLVE prematurely.
#[test]
fn extract_step_returns_extract() {
    let controller = Controller::new();
    // Simulate a typical EXTRACT result (not done).
    let result = step_result_done(false);
    let action = controller.next_action(&result);
    assert_eq!(
        action,
        StepAction::EXTRACT,
        "Expected EXTRACT after a regular step with done=false, got {:?}",
        action
    );
    println!(
        "[TEST] extract_step_returns_extract: action={:?}",
        action
    );
}