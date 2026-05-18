// src/control.rs
// Production-grade agent controller with step transition logic.
// Bug fix: after ALIGN, return Resolve when conflicts are zero.
// Simulation now dynamically reduces conflicts to enable resolution.

use std::fmt;
use std::time::Instant;
use log::{info, warn, error, debug};
use thiserror::Error;

/// Actions that the agent can perform.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Action {
    /// Extract facts from the environment.
    Extract,
    /// Align (resolve) conflicts between extracted facts.
    Align,
    /// Resolve the episode after all conflicts are resolved.
    Resolve,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Extract => write!(f, "EXTRACT"),
            Action::Align => write!(f, "ALIGN"),
            Action::Resolve => write!(f, "RESOLVE"),
        }
    }
}

/// Represents the state of a single episode.
#[derive(Debug, Clone)]
pub struct EpisodeState {
    /// Task description.
    pub task: String,
    /// Environment identifier.
    pub environment: String,
    /// Model name.
    pub model: String,
    /// Current step number (1-indexed).
    pub step: u32,
    /// Maximum number of steps allowed.
    pub max_steps: u32,
    /// Number of facts extracted so far.
    pub facts_count: u32,
    /// Number of conflicts remaining to align.
    pub conflict_alignments_count: u32,
    /// Accumulated reward (may go negative).
    pub reward: f64,
    /// Whether the episode is finished.
    pub done: bool,
    /// Optional error message from the last step.
    pub error: Option<String>,
}

impl EpisodeState {
    /// Creates a new episode state with initial values.
    ///
    /// # Arguments
    ///
    /// * `task` - Description of the task (must be non-empty).
    /// * `environment` - Environment identifier (must be non-empty).
    /// * `model` - Model name (must be non-empty).
    /// * `max_steps` - Maximum number of steps allowed (must be > 0).
    ///
    /// # Panics
    ///
    /// Panics if any string argument is empty or if `max_steps` is 0.
    pub fn new(task: String, environment: String, model: String, max_steps: u32) -> Self {
        assert!(!task.is_empty(), "task must not be empty");
        assert!(!environment.is_empty(), "environment must not be empty");
        assert!(!model.is_empty(), "model must not be empty");
        assert!(max_steps > 0, "max_steps must be greater than 0");
        Self {
            task,
            environment,
            model,
            step: 0,
            max_steps,
            facts_count: 0,
            conflict_alignments_count: 2, // initial conflicts for simulation
            reward: 0.0,
            done: false,
            error: None,
        }
    }

    /// Validates the state is consistent.
    ///
    /// Returns `Ok(())` if the state is valid, or an `AgentError` otherwise.
    fn validate(&self) -> Result<(), AgentError> {
        if self.step > self.max_steps {
            return Err(AgentError::InvalidState {
                detail: format!("step {} exceeds max_steps {}", self.step, self.max_steps),
            });
        }
        if self.reward.is_nan() || self.reward.is_infinite() {
            return Err(AgentError::InvalidState {
                detail: "reward is NaN or infinite".into(),
            });
        }
        if self.done && self.step == 0 {
            return Err(AgentError::InvalidState {
                detail: "episode marked done before any step".into(),
            });
        }
        Ok(())
    }
}

/// Possible errors during agent execution.
#[derive(Debug, Error)]
pub enum AgentError {
    /// Reached the maximum number of steps without success.
    #[error("Reached maximum steps ({0}) without success")]
    MaxStepsReached(u32),
    /// Attempted to act on an already finished episode.
    #[error("Episode already finished")]
    EpisodeFinished,
    /// Invalid action transition (e.g., trying to transition from `Resolve`).
    #[error("Invalid action transition from {from:?}")]
    InvalidTransition { from: Action },
    /// State inconsistency.
    #[error("Invalid state: {detail}")]
    InvalidState { detail: String },
    /// Errors originating from the simulator.
    #[error("Simulation error: {0}")]
    SimulationError(String),
}

/// Trait for simulating the effect of an action on the episode state.
pub trait Simulator: Send + Sync {
    /// Perform the action and update the state accordingly.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::SimulationError` if the simulation cannot be completed.
    fn simulate(&self, action: &Action, state: &mut EpisodeState) -> Result<(), AgentError>;
}

/// Default simulator that implements a realistic conflict resolution process.
///
/// - `Extract`: adds 3 facts and increases conflicts by 1.
/// - `Align`: resolves **all** remaining conflicts (sets count to 0).
/// - `Resolve`: marks the episode as done.
pub struct DefaultSimulator;

impl DefaultSimulator {
    /// Creates a new `DefaultSimulator`.
    pub fn new() -> Self {
        Self
    }
}

impl Simulator for DefaultSimulator {
    fn simulate(&self, action: &Action, state: &mut EpisodeState) -> Result<(), AgentError> {
        match action {
            Action::Extract => {
                state.facts_count += 3;
                state.conflict_alignments_count += 1;
                state.reward -= 0.20;
                info!(
                    "[STEP] step={} action=EXTRACT:facts_count={} reward={:.2} done=false error=null",
                    state.step, state.facts_count, state.reward
                );
            }
            Action::Align => {
                // Resolve all remaining conflicts at once (realistic assumption).
                if state.conflict_alignments_count > 0 {
                    state.conflict_alignments_count = 0;
                    state.reward += 0.20;
                } else {
                    warn!("ALIGN called with zero conflicts – nothing to resolve");
                }
                info!(
                    "[STEP] step={} action=ALIGN:conflict_alignments_count={} reward={:.2} done=false error=null",
                    state.step, state.conflict_alignments_count, state.reward
                );
            }
            Action::Resolve => {
                if state.done {
                    return Err(AgentError::SimulationError("Resolve called on already done episode".into()));
                }
                state.done = true;
                state.reward += 0.50;
                info!(
                    "[STEP] step={} action=RESOLVE:reward={:.2} done=true",
                    state.step, state.reward
                );
            }
        }
        state.validate()?;
        Ok(())
    }
}

/// Determines the next action based on the current step and state.
///
/// # Arguments
///
/// * `current_action` - The action that was just completed.
/// * `state` - The current episode state (including conflict counts, facts counts, etc.).
///
/// # Returns
///
/// * `Ok(Action)` - The next action to perform.
/// * `Err(AgentError)` - If the transition is invalid (e.g., already done).
///
/// # Errors
///
/// Returns `AgentError::EpisodeFinished` if `state.done` is true.
/// Returns `AgentError::InvalidTransition` if trying to move beyond `Resolve`.
pub fn decide_next_action(current_action: &Action, state: &EpisodeState) -> Result<Action, AgentError> {
    if state.done {
        return Err(AgentError::EpisodeFinished);
    }

    match current_action {
        Action::Extract => {
            // After extraction, always go to ALIGN to detect conflicts.
            Ok(Action::Align)
        }
        Action::Align => {
            // BUG FIX: Check if all conflicts have been resolved.
            // Original code always returned Action::Extract, causing endless loop.
            if state.conflict_alignments_count == 0 {
                info!("All conflicts resolved, transitioning to RESOLVE");
                Ok(Action::Resolve)
            } else {
                // If conflicts remain, go back to EXTRACT to gather more facts.
                info!("Conflicts remain ({}), transitioning to EXTRACT", state.conflict_alignments_count);
                Ok(Action::Extract)
            }
        }
        Action::Resolve => {
            // After RESOLVE, no further actions should be taken.
            Err(AgentError::InvalidTransition { from: Action::Resolve })
        }
    }
}

/// Runs a single episode from start to finish.
///
/// # Arguments
///
/// * `simulator` - The simulator that applies action effects.
/// * `initial_state` - The initial episode state.
///
/// # Returns
///
/// * `Ok(EpisodeState)` - The final state after the episode ends (success or max steps).
/// * `Err(AgentError)` - If an unrecoverable error occurs.
///
/// # Examples
///
///