//! Application entry point; sets up initial state, runs the episode loop using Agent and Controller.

use std::error::Error;

use agent::Agent;
use controller::Controller;
use steps::StepAction;

/// Initializes logging for the application.
fn init_logging() {
    // Initialize simple logging.
    #[cfg(feature = "logging")]
    env_logger::init();
}

/// Application entry point with error handling.
fn main() -> Result<(), Box<dyn Error>> {
    init_logging();

    // Parse task parameters from environment variables.
    let task = std::env::var("TASK").unwrap_or_else(|_| "ui_backend_latency".to_string());
    let env = std::env::var("ENV").unwrap_or_else(|_| "conversation_repair".to_string());
    let model = std::env::var("MODEL").unwrap_or_else(|_| "default".to_string());

    println!("[START] task={} env={} model={}", task, env, model);

    // Initialize agent and controller.
    let mut agent = Agent::new(&task, &env, &model)?;
    let mut controller = Controller::new();

    // Maximum number of steps per episode.
    const MAX_STEPS: u32 = 10;

    let mut step_count: u32 = 0;
    let mut total_reward: f64 = 0.0;
    let mut success: bool = false;

    // Episode loop.
    while step_count < MAX_STEPS {
        // Controller selects next action.
        let action = controller.next_action();

        // Agent executes the step.
        let result = agent.step(action)?;

        // Accumulate reward.
        total_reward += result.reward;

        // Log step details.
        println!(
            "[STEP] step={} action={:?}:facts_count={} conflict_alignments_count={} reward={:.2} done={} error={:?}",
            step_count + 1,
            action,
            result.facts_count,
            result.conflict_alignments_count,
            result.reward,
            result.done,
            result.error
        );

        // Update controller with step result.
        controller.update(&result);

        // Check for episode termination.
        if result.done {
            success = true;
            break;
        }

        step_count += 1;
    }

    // Log end of episode.
    println!(
        "[END] success={} steps={} score={:.2} rewards={}",
        success,
        step_count,
        total_reward,
        // Optionally list individual rewards if available.
    );
    Ok(())
}