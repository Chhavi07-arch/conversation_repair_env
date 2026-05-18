/// Enum representing the possible step actions in a simulator episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepAction {
    /// Extract facts from the environment.
    Extract,
    /// Align extracted facts to resolve conflicts.
    Align,
    /// Resolve remaining conflicts to produce a final output.
    Resolve,
    /// Finalize the episode after all conflicts are resolved.
    Finalize,
}

/// Struct representing the outcome of a single step execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepResult {
    /// Number of facts extracted during an EXTRACT step.
    pub facts_count: u32,
    /// Number of conflict alignments identified during an ALIGN step.
    pub conflict_alignments_count: u32,
    /// Reward signal from the step.
    pub reward: f64,
    /// Whether the episode is complete.
    pub done: bool,
    /// Optional error message if the step failed.
    pub error: Option<String>,
}