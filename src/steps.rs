/// Represents the possible actions an agent can take in a step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepAction {
    /// Extracts facts from the conversation.
    EXTRACT,
    /// Aligns conflicting facts.
    ALIGN,
    /// Resolves the episode.
    RESOLVE,
}

/// Represents the result of executing a step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Number of facts extracted.
    pub facts_count: u32,
    /// Number of conflict alignments performed.
    pub conflict_alignments_count: u32,
    /// Reward value for this step.
    pub reward: f64,
    /// Whether the episode is done.
    pub done: bool,
    /// Optional error message.
    pub error: Option<String>,
}