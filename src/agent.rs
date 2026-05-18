pub fn new(metadata: HashMap<String, String>) -> Result<Self, AgentError> {
        if metadata.is_empty() {
            return Err(AgentError::MetadataError(
                "Episode metadata must contain at least one entry".into(),
            ));
        }
        if metadata.values().any(|v| v.is_empty()) {
            return Err(AgentError::MetadataError(
                "Metadata values cannot be empty".into(),
            ));
        }
        Ok(Self {
            current_step: Step::Extract,
            step_number: 0,
            reward: 0.0,
            remaining_conflicts: 0,
            done: false,
            episode_metadata: metadata,
            conflicts_set: false,
        })
    }

    /// Returns the current step.
    #[must_use]
    pub fn current_step(&self) -> Step {
        self.current_step
    }

    /// Returns the total number of steps taken (0‑based).
    #[must_use]
    pub fn step_number(&self) -> u64 {
        self.step_number
    }

    /// Returns the current accumulated reward.
    #[must_use]
    pub fn reward(&self) -> f64 {
        self.reward
    }

    /// Returns the number of conflicts remaining (only meaningful after an `Align` step).
    #[must_use]
    pub fn remaining_conflicts(&self) -> u64 {
        self.remaining_conflicts
    }

    /// Returns `true` if the episode has finished.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Returns an immutable reference to the episode metadata.
    #[must_use]
    pub fn episode_metadata(&self) -> &HashMap<String, String> {
        &self.episode_metadata
    }

    /// Adds a delta to the current reward.
    ///
    /// # Arguments
    ///
    /// * `delta` – The change in reward (must be a finite `f64`).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::InvalidReward`] if `delta` is NaN or infinite, or if the
    /// resulting accumulated reward becomes non‑finite.
    pub fn add_reward(&mut self, delta: f64) -> Result<(), AgentError> {
        if !delta.is_finite() {
            error!("Attempted to add non‑finite reward delta: {delta}");
            return Err(AgentError::InvalidReward(delta));
        }
        let new_reward = self.reward + delta;
        if !new_reward.is_finite() {
            error!(
                "Reward became non‑finite after adding {delta}: previous={}, new={new_reward}",
                self.reward
            );
            return Err(AgentError::InvalidReward(new_reward));
        }
        debug!("Reward updated: {} → {}", self.reward, new_reward);
        self.reward = new_reward;
        Ok(())
    }

    /// Sets the number of remaining conflicts after an `Align` step.
    ///
    /// This **must** be called exactly once after every `Align` step, before
    /// [`advance`](Self::advance) is invoked. It ensures the state machine can
    /// correctly decide the next step (→ `Resolve` if zero, → `Extract` otherwise).
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::StateMachineError`] if:
    /// * the current step is not `Align`, or
    /// * this method has already been called for the current `Align` step.
    pub fn set_remaining_conflicts(&mut self, count: u64) -> Result<(), AgentError> {
        if self.current_step != Step::Align {
            warn!(
                "set_remaining_conflicts called outside ALIGN step (current: {:?})",
                self.current_step
            );
            return Err(AgentError::StateMachineError(
                "Conflicts can only be set after ALIGN step".into(),
            ));
        }
        if self.conflicts_set {
            warn!("set_remaining_conflicts called twice for the same ALIGN step");
            return Err(AgentError::StateMachineError(
                "set_remaining_conflicts already called for this ALIGN step".into(),
            ));
        }
        self.remaining_conflicts = count;
        self.conflicts_set = true;
        debug!("Remaining conflicts set to {count}");
        Ok(())
    }

    /// Convenience method that combines [`add_reward`](Self::add_reward) and
    /// [`set_remaining_conflicts`](Self::set_remaining_conflicts), intended for
    /// use after the ALIGN step.
    ///
    /// # Arguments
    ///
    /// * `reward_delta` – Reward change for completing the ALIGN step.
    /// * `remaining_conflicts` – Number of conflicts still unresolved.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered from either operation.
    pub fn finish_align(&mut self, reward_delta: f64, remaining_conflicts: u64) -> Result<(), AgentError> {
        self.add_reward(reward_delta)?;
        self.set_remaining_conflicts(remaining_conflicts)
    }

    /// Computes the next step **without** mutating the state.
    ///
    /// # Errors
    ///
    /// Returns an [`AgentError::StateMachineError`] if:
    /// * the episode is already done,
    /// * the current step is `Align` but [`set_remaining_conflicts`](Self::set_remaining_conflicts)
    ///   has not been called.
    #[must_use]
    pub fn next_step(&self) -> Result<Option<Step>, AgentError> {
        if self.done {
            return Err(AgentError::StateMachineError(
                "Cannot compute next step from terminal state".into(),
            ));
        }
        if self.current_step == Step::Align && !self.conflicts_set {
            return Err(AgentError::StateMachineError(
                "Must call set_remaining_conflicts after ALIGN step before computing next step"
                    .into(),
            ));
        }
        let next = self.current_step.next(self.remaining_conflicts)?;
        Ok(match next {
            Step::Done => None,
            other => Some(other),
        })
    }

    /// Advances the state to the next step in the lifecycle.
    ///
    /// This method:
    /// 1. Checks that the transition is allowed (see [`next_step`](Self::next_step)).
    /// 2. Adds the given `reward_delta` to the accumulated reward.
    /// 3. Increments the step counter.
    /// 4. Sets `done` if the new step is `Done`.
    /// 5. Resets the `conflicts_set` guard for the next cycle.
    ///
    /// # Arguments
    ///
    /// * `reward_delta` – Reward change earned for completing the current step.
    ///
    /// # Returns
    ///
    /// The **new** current step after advancing. `None` if the episode has ended (`Done`).
    ///
    /// # Errors
    ///
    /// Returns an [`AgentError`] if:
    /// * the current step is `Done` (terminal),
    /// * the current step is `Align` but [`set_remaining_conflicts`](Self::set_remaining_conflicts)
    ///   has not been called,
    /// * the reward delta is non‑finite.
    pub fn advance(&mut self, reward_delta: f64) -> Result<Option<Step>, AgentError> {
        // Validate that we can move forward.
        let next = match self.next_step()? {
            Some(s) => s,
            None => return Ok(None), // Already at Done? But next_step would have errored if done.
        };

        // Consume the reward and update the step number.
        self.add_reward(reward_delta)?;
        self.step_number = self
            .step_number
            .checked_add(1)
            .ok_or_else(|| AgentError::StateMachineError("Step number overflow".into()))?;

        // Update the current step and terminal flag.
        let old_step = self.current_step;
        self.current_step = next;
        if self.current_step.is_terminal() {
            self.done = true;
            info!("Episode reached terminal state after {} steps", self.step_number);
        }

        // Reset the conflicts guard for the next ALIGN step.
        self.conflicts_set = false;

        debug!(
            "Step transition: {:?} → {:?} (step {})",
            old_step, self.current_step, self.step_number
        );
        Ok(if self.done { None } else { Some(self.current_step) })
    }
}

// -----------------------------------------------------------------------------
// Unit tests
// -----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_metadata() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("task".into(), "test".into());
        m
    }

    #[test]
    fn test_initial_state() {
        let state = AgentState::new(sample_metadata()).unwrap();
        assert_eq!(state.current_step(), Step::Extract);
        assert_eq!(state.step_number(), 0);
        assert_eq!(state.reward(), 0.0);
        assert_eq!(state.remaining_conflicts(), 0);
        assert!(!state.is_done());
    }

    #[test]
    fn test_empty_metadata_rejected() {
        let err = AgentState::new(HashMap::new()).unwrap_err();
        assert!(matches!(err, AgentError::MetadataError(_)));
    }

    #[test]
    fn test_align_to_resolve_when_zero_conflicts() {
        let mut state = AgentState::new(sample_metadata()).unwrap();
        // EXTRACT → ALIGN (reward +10)
        state.advance(10.0).unwrap();
        assert_eq!(state.current_step(), Step::Align);

        // ALIGN, no conflicts remaining.
        state.finish_align(5.0, 0).unwrap();
        let next = state.advance(0.0).unwrap();
        assert_eq!(next, Some(Step::Resolve));
        assert_eq!(state.reward(), 15.0);
    }

    #[test]
    fn test_align_to_extract_when_conflicts_remain() {
        let mut state = AgentState::new(sample_metadata()).unwrap();
        state.advance(0.0).unwrap();
        assert_eq!(state.current_step(), Step::Align);

        state.finish_align(0.0, 2).unwrap();
        let next = state.advance(0.0).unwrap();
        assert_eq!(next, Some(Step::Extract)); // back to Extract
    }

    #[test]
    fn test_resolve_to_done() {
        let mut state = AgentState::new(sample_metadata()).unwrap();
        // EXTRACT → ALIGN
        state.advance(0.0).unwrap();
        // ALIGN → RESOLVE (0 conflicts)
        state.finish_align(0.0, 0).unwrap();
        state.advance(0.0).unwrap();
        assert_eq!(state.current_step(), Step::Resolve);

        // RESOLVE → DONE
        let next = state.advance(10.0).unwrap();
        assert!(next.is_none());
        assert!(state.is_done());
        assert_eq!(state.reward(), 10.0);
    }

    #[test]
    fn test_advance_without_setting_conflicts_fails() {
        let mut state = AgentState::new(sample_metadata()).unwrap();
        state.advance(0.0).unwrap(); // now at ALIGN
        // Forget to call set_remaining_conflicts
        let err = state.advance(0.0).unwrap_err();
        assert!(matches!(err, AgentError::StateMachineError(_)));
    }

    #[test]
    fn test_double_set_conflicts_fails() {
        let mut state = AgentState::new(sample_metadata()).unwrap();
        state.advance(0.0).unwrap(); // ALIGN
        state.set_remaining_conflicts(1).unwrap();
        let err = state.set_remaining_conflicts(0).unwrap_err();
        assert!(matches!(err, AgentError::StateMachineError(_)));
    }

    #[test]
    fn test_non_finite_reward_rejected() {
        let mut state = AgentState::new(sample_metadata()).unwrap();
        let err = state.add_reward(f64::NAN).unwrap_err();
        assert!(matches!(err, AgentError::InvalidReward(_)));

        let err = state.add_reward(f64::INFINITY).unwrap_err();
        assert!(matches!(err, AgentError::InvalidReward(_)));
    }

    #[test]
    fn test_advance_from_done_fails() {
        let mut state = AgentState::new(sample_metadata()).unwrap();
        // Quickly reach Done
        state.advance(0.0).unwrap(); // EXTRACT → ALIGN
        state.finish_align(0.0, 0).unwrap(); // ALIGN → RESOLVE
        state.advance(0.0).unwrap(); // RESOLVE → DONE
        let err = state.advance(0.0).unwrap_err();
        assert!(matches!(err, AgentError::StateMachineError(_)));
    }

    #[test]
    fn test_step_display() {
        assert_eq!(Step::Extract.to_string(), "EXTRACT");
        assert_eq!(Step::Align.to_string(), "ALIGN");
        assert_eq!(Step::Resolve.to_string(), "RESOLVE");
        assert_eq!(Step::Done.to_string(), "DONE");
    }
}