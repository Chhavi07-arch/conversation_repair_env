pub fn add_reward(&mut self, reward: f64) {
        if !reward.is_finite() {
            panic!("Reward must be a finite f64, got {}", reward);
        }
        self.score += reward;
        info!(
            "Reward {} added, total score: {:.2}",
            reward, self.score
        );
    }

    /// Marks the episode as done (either success or failure).
    ///
    /// # Arguments
    ///
    /// * `success` - `true` if the episode finished successfully, `false` otherwise.
    pub fn finish(&mut self, success: bool) {
        info!(
            "Episode finished. success={}, final step={}, score={:.2}",
            success, self.step, self.score
        );
        self.done = true;
    }

    /// Resets the state for a new episode, keeping the same `max_steps`.
    ///
    /// Useful for reusing the container allocation.
    pub fn reset(&mut self) {
        info!(
            "Resetting EpisodeState (max_steps={})",
            self.max_steps
        );
        self.step = 0;
        self.current_action = StepAction::Extract;
        self.last_action = None;
        self.conflict_alignments_count = 0;
        self.done = false;
        self.score = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state_valid() {
        let state = EpisodeState::new(10);
        assert_eq!(state.step, 0);
        assert_eq!(state.max_steps, 10);
        assert_eq!(state.current_action, StepAction::Extract);
        assert_eq!(state.last_action, None);
        assert_eq!(state.conflict_alignments_count, 0);
        assert!(!state.done);
        assert_eq!(state.score, 0.0);
    }

    #[test]
    fn test_advance_step_within_limit() {
        let mut state = EpisodeState::new(3);
        assert!(state.advance_step());
        assert_eq!(state.step, 1);
        assert!(!state.done);
    }

    #[test]
    fn test_advance_step_exceeds_limit() {
        let mut state = EpisodeState::new(1);
        assert!(state.advance_step()); // step 1
        assert!(!state.done);
        assert!(!state.advance_step()); // step 2, exceeds max=1
        assert_eq!(state.step, 2);
        assert!(state.done);
    }

    #[test]
    fn test_advance_step_already_done() {
        let mut state = EpisodeState::new(5);
        state.finish(true);
        assert!(state.done);
        assert!(!state.advance_step()); // should return false and not advance
        assert_eq!(state.step, 0); // step not incremented
    }

    #[test]
    fn test_set_next_action() {
        let mut state = EpisodeState::new(5);
        state.set_next_action(StepAction::Align);
        assert_eq!(state.last_action, Some(StepAction::Extract));
        assert_eq!(state.current_action, StepAction::Align);
    }

    #[test]
    fn test_add_reward() {
        let mut state = EpisodeState::new(5);
        state.add_reward(1.5);
        assert!((state.score - 1.5).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Reward must be a finite")]
    fn test_add_reward_nan() {
        let mut state = EpisodeState::new(5);
        state.add_reward(f64::NAN);
    }

    #[test]
    fn test_finish() {
        let mut state = EpisodeState::new(5);
        state.finish(true);
        assert!(state.done);
    }

    #[test]
    fn test_reset() {
        let mut state = EpisodeState::new(10);
        state.step = 5;
        state.current_action = StepAction::Resolve;
        state.last_action = Some(StepAction::Align);
        state.conflict_alignments_count = 3;
        state.done = true;
        state.score = 4.2;

        state.reset();

        assert_eq!(state.step, 0);
        assert_eq!(state.current_action, StepAction::Extract);
        assert_eq!(state.last_action, None);
        assert_eq!(state.conflict_alignments_count, 0);
        assert!(!state.done);
        assert_eq!(state.score, 0.0);
        assert_eq!(state.max_steps, 10); // unchanged
    }
}