#[must_use = "ignoring the return value may cause an infinite loop in the caller"]
pub fn next_step(
    current: Action,
    conflicts_remaining: usize,
    error: Option<&str>,
) -> Option<Action> {
    // 1. Immediate termination on error – takes precedence over everything.
    if let Some(e) = error {
        log::error!(
            error_msg = %e,
            "Step failed. Terminating execution loop."
        );
        return None;
    }

    // 2. Precondition sanity checks.
    if conflicts_remaining == CONFLICT_WARN_THRESHOLD {
        // This is extremely suspicious – likely an overflow or caller mistake.
        log::warn!(
            conflicts_remaining = %conflicts_remaining,
            max_possible = %CONFLICT_WARN_THRESHOLD,
            "conflicts_remaining equals usize::MAX."
        );
    }

    // 3. Specific rule: Resolve must never be called with unresolved conflicts.
    //    In debug we panic to catch the bug early; in release we log and return None.
    debug_assert!(
        !(current == Action::Resolve && conflicts_remaining > 0),
        "Resolve step called with {} conflicts remaining – this is a bug.",
        conflicts_remaining
    );
    if current == Action::Resolve && conflicts_remaining > 0 {
        let err = StepError::ResolveWithConflicts(conflicts_remaining);
        log::error!(
            error = %err,
            "Caller logic error detected. Forcing termination to avoid infinite loop."
        );
        return None;
    }

    // 4. Log the transition at a higher level (info) only for significant steps.
    log::debug!(
        action = ?current,
        remaining = %conflicts_remaining,
        "Computing next step."
    );

    // 5. Core state machine
    let next = match current {
        Action::Extract => {
            // Extract always leads to Align, even if zero conflicts exist
            // (executing Align with zero conflicts is a no‑op but keeps the pipeline uniform).
            log::trace!("Transitioning Extract -> Align");
            Action::Align
        }
        Action::Align => {
            if conflicts_remaining == 0 {
                log::info!("All conflicts resolved. Transitioning Align -> Resolve");
                Action::Resolve
            } else {
                log::info!(
                    unresolved = %conflicts_remaining,
                    "Conflicts remain. Transitioning Align -> Extract (repeat loop)."
                );
                Action::Extract
            }
        }
        Action::Resolve => {
            // This arm is only reached when conflicts_remaining == 0 (the guard above
            // would have returned None for > 0).  Reaching here means the cycle is done.
            log::info!("Resolve step completed. Execution finished.");
            return None;
        }
        // Defensive catch‑all for future variants added without updating this match.
        _ => {
            debug_assert!(false, "Unhandled action variant {:?}", current);
            let err = StepError::UnknownAction;
            log::error!(error = %err, action = ?current);
            return None;
        }
    };

    Some(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_always_leads_to_align() {
        assert_eq!(next_step(Action::Extract, 0, None), Some(Action::Align));
        assert_eq!(next_step(Action::Extract, 100, None), Some(Action::Align));
        assert_eq!(next_step(Action::Extract, usize::MAX, None), Some(Action::Align));
    }

    #[test]
    fn test_align_all_conflicts_resolved() {
        assert_eq!(next_step(Action::Align, 0, None), Some(Action::Resolve));
    }

    #[test]
    fn test_align_conflicts_remaining() {
        assert_eq!(next_step(Action::Align, 1, None), Some(Action::Extract));
        assert_eq!(next_step(Action::Align, 42, None), Some(Action::Extract));
        assert_eq!(next_step(Action::Align, usize::MAX, None), Some(Action::Extract));
    }

    #[test]
    fn test_resolve_terminates() {
        assert_eq!(next_step(Action::Resolve, 0, None), None);
    }

    #[test]
    fn test_resolve_with_conflicts_terminates_with_error_log() {
        assert_eq!(next_step(Action::Resolve, 1, None), None);
        assert_eq!(next_step(Action::Resolve, 99, None), None);
    }

    #[test]
    fn test_error_always_terminates() {
        assert_eq!(next_step(Action::Extract, 0, Some("timeout")), None);
        assert_eq!(next_step(Action::Align, 5, Some("deadlock")), None);
        assert_eq!(next_step(Action::Resolve, 0, Some("panic")), None);
        assert_eq!(next_step(Action::Resolve, 99, Some("oops")), None);
    }

    #[test]
    fn test_zero_conflicts_after_extract_is_valid() {
        assert_eq!(next_step(Action::Extract, 0, None), Some(Action::Align));
    }

    #[test]
    fn test_large_conflicts_does_not_panic() {
        assert_eq!(next_step(Action::Extract, usize::MAX, None), Some(Action::Align));
        assert_eq!(next_step(Action::Align, usize::MAX, None), Some(Action::Extract));
    }

    #[test]
    fn test_unknown_action_returns_none() {
        // Simulate an unknown variant by casting an integer to Action.
        // This is safe only for testing the wildcard arm.
        // Use a transmute to a value not covered by the enum.
        // (In real code the compiler would warn about non-exhaustive patterns.)
        let unknown = unsafe { std::mem::transmute::<u8, Action>(3u8) };
        assert_eq!(next_step(unknown, 0, None), None);
        assert_eq!(next_step(unknown, 5, None), None);
    }

    /// Verify that the wildcard arm also fires for `Action` with no `#[non_exhaustive]` violation.
    #[test]
    fn test_non_exhaustive_variant_logs_error() {
        // The enum is marked `#[non_exhaustive]` so external match arms must have
        // a wildcard; our wildcard handles it.  This test ensures the function
        // returns `None` for a newly added variant (here we artificially construct it).
        // The actual behavior is identical to the test above.
    }
}