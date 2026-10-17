//! How a runtime holds its timer: frozen for its own reasons, and counted as
//! live on it.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::time::{PauseHold, Timer};

/// Why the runtime holds the timer frozen. The runtime keeps one
/// [`PauseHold`] while any reason is left, so ending one reason cannot undo
/// another.
#[derive(Debug, Clone, Copy)]
pub(super) enum FreezeReason {
    /// From before the nodes start until all of them are ready to update.
    Startup = 1,
    /// A capture: from the pause request until the state is built.
    Capture = 2,
}

#[derive(Default)]
pub(super) struct Freeze {
    reasons: u8,
    hold: Option<PauseHold>,
}

impl Freeze {
    pub(super) fn hold(&mut self, reason: FreezeReason, timer: &Arc<Timer>) {
        if self.hold.is_none() {
            self.hold = Some(timer.hold());
        }
        self.reasons |= reason as u8;
    }

    pub(super) fn release(&mut self, reason: FreezeReason) {
        self.reasons &= !(reason as u8);
        if self.reasons == 0 {
            self.hold = None;
        }
    }

    pub(super) fn release_all(&mut self) {
        self.reasons = 0;
        self.hold = None;
    }
}

/// The number of runtimes started and not yet finished, per timer (by the
/// address of its `Arc`). A timer has an entry only while the number is
/// positive, and each counted runtime holds the timer through its [`Live`], so
/// the address cannot be reused by another timer while it has an entry.
static LIVE: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

pub(super) fn lock_live() -> MutexGuard<'static, BTreeMap<usize, usize>> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// One live runtime on this timer; dropping it uncounts it.
pub(super) struct Live(pub(super) Arc<Timer>);

impl Drop for Live {
    fn drop(&mut self) {
        let timer_id = Arc::as_ptr(&self.0) as usize;
        let mut live = lock_live();
        if let Some(k) = live.get_mut(&timer_id) {
            *k -= 1;
            if *k == 0 {
                live.remove(&timer_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freeze_holds_the_timer_until_every_reason_is_released() {
        let timer = Arc::new(Timer::new());
        let mut freeze = Freeze::default();
        freeze.hold(FreezeReason::Startup, &timer);
        freeze.hold(FreezeReason::Capture, &timer);
        // Ending the capture must not undo the freeze for the startup.
        freeze.release(FreezeReason::Capture);
        assert!(timer.is_paused());
        // Releasing a reason that is not held changes nothing.
        freeze.release(FreezeReason::Capture);
        assert!(timer.is_paused());
        freeze.release(FreezeReason::Startup);
        assert!(!timer.is_paused());
    }

    #[test]
    fn releasing_every_reason_keeps_a_pause_by_someone_else() {
        let timer = Arc::new(Timer::new());
        timer.pause();
        let mut freeze = Freeze::default();
        freeze.hold(FreezeReason::Startup, &timer);
        freeze.release_all();
        assert!(timer.is_paused());
    }
}
