use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use std::sync::atomic::{AtomicBool, Ordering};

use mujou_core::time::{Timer, TimerError, TimerState, Timestamp, Wait, global, seconds_to_ns};

const SECOND: u64 = 1_000_000_000;

fn real_ns(d: Duration) -> f64 {
    d.as_nanos() as f64
}

/// Measures `scale` against the real clock by bracketing two timer reads with
/// real-clock reads: `scale * inner <= global elapsed <= scale * outer`.
fn assert_scaled_elapsed(timer: &Timer, scale: f64, real_wait: Duration) {
    let outer_start = Instant::now();
    let g0 = timer.time_ns();
    let inner_start = Instant::now();
    thread::sleep(real_wait);
    let inner_end = Instant::now();
    let g1 = timer.time_ns();
    let outer_end = Instant::now();

    let elapsed = (g1 - g0) as f64;
    // 1 ns of slack per read for the float-to-integer truncation.
    assert!(elapsed + 2.0 >= scale * real_ns(inner_end - inner_start));
    assert!(elapsed <= scale * real_ns(outer_end - outer_start) + 2.0);
}

mod real_clock {
    use super::*;

    #[test]
    fn advances_with_real_time_at_unit_scale() {
        let timer = Timer::new();
        assert_scaled_elapsed(&timer, 1.0, Duration::from_millis(5));
    }

    #[test]
    fn advances_at_the_set_scale() {
        let timer = Timer::new();
        timer.set_scale(250.0).unwrap();
        assert_eq!(timer.scale(), 250.0);
        assert_scaled_elapsed(&timer, 250.0, Duration::from_millis(5));
    }

    #[test]
    fn changing_the_scale_does_not_jump_the_time() {
        let timer = Timer::new();
        thread::sleep(Duration::from_millis(2));
        let real_start = Instant::now();
        let before = timer.time_ns();
        timer.set_scale(1000.0).unwrap();
        let after = timer.time_ns();
        let real_elapsed = real_ns(real_start.elapsed());
        // At most the new scale applies since `before`; no jump on the change.
        assert!((after - before) as f64 <= 1000.0 * real_elapsed + 2.0);
    }

    #[test]
    fn rejects_non_positive_or_non_finite_scales() {
        let timer = Timer::new();
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                timer.set_scale(scale),
                Err(TimerError::InvalidScale(_))
            ));
        }
        assert_eq!(timer.scale(), 1.0);
    }

    #[test]
    fn seconds_and_nanoseconds_agree() {
        let timer = Timer::new();
        timer.pause();
        assert_eq!(timer.time(), timer.time_ns() as f64 / 1e9);
    }

    #[test]
    fn sleep_waits_at_least_the_global_duration() {
        let timer = Timer::new();
        timer.set_scale(100.0).unwrap();
        let start = timer.time_ns();
        timer.sleep_ns(SECOND / 2);
        assert!(timer.time_ns() - start >= SECOND / 2);
    }

    #[test]
    fn rejects_advance() {
        let timer = Timer::new();
        assert!(matches!(
            timer.advance_ns(SECOND),
            Err(TimerError::NotVirtual)
        ));
    }
}

mod pause {
    use super::*;

    #[test]
    fn freezes_the_time() {
        let timer = Timer::new();
        timer.pause();
        assert!(timer.is_paused());
        let frozen = timer.time_ns();
        thread::sleep(Duration::from_millis(5));
        assert_eq!(timer.time_ns(), frozen);
    }

    #[test]
    fn resume_continues_from_the_frozen_time() {
        let timer = Timer::new();
        timer.set_scale(1000.0).unwrap();
        timer.pause();
        let frozen = timer.time_ns();
        thread::sleep(Duration::from_millis(5));
        let real_start = Instant::now();
        timer.resume();
        let resumed = timer.time_ns();
        let real_elapsed = real_ns(real_start.elapsed());
        assert!(!timer.is_paused());
        // The paused interval is not counted.
        assert!((resumed - frozen) as f64 <= 1000.0 * real_elapsed + 2.0);
    }

    #[test]
    fn sleep_does_not_return_until_resumed() {
        let timer = Arc::new(Timer::new());
        timer.set_scale(1000.0).unwrap();
        timer.pause();
        let target = timer.time_ns() + 1;
        // Paused: a long real wait does not reach even 1 ns ahead.
        assert!(!timer.wait_until_ns(target, Duration::from_millis(20)));

        let sleeper = {
            let timer = Arc::clone(&timer);
            thread::spawn(move || timer.sleep_ns(SECOND))
        };
        thread::sleep(Duration::from_millis(20));
        assert!(!sleeper.is_finished());
        timer.resume();
        sleeper.join().unwrap();
    }
}

mod hold {
    use super::*;

    #[test]
    fn the_time_runs_again_only_when_every_hold_is_dropped() {
        let timer = Arc::new(Timer::new());
        let first = timer.hold();
        let second = timer.hold();
        let frozen = timer.time_ns();
        drop(first);
        // Ending one freeze must not let the time run during the other one.
        assert!(timer.is_paused());
        thread::sleep(Duration::from_millis(2));
        assert_eq!(timer.time_ns(), frozen);
        drop(second);
        assert!(!timer.is_paused());
    }

    #[test]
    fn resume_does_not_undo_a_hold() {
        let timer = Arc::new(Timer::new());
        let hold = timer.hold();
        timer.pause();
        timer.resume();
        assert!(timer.is_paused());
        drop(hold);
        assert!(!timer.is_paused());
    }

    #[test]
    fn dropping_a_hold_does_not_undo_pause() {
        let timer = Arc::new(Timer::new());
        timer.pause();
        drop(timer.hold());
        assert!(timer.is_paused());
        timer.resume();
        assert!(!timer.is_paused());
    }

    #[test]
    fn a_held_virtual_clock_cannot_advance() {
        let timer = Arc::new(Timer::new());
        timer.use_virtual_clock(true);
        let _hold = timer.hold();
        assert!(matches!(timer.advance_ns(1), Err(TimerError::Paused)));
    }
}

mod virtual_clock {
    use super::*;

    fn virtual_timer() -> Timer {
        let timer = Timer::new();
        timer.use_virtual_clock(true);
        timer
    }

    #[test]
    fn stands_still_until_advanced() {
        let timer = virtual_timer();
        let start = timer.time_ns();
        thread::sleep(Duration::from_millis(5));
        assert_eq!(timer.time_ns(), start);
    }

    #[test]
    fn advances_by_exactly_the_given_amount_regardless_of_scale() {
        let timer = virtual_timer();
        timer.set_scale(3.0).unwrap();
        let start = timer.time_ns();
        timer.advance_ns(SECOND / 3).unwrap();
        timer.advance_ns(7).unwrap();
        assert_eq!(timer.time_ns(), start + SECOND / 3 + 7);
    }

    #[test]
    fn switching_clocks_keeps_the_time_continuous() {
        let timer = Timer::new();
        timer.set_scale(1000.0).unwrap();
        thread::sleep(Duration::from_millis(2));
        let before = timer.time_ns();
        timer.use_virtual_clock(true);
        let switched = timer.time_ns();
        assert!(switched >= before);
        thread::sleep(Duration::from_millis(5));
        assert_eq!(timer.time_ns(), switched);

        let real_start = Instant::now();
        timer.use_virtual_clock(false);
        let back = timer.time_ns();
        assert!(back >= switched);
        assert!((back - switched) as f64 <= 1000.0 * real_ns(real_start.elapsed()) + 2.0);
    }

    #[test]
    fn sleep_returns_once_advanced_by_the_duration() {
        let timer = Arc::new(virtual_timer());
        // The sleeper measures its own sleep, so the result does not depend on
        // when the thread starts relative to the advances below.
        let sleeper = {
            let timer = Arc::clone(&timer);
            thread::spawn(move || {
                let start = timer.time_ns();
                timer.sleep_ns(SECOND);
                timer.time_ns() - start
            })
        };
        while !sleeper.is_finished() {
            thread::sleep(Duration::from_millis(1));
            timer.advance_ns(SECOND / 10).unwrap();
        }
        assert!(sleeper.join().unwrap() >= SECOND);
    }

    #[test]
    fn wait_until_times_out_without_advance() {
        let timer = virtual_timer();
        let target = timer.time_ns() + 1;
        assert!(!timer.wait_until_ns(target, Duration::from_millis(10)));
        timer.advance_ns(1).unwrap();
        assert!(timer.wait_until_ns(target, Duration::ZERO));
    }

    #[test]
    fn wait_until_or_wakes_on_interrupt() {
        let timer = Arc::new(virtual_timer());
        let flag = Arc::new(AtomicBool::new(false));
        let waiter = {
            let (timer, flag) = (Arc::clone(&timer), Arc::clone(&flag));
            thread::spawn(move || {
                let target = timer.time_ns() + 1;
                timer.wait_until_ns_or(target, Duration::MAX, || flag.load(Ordering::SeqCst))
            })
        };
        // Whether the waiter is already asleep or not, it must see the flag.
        flag.store(true, Ordering::SeqCst);
        timer.interrupt_waiters();
        assert_eq!(waiter.join().unwrap(), Wait::Interrupted);
    }

    #[test]
    fn wait_until_or_wakes_at_the_target() {
        let timer = Arc::new(virtual_timer());
        let target = timer.time_ns() + 5;
        let waiter = {
            let timer = Arc::clone(&timer);
            thread::spawn(move || timer.wait_until_ns_or(target, Duration::MAX, || false))
        };
        timer.advance_ns(5).unwrap();
        assert_eq!(waiter.join().unwrap(), Wait::Reached);
        assert_eq!(
            timer.wait_until_ns_or(target + 1, Duration::from_millis(5), || false),
            Wait::TimedOut
        );
    }

    #[test]
    fn saturates_instead_of_wrapping_around() {
        let timer = virtual_timer();
        timer.advance_ns(u64::MAX).unwrap();
        timer.advance_ns(1).unwrap();
        assert_eq!(timer.time_ns(), u64::MAX);
    }

    #[test]
    fn sleep_beyond_the_representable_time_does_not_return_early() {
        let timer = Arc::new(virtual_timer());
        timer.advance_ns(SECOND).unwrap();
        let sleeper = {
            let timer = Arc::clone(&timer);
            thread::spawn(move || timer.sleep_ns(u64::MAX))
        };
        thread::sleep(Duration::from_millis(20));
        assert!(!sleeper.is_finished());
        timer.advance_ns(u64::MAX).unwrap();
        sleeper.join().unwrap();
    }

    #[test]
    fn rejects_advance_while_paused() {
        let timer = virtual_timer();
        timer.pause();
        assert!(matches!(timer.advance_ns(1), Err(TimerError::Paused)));
    }
}

mod state {
    use super::*;

    #[test]
    fn restores_the_time_and_scale() {
        let source = Timer::new();
        source.use_virtual_clock(true);
        source.set_scale(4.0).unwrap();
        let taken_ns = source.time_ns() + 12 * SECOND + 345;
        source.advance_ns(12 * SECOND + 345).unwrap();
        let state = source.state();
        assert_eq!(
            state,
            TimerState {
                time_ns: taken_ns,
                scale: 4.0
            }
        );

        let restored = Timer::new();
        restored.use_virtual_clock(true);
        restored.load_state(state).unwrap();
        assert_eq!(restored.time_ns(), taken_ns);
        assert_eq!(restored.scale(), 4.0);
    }

    #[test]
    fn real_clock_continues_from_the_restored_time() {
        let timer = Timer::new();
        let real_start = Instant::now();
        timer
            .load_state(TimerState {
                time_ns: 100 * SECOND,
                scale: 1.0,
            })
            .unwrap();
        let now = timer.time_ns();
        assert!(now >= 100 * SECOND);
        assert!((now - 100 * SECOND) as f64 <= real_ns(real_start.elapsed()) + 2.0);
    }

    #[test]
    fn rejects_an_invalid_scale() {
        let timer = Timer::new();
        let state = TimerState {
            time_ns: SECOND,
            scale: 0.0,
        };
        assert!(matches!(
            timer.load_state(state),
            Err(TimerError::InvalidScale(_))
        ));
    }
}

mod conversion {
    use super::*;

    #[test]
    fn rounds_seconds_to_the_nearest_nanosecond() {
        assert_eq!(seconds_to_ns(1.5).unwrap(), 1_500_000_000);
        assert_eq!(seconds_to_ns(1e-9).unwrap(), 1);
        assert_eq!(seconds_to_ns(0.0).unwrap(), 0);
    }

    #[test]
    fn rejects_negative_or_non_finite_seconds() {
        for seconds in [-1e-9, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                seconds_to_ns(seconds),
                Err(TimerError::InvalidDuration(_))
            ));
        }
    }
}

mod independent_timers {
    use super::*;

    /// A virtual clock at exactly 0 (the real clock ran until it was switched).
    fn virtual_timer() -> Timer {
        let timer = Timer::new();
        timer.use_virtual_clock(true);
        let state = TimerState {
            time_ns: 0,
            scale: 1.0,
        };
        timer.load_state(state).unwrap();
        timer
    }

    #[test]
    fn a_new_timer_starts_at_zero() {
        let before = Instant::now();
        let timer = Timer::new();
        let elapsed = timer.time_ns();
        assert!(elapsed as f64 <= real_ns(before.elapsed()));
    }

    #[test]
    fn advancing_one_does_not_move_the_other() {
        let (a, b) = (virtual_timer(), virtual_timer());
        a.advance_ns(3 * SECOND).unwrap();
        b.advance_ns(SECOND).unwrap();
        assert_eq!((a.time_ns(), b.time_ns()), (3 * SECOND, SECOND));
    }

    #[test]
    fn a_hold_freezes_only_its_own_timer() {
        let (a, b) = (Arc::new(virtual_timer()), Arc::new(virtual_timer()));
        let _hold = a.hold();
        assert!(a.is_paused());
        assert!(!b.is_paused());
        b.advance_ns(SECOND).unwrap();
        assert!(matches!(a.advance_ns(1), Err(TimerError::Paused)));
        assert_eq!((a.time_ns(), b.time_ns()), (0, SECOND));
    }

    #[test]
    fn global_is_one_shared_timer() {
        assert!(Arc::ptr_eq(global(), global()));
    }
}

mod timestamp {
    use super::*;

    #[test]
    fn converts_nanoseconds_to_seconds() {
        let ts = Timestamp::from_ns(2_500_000_001);
        assert_eq!(ts.as_ns(), 2_500_000_001);
        assert_eq!(ts.as_secs_f64(), 2_500_000_001.0 / 1e9);
        assert_eq!(ts.whole_secs(), 2);
    }

    #[test]
    fn subtracts_an_earlier_timestamp() {
        let (early, late) = (Timestamp::from_ns(3), Timestamp::from_ns(10));
        assert_eq!(late.checked_sub(early), Some(Timestamp::from_ns(7)));
        assert_eq!(early.checked_sub(late), None);
    }

    #[test]
    fn now_is_the_time_of_the_timer() {
        let timer = Timer::new();
        timer.use_virtual_clock(true);
        let ns = timer.time_ns();
        assert_eq!(timer.now(), Timestamp::from_ns(ns));
    }

    #[test]
    fn orders_by_time() {
        let (early, late) = (Timestamp::from_ns(1), Timestamp::from_ns(2));
        assert!(early < late);
        assert_eq!(early, Timestamp::from_ns(1));
    }
}
