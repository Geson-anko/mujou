use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mujou_core::time::{RunClock, Source, Timer, TimerError, Timestamp, global, seconds_to_ns};

const SECOND: u64 = 1_000_000_000;

/// Blocks for `dt_ns` of `timer`'s time. Waits through pauses.
fn sleep_ns(timer: &Timer, dt_ns: u64) {
    let target_ns = timer.time_ns().saturating_add(dt_ns);
    while !timer.wait_until_ns(target_ns, Duration::MAX) {}
}

/// A virtual clock at 0.
fn virtual_timer() -> Timer {
    Timer::new(Source::Virtual, Timestamp::from_ns(0))
}

/// A real clock at 0.
fn real_timer() -> Timer {
    Timer::new(Source::Real, Timestamp::from_ns(0))
}

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
        let timer = real_timer();
        assert_scaled_elapsed(&timer, 1.0, Duration::from_millis(5));
    }

    #[test]
    fn advances_at_the_set_scale() {
        let timer = real_timer();
        timer.set_time_scale(250.0).unwrap();
        assert_eq!(timer.time_scale(), 250.0);
        assert_scaled_elapsed(&timer, 250.0, Duration::from_millis(5));
    }

    #[test]
    fn changing_the_scale_does_not_jump_the_time() {
        let timer = real_timer();
        thread::sleep(Duration::from_millis(2));
        let real_start = Instant::now();
        let before = timer.time_ns();
        timer.set_time_scale(1000.0).unwrap();
        let after = timer.time_ns();
        let real_elapsed = real_ns(real_start.elapsed());
        // At most the new scale applies since `before`; no jump on the change.
        assert!((after - before) as f64 <= 1000.0 * real_elapsed + 2.0);
    }

    #[test]
    fn rejects_non_positive_or_non_finite_scales() {
        let timer = real_timer();
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                timer.set_time_scale(scale),
                Err(TimerError::InvalidScale(_))
            ));
        }
        assert_eq!(timer.time_scale(), 1.0);
    }

    #[test]
    fn sleep_waits_at_least_the_global_duration() {
        let timer = real_timer();
        timer.set_time_scale(100.0).unwrap();
        let start = timer.time_ns();
        sleep_ns(&timer, SECOND / 2);
        assert!(timer.time_ns() - start >= SECOND / 2);
    }

    #[test]
    fn advance_moves_the_real_clock_forward_at_once() {
        let timer = real_timer();
        let before = timer.time_ns();
        timer.advance_ns(100 * SECOND).unwrap();
        assert!(timer.time_ns() >= before + 100 * SECOND);
    }
}

mod pause {
    use super::*;

    #[test]
    fn freezes_the_time() {
        let timer = real_timer();
        timer.pause();
        assert!(timer.is_paused());
        let frozen = timer.time_ns();
        thread::sleep(Duration::from_millis(5));
        assert_eq!(timer.time_ns(), frozen);
    }

    #[test]
    fn resume_continues_from_the_frozen_time() {
        let timer = real_timer();
        timer.set_time_scale(1000.0).unwrap();
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
        let timer = Arc::new(real_timer());
        timer.set_time_scale(1000.0).unwrap();
        timer.pause();
        let target = timer.time_ns() + 1;
        // Paused: a long real wait does not reach even 1 ns ahead.
        assert!(!timer.wait_until_ns(target, Duration::from_millis(20)));

        let sleeper = {
            let timer = Arc::clone(&timer);
            thread::spawn(move || sleep_ns(&timer, SECOND))
        };
        thread::sleep(Duration::from_millis(20));
        assert!(!sleeper.is_finished());
        timer.resume();
        sleeper.join().unwrap();
    }
}

mod virtual_clock {
    use super::*;

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
        timer.set_time_scale(3.0).unwrap();
        let start = timer.time_ns();
        timer.advance_ns(SECOND / 3).unwrap();
        timer.advance_ns(7).unwrap();
        assert_eq!(timer.time_ns(), start + SECOND / 3 + 7);
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
                sleep_ns(&timer, SECOND);
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
            thread::spawn(move || sleep_ns(&timer, u64::MAX))
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

mod creation {
    use super::*;

    #[test]
    fn a_real_clock_runs_from_its_start() {
        let real_start = Instant::now();
        let timer = Timer::new(Source::Real, Timestamp::from_ns(100 * SECOND));
        let now = timer.time_ns();
        assert!(now >= 100 * SECOND);
        assert!((now - 100 * SECOND) as f64 <= real_ns(real_start.elapsed()) + 2.0);
        assert_eq!(timer.time_scale(), 1.0);
    }

    #[test]
    fn a_virtual_clock_stands_at_its_start() {
        let timer = Timer::new(Source::Virtual, Timestamp::from_ns(7));
        thread::sleep(Duration::from_millis(2));
        assert_eq!(timer.time_ns(), 7);
    }
}

mod run_clock {
    use super::*;

    #[test]
    fn reads_the_time_and_the_scale_of_its_timer() {
        let timer = Arc::new(virtual_timer());
        let clock = RunClock::new(Arc::clone(&timer));
        timer.set_time_scale(3.0).unwrap();
        timer.advance_ns(SECOND).unwrap();
        assert_eq!(clock.now(), Timestamp::from_ns(SECOND));
        assert_eq!(clock.now_ns(), SECOND);
        assert_eq!(clock.time_scale(), 3.0);
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

    #[test]
    fn a_new_timer_starts_at_zero() {
        let before = Instant::now();
        let timer = real_timer();
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
    fn the_time_between_two_is_signed_seconds() {
        let (early, late) = (Timestamp::from_ns(3), Timestamp::from_ns(2_500_000_010));
        assert_eq!(late.secs_since(early), 2_500_000_007.0 / 1e9);
        assert_eq!(early.secs_since(late), -2_500_000_007.0 / 1e9);
        assert_eq!(late.secs_since(late), 0.0);
        // The difference is taken in nanoseconds before it becomes a float, so
        // the far ends of the range differ by their exact count.
        let max = Timestamp::from_ns(u64::MAX);
        let zero = Timestamp::from_ns(0);
        assert_eq!(max.secs_since(zero), u64::MAX as f64 / 1e9);
        assert_eq!(zero.secs_since(max), -(u64::MAX as f64) / 1e9);
    }

    #[test]
    fn now_is_the_time_of_the_timer() {
        let timer = virtual_timer();
        // Away from 0, so that a `now` stuck at 0 would fail.
        timer.advance_ns(SECOND / 3).unwrap();
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
