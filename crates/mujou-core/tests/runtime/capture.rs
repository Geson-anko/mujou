use std::sync::mpsc;
use std::time::Instant;

use super::*;

#[test]
fn builds_from_every_state_while_the_time_is_frozen() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, nodes } = probes(2);
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        let frozen = Arc::clone(&timer);
        let (states, paused) = rt
            .capture(move |states| (states, frozen.is_paused()), PAUSE)
            .wait()
            .unwrap();
        assert_eq!(states, vec![1, 1]);
        assert!(paused);
        assert!(!timer.is_paused());
        shut_down(&rt).unwrap();
        for id in 0..2 {
            let events = events(&log, id);
            assert_eq!(
                &events[2..5],
                &[Event::Pause, Event::Capture, Event::Resume]
            );
        }
    });
}

#[test]
fn a_capture_freezes_only_its_own_timer() {
    each_mode(|mode| {
        let (ta, tb) = (virtual_timer(), virtual_timer());
        let Probes { nodes, .. } = probes(1);
        let (a, started) = mode.start(manual(nodes), Arc::clone(&ta)).unwrap();
        started.wait().unwrap();
        let Probes { log: lb, nodes } = probes(1);
        let (b, started) = mode.start(manual(nodes), Arc::clone(&tb)).unwrap();
        started.wait().unwrap();
        let b = Arc::new(b);

        // While `a` is paused, the runtime on the other timer keeps updating.
        let (own, other, other_rt) = (Arc::clone(&ta), Arc::clone(&tb), Arc::clone(&b));
        let (own_paused, other_paused) = a
            .capture(
                move |_| {
                    other.advance_ns(H_NS).unwrap();
                    other_rt.step(None).wait().unwrap();
                    (own.is_paused(), other.is_paused())
                },
                PAUSE,
            )
            .wait()
            .unwrap();
        assert!(own_paused);
        assert!(!other_paused);
        assert_eq!(events(&lb, 0)[1], Event::Update(H_NS as f64 / 1e9));
        shut_down(&a).unwrap();
        shut_down(&b).unwrap();
    });
}

/// A capture while the only worker is in an update: the update runs to its
/// end, no update starts after the freeze, and every node's last update is
/// at or before the frozen time.
#[test]
fn a_capture_freezes_the_time_before_any_later_update_in_the_pool() {
    let timer = virtual_timer();
    let Probes { log, mut nodes } = probes(2);
    let (entered_tx, entered) = mpsc::channel();
    let (release, release_rx) = mpsc::channel();
    nodes[0].gate = Some((entered_tx, release_rx));
    let config = pool(1, config(nodes));
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    let t0 = timer.time_ns();
    timer.advance_ns(H_NS).unwrap();
    // Node 0 holds the worker; node 1 is due and waits for it.
    entered.recv_timeout(LONG).unwrap();
    let capture = rt.capture(|states| states, PAUSE);
    // The control thread freezes the time right after the pause request.
    let deadline = Instant::now() + LONG;
    while !timer.is_paused() {
        assert!(Instant::now() < deadline, "the capture did not freeze");
        thread::sleep(Duration::from_millis(1));
    }
    let frozen_ns = timer.time_ns();
    release.send(()).unwrap();
    assert_eq!(capture.wait().unwrap(), [1, 0]);
    let dt = Event::Update(H_NS as f64 / 1e9);
    let expected = [
        (0, Event::Start),
        (1, Event::Start),
        (0, dt),
        // Node 1 was due but pauses without updating.
        (1, Event::Pause),
        (0, Event::Pause),
        (0, Event::Capture),
        (1, Event::Capture),
    ];
    let log_now: Vec<_> = (log.lock().unwrap().iter())
        .map(|(i, e, _)| (*i, e.clone()))
        .take(expected.len())
        .collect();
    assert_eq!(log_now, expected);
    assert_eq!(frozen_ns, t0 + H_NS);
    assert!(rt.last_update_ns().iter().all(|&ns| ns <= frozen_ns));
    drop(release);
    shut_down(&rt).unwrap();
}

#[test]
fn a_capture_after_the_end_runs_on_the_calling_thread() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, nodes } = probes(2);
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        shut_down(&rt).unwrap();
        let caller = thread::current().name().map(str::to_owned);
        let (states, builder) = rt
            .capture(
                |states| (states, thread::current().name().map(str::to_owned)),
                PAUSE,
            )
            .wait()
            .unwrap();
        assert_eq!(states, [1, 1]);
        assert_eq!(builder, caller);
        let log = log.lock().unwrap();
        let captured: Vec<_> = (log.iter())
            .filter(|(_, e, _)| *e == Event::Capture)
            .map(|(_, _, thread)| Some(thread.clone()))
            .collect();
        assert_eq!(captured, [caller.clone(), caller]);
    });
}

#[test]
fn a_capture_after_an_exception_fails() {
    let timer = virtual_timer();
    let Probes { mut nodes, .. } = probes(1);
    nodes[0].fail_on_update = Some((1, Failure::Error));
    let (rt, started) = Runtime::start(manual(nodes), clock(&timer)).unwrap();
    started.wait().unwrap();
    timer.advance_ns(H_NS).unwrap();
    let _ = rt.step(None).wait();
    let _ = rt.join(None);
    let result = rt.capture(|states: Vec<usize>| states, PAUSE).wait();
    assert!(matches!(result, Err(RuntimeError::Failed(_))));
}

#[test]
fn a_failed_capture_after_a_normal_end_changes_nothing() {
    let timer = virtual_timer();
    let Probes { mut nodes, .. } = probes(1);
    nodes[0].fail_on_capture = true;
    let (rt, started) = Runtime::start(manual(nodes), clock(&timer)).unwrap();
    started.wait().unwrap();
    shut_down(&rt).unwrap();
    let result = rt.capture(|states: Vec<usize>| states, PAUSE).wait();
    let Err(RuntimeError::CaptureFailed(failure)) = result else {
        panic!("expected a failed capture, got {result:?}");
    };
    assert_eq!((failure.name.as_str(), failure.hook), ("p0", Hook::Capture));
    // The system ended normally, and still has.
    assert!(matches!(rt.join(None), Ok(true)));
}
