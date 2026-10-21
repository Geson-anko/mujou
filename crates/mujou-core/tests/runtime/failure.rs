use std::sync::mpsc;

use super::*;

#[test]
fn an_update_error_ends_every_node_with_exception_then_stop() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, mut nodes } = probes(3);
        nodes[1].fail_on_update = Some((3, Failure::Error));
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        let mut result = Ok(());
        for _ in 0..3 {
            timer.advance_ns(H_NS).unwrap();
            result = rt.step(None).wait();
            if result.is_err() {
                break;
            }
        }
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!((failure.name.as_str(), failure.hook), ("p1", Hook::Update));
        assert!(rt.finished());
        for id in 0..3 {
            assert_exception_then_stop_once(&events(&log, id));
        }
        // Later requests report the same failure.
        assert!(matches!(shut_down(&rt), Err(RuntimeError::Failed(_))));
        assert!(matches!(rt.join(None), Err(RuntimeError::Failed(_))));
    });
}

#[test]
fn a_panic_is_a_failure() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, mut nodes } = probes(2);
        nodes[0].fail_on_update = Some((1, Failure::Panic));
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        let result = rt.step(None).wait();
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!((failure.name.as_str(), failure.hook), ("p0", Hook::Update));
        assert!(
            failure
                .error
                .to_string()
                .contains("panic in on_update of node 0")
        );
        for id in 0..2 {
            assert_exception_then_stop_once(&events(&log, id));
        }
    });
}

#[test]
fn a_failure_with_fewer_workers_than_nodes_ends_every_node() {
    let timer = virtual_timer();
    let Probes { log, mut nodes, .. } = probes(8);
    nodes[3].fail_on_update = Some((2, Failure::Error));
    let config = pool(2, config(nodes));
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    let mut ended = Ok(false);
    for _ in 0..100 {
        timer.advance_ns(H_NS).unwrap();
        ended = rt.join(Some(Duration::from_millis(20)));
        if !matches!(ended, Ok(false)) {
            break;
        }
    }
    let Err(RuntimeError::Failed(failure)) = ended else {
        panic!("expected a failure, got {ended:?}");
    };
    assert_eq!((failure.name.as_str(), failure.hook), ("p3", Hook::Update));
    for id in 0..8 {
        assert_exception_then_stop_once(&events(&log, id));
    }
}

#[test]
fn a_failed_start_in_the_pool_lets_no_node_update() {
    let timer = virtual_timer();
    let Probes { log, mut nodes, .. } = probes(4);
    nodes[2].fail_on_start = true;
    let config = pool(2, config(nodes));
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    let result = started.wait();
    let Err(RuntimeError::Failed(failure)) = result else {
        panic!("expected a failure, got {result:?}");
    };
    assert_eq!((failure.name.as_str(), failure.hook), ("p2", Hook::Start));
    assert!(rt.finished());
    for id in 0..4 {
        let events = events(&log, id);
        assert_eq!(events, [Event::Start, Event::Exception, Event::Stop]);
    }
}

#[test]
fn an_exception_after_a_normal_stop_still_comes_before_stop() {
    let timer = virtual_timer();
    let Probes { log, mut nodes, .. } = probes(2);
    let (entered_tx, entered) = mpsc::channel();
    let (release, release_rx) = mpsc::channel();
    nodes[0].gate = Some((entered_tx, release_rx));
    nodes[0].fail_on_update = Some((1, Failure::Error));
    let config = pool(2, config(nodes));
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    timer.advance_ns(H_NS).unwrap();
    entered.recv_timeout(LONG).unwrap();
    // Node 1 leaves its loop for the stop while node 0 is still updating.
    // The sleep only makes that order likely: the outcome is the same if
    // the exception comes first.
    rt.stop();
    thread::sleep(QUIET);
    release.send(()).unwrap();
    assert!(matches!(rt.join(Some(LONG)), Err(RuntimeError::Failed(_))));
    for id in 0..2 {
        assert_exception_then_stop_once(&events(&log, id));
    }
}

#[test]
fn a_stop_while_paused_ends_the_nodes_without_resume() {
    let timer = virtual_timer();
    let Probes { log, mut nodes, .. } = probes(3);
    nodes[0].fail_on_capture = true;
    let (rt, started) = Runtime::start(pool(1, manual(nodes)), clock(&timer)).unwrap();
    started.wait().unwrap();
    let result = rt.capture(|states| states, PAUSE).wait();
    let Err(RuntimeError::Failed(failure)) = result else {
        panic!("expected a failure, got {result:?}");
    };
    assert_eq!((failure.name.as_str(), failure.hook), ("p0", Hook::Capture));
    for id in 0..3 {
        let events = events(&log, id);
        assert!(!events.contains(&Event::Resume), "{events:?}");
        assert_exception_then_stop_once(&events);
    }
}

/// The failed node calls `exception` first, as a dedicated thread does
/// right after catching, even though its failure queued the others first.
#[test]
fn the_failed_node_calls_exception_before_the_nodes_its_failure_woke() {
    let timer = virtual_timer();
    let Probes { log, mut nodes } = probes(3);
    nodes[0].fail_on_update = Some((1, Failure::Error));
    let (rt, started) = Runtime::start(pool(1, manual(nodes)), clock(&timer)).unwrap();
    started.wait().unwrap();
    timer.advance_ns(H_NS).unwrap();
    assert!(rt.step(Some(&[0])).wait().is_err());
    let exceptions: Vec<_> = (log.lock().unwrap().iter())
        .filter(|(_, e, _)| *e == Event::Exception)
        .map(|(i, _, _)| *i)
        .collect();
    assert_eq!(exceptions, [0, 1, 2]);
}

#[test]
fn the_start_of_each_dt_is_known_when_starting_fails() {
    each_mode(|mode| {
        let timer = virtual_timer();
        timer.advance_ns(1_000_000_000).unwrap();
        let Probes { mut nodes, .. } = probes(2);
        nodes[0].fail_on_start = true;
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        assert!(started.wait().is_err());
        assert_eq!(rt.last_update_ns(), [1_000_000_000; 2]);
    });
}
