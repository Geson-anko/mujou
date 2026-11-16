use super::*;

#[test]
fn steps_in_the_given_order_with_the_advanced_dt() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, nodes } = probes(3);
        let t0 = timer.time_ns();
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        for _ in 0..2 {
            timer.advance_ns(H_NS).unwrap();
            rt.step(Some(&[2, 0, 1])).wait().unwrap();
        }
        shut_down(&rt).unwrap();

        let order: Vec<usize> = (log.lock().unwrap().iter()).map(|(i, _, _)| *i).collect();
        // 3 starts in any order, then the updates in the given order, then 3 stops.
        assert_eq!(&order[3..9], &[2, 0, 1, 2, 0, 1]);
        for id in 0..3 {
            let dt = H_NS as f64 / 1e9;
            let expected = [
                Event::Start,
                Event::Update(dt),
                Event::Update(dt),
                Event::Stop,
            ];
            assert_eq!(events(&log, id), expected);
        }
        assert_eq!(rt.last_update_ns(), vec![t0 + 2 * H_NS; 3]);
    });
}

#[test]
fn a_step_without_time_passing_does_not_update() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, nodes } = probes(1);
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        rt.step(None).wait().unwrap();
        shut_down(&rt).unwrap();
        assert_eq!(events(&log, 0), [Event::Start, Event::Stop]);
    });
}

/// A step of a free-running runtime would never end: the caller broke the
/// contract.
#[test]
fn step_needs_the_manual_mode() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { nodes, .. } = probes(1);
        let (rt, started) = mode.start(config(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        assert!(matches!(
            rt.step(None).wait(),
            Err(RuntimeError::Internal(_))
        ));
        shut_down(&rt).unwrap();
    });
}

/// Every node updates once, at once, with the same Δt.
#[test]
fn an_all_nodes_step_updates_each_node_once_with_the_same_dt() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, nodes } = probes(4);
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        shut_down(&rt).unwrap();
        let dt = H_NS as f64 / 1e9;
        for id in 0..4 {
            assert_eq!(
                events(&log, id),
                [Event::Start, Event::Update(dt), Event::Stop]
            );
        }
    });
}

/// Two nodes that wait for each other in their updates both finish only if
/// they run in parallel: run one after another, the first would time out.
#[test]
fn an_all_nodes_step_runs_the_nodes_in_parallel_on_two_workers() {
    let timer = virtual_timer();
    let Probes { mut nodes, .. } = probes(2);
    let meeting = Meeting::new(2);
    for node in &mut nodes {
        node.meeting = Some(Arc::clone(&meeting));
    }
    let config = mode_config(Mode::TwoWorkers, manual(nodes));
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    timer.advance_ns(H_NS).unwrap();
    rt.step(None).wait().unwrap();
    shut_down(&rt).unwrap();
    assert_eq!(meeting.met(), 2);
}

/// The nodes of the pool and the one on a dedicated thread update in the same
/// parallel step.
#[test]
fn an_all_nodes_step_runs_the_pool_and_a_dedicated_thread_together() {
    let timer = virtual_timer();
    let Probes { mut nodes, .. } = probes(3);
    let meeting = Meeting::new(3);
    for node in &mut nodes {
        node.meeting = Some(Arc::clone(&meeting));
    }
    let mut config = mode_config(Mode::TwoWorkers, manual(nodes));
    config.nodes[2].dedicated_thread = true;
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    assert_eq!(rt.workers(), 2);
    timer.advance_ns(H_NS).unwrap();
    rt.step(None).wait().unwrap();
    shut_down(&rt).unwrap();
    assert_eq!(meeting.met(), 3);
}

/// The updates already running when one fails run to the end, then the step
/// fails and every node ends with `exception` and `stop`. (A node whose update
/// has not begun by then sees the stop and does not update.)
#[test]
fn a_failure_in_an_all_nodes_step_lets_the_running_updates_finish() {
    let timer = virtual_timer();
    let Probes { log, mut nodes } = probes(2);
    let meeting = Meeting::new(2);
    for node in &mut nodes {
        node.meeting = Some(Arc::clone(&meeting));
    }
    nodes[1].fail_on_update = Some((1, Failure::Error));
    let config = mode_config(Mode::TwoWorkers, manual(nodes));
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    timer.advance_ns(H_NS).unwrap();
    let result = rt.step(None).wait();
    let Err(RuntimeError::Failed(failure)) = result else {
        panic!("expected a failure, got {result:?}");
    };
    assert_eq!((failure.name.as_str(), failure.hook), ("p1", Hook::Update));
    assert_eq!(meeting.met(), 2);
    for id in 0..2 {
        let events = events(&log, id);
        assert!(events.iter().any(|e| matches!(e, Event::Update(_))));
        assert_exception_then_stop_once(&events);
    }
}

/// A step without time passing grants every node and returns without any
/// update.
#[test]
fn an_all_nodes_step_without_time_passing_updates_nobody() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, nodes } = probes(3);
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        rt.step(None).wait().unwrap();
        rt.step(None).wait().unwrap();
        shut_down(&rt).unwrap();
        for id in 0..3 {
            assert_eq!(events(&log, id), [Event::Start, Event::Stop]);
        }
    });
}
