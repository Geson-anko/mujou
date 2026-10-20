use super::*;

#[test]
fn steps_in_the_given_order_with_the_advanced_dt() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, nodes, names } = probes(3);
        let t0 = timer.time_ns();
        let (rt, started) = mode
            .start(nodes, names, Arc::clone(&timer), manual())
            .unwrap();
        started.wait().unwrap();
        for _ in 0..2 {
            timer.advance_ns(H_NS).unwrap();
            rt.step(Some(&[2, 0, 1])).wait().unwrap();
        }
        rt.shutdown().wait().unwrap();

        let order: Vec<usize> = (log.lock().unwrap().iter())
            .filter(|(_, e, _)| !matches!(e, Event::Enter(_)))
            .map(|(i, _, _)| *i)
            .collect();
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
        let Probes { log, nodes, names } = probes(1);
        let (rt, started) = mode
            .start(nodes, names, Arc::clone(&timer), manual())
            .unwrap();
        started.wait().unwrap();
        rt.step(None).wait().unwrap();
        rt.shutdown().wait().unwrap();
        assert_eq!(events(&log, 0), [Event::Start, Event::Stop]);
    });
}

#[test]
fn step_needs_the_manual_mode() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { nodes, names, .. } = probes(1);
        let config = RuntimeConfig::default();
        let (rt, started) = mode
            .start(nodes, names, Arc::clone(&timer), config)
            .unwrap();
        started.wait().unwrap();
        assert!(matches!(rt.step(None).wait(), Err(RuntimeError::NotManual)));
        rt.shutdown().wait().unwrap();
    });
}
