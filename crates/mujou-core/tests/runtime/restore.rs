use super::*;

#[test]
fn a_node_restores_its_state_right_after_start() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, mut nodes } = probes(2);
        nodes[0].state = Some(5);
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        let states = rt.capture(|states| states, PAUSE).wait().unwrap();
        assert_eq!(states, [5, 0]);
        shut_down(&rt).unwrap();
        let after_start = [Event::Pause, Event::Capture, Event::Resume, Event::Stop];
        assert_eq!(events(&log, 0)[..2], [Event::Start, Event::Restore(5)]);
        assert_eq!(events(&log, 0)[2..], after_start);
        // Nothing to restore: no call.
        assert_eq!(events(&log, 1)[..1], [Event::Start]);
        assert_eq!(events(&log, 1)[1..], after_start);
    });
}

#[test]
fn a_failed_start_does_not_restore() {
    each_mode(|mode| {
        let Probes { log, mut nodes } = probes(1);
        nodes[0].state = Some(5);
        nodes[0].fail_on_start = true;
        let (_rt, started) = mode.start(manual(nodes), virtual_timer()).unwrap();
        let result = started.wait();
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!(failure.hook, Hook::Start);
        assert_eq!(
            events(&log, 0),
            [Event::Start, Event::Exception, Event::Stop]
        );
    });
}

#[test]
fn a_failed_restore_is_reported_as_restore() {
    each_mode(|mode| {
        let Probes { log, mut nodes } = probes(1);
        nodes[0].state = Some(5);
        nodes[0].fail_on_restore = true;
        let (_rt, started) = mode.start(manual(nodes), virtual_timer()).unwrap();
        let result = started.wait();
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!((failure.name.as_str(), failure.hook), ("p0", Hook::Restore));
        let events = events(&log, 0);
        assert_eq!(
            events,
            [
                Event::Start,
                Event::Restore(5),
                Event::Exception,
                Event::Stop
            ]
        );
    });
}
