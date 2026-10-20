use std::sync::mpsc;

use super::*;

/// Tries to move the time in `start`, as a slow start would on the real
/// clock, and keeps whether the timer was held and whether it moved.
struct Starter {
    timer: Arc<Timer>,
    seen: Arc<Mutex<Option<(bool, bool)>>>,
    dts: Arc<Mutex<Vec<f64>>>,
}

impl NodeHooks for Starter {
    type State = ();

    fn start(&mut self) -> Result<(), HookError> {
        let moved = self.timer.advance_ns(H_NS).is_ok();
        *self.seen.lock().unwrap() = Some((self.timer.is_paused(), moved));
        Ok(())
    }

    fn update(&mut self, dt: f64) -> Result<(), HookError> {
        self.dts.lock().unwrap().push(dt);
        Ok(())
    }

    fn pause(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    fn resume(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    fn exception(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    fn stop(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    fn state_dict(&mut self) -> Result<(), HookError> {
        Ok(())
    }
}

#[test]
fn the_time_is_held_while_the_nodes_start() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let seen = Arc::new(Mutex::new(None));
        let dts = Arc::new(Mutex::new(Vec::new()));
        let node = Starter {
            timer: Arc::clone(&timer),
            seen: Arc::clone(&seen),
            dts: Arc::clone(&dts),
        };
        let t0 = timer.time_ns();
        let (rt, started) = mode
            .start(
                vec![node],
                vec!["n".to_owned()],
                Arc::clone(&timer),
                manual(),
            )
            .unwrap();
        started.wait().unwrap();
        // Held while starting, so the start took no time; running afterwards.
        assert_eq!(*seen.lock().unwrap(), Some((true, false)));
        assert!(!timer.is_paused());
        assert_eq!(timer.time_ns(), t0);
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        rt.shutdown().wait().unwrap();
        // The first Δt is only the time after the start.
        assert_eq!(*dts.lock().unwrap(), [H_NS as f64 / 1e9]);
    });
}

#[test]
fn returns_while_a_node_is_still_updating_and_join_sees_the_end() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes {
            log,
            mut nodes,
            names,
        } = probes(1);
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        nodes[0].gate = Some((entered_tx, release_rx));
        let config = RuntimeConfig::default();
        let (rt, started) = mode
            .start(nodes, names, Arc::clone(&timer), config)
            .unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        entered.recv_timeout(LONG).unwrap();

        // The node is blocked in its update, so a waiting stop would not return.
        rt.request_stop();
        assert!(!rt.join(Some(QUIET)).unwrap());
        assert!(!events(&log, 0).contains(&Event::Stop));

        release.send(()).unwrap();
        assert!(rt.join(Some(LONG)).unwrap());
        assert!(events(&log, 0).contains(&Event::Stop));
        assert!(rt.is_finished());
    });
}

#[test]
fn does_nothing_once_the_system_has_ended() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, nodes, names } = probes(2);
        let (rt, started) = mode
            .start(nodes, names, Arc::clone(&timer), manual())
            .unwrap();
        started.wait().unwrap();
        rt.request_stop();
        assert!(rt.join(Some(Duration::from_secs(5))).unwrap());
        rt.request_stop();
        rt.shutdown().wait().unwrap();
        for id in 0..2 {
            assert_eq!(events(&log, id), [Event::Start, Event::Stop]);
        }
    });
}

#[test]
fn each_runtime_measures_dt_on_its_own_timer() {
    each_mode(|mode| {
        let (ta, tb) = (virtual_timer(), virtual_timer());
        let Probes {
            log: la,
            nodes,
            names,
        } = probes(1);
        let (a, started) = mode.start(nodes, names, Arc::clone(&ta), manual()).unwrap();
        started.wait().unwrap();
        let Probes {
            log: lb,
            nodes,
            names,
        } = probes(1);
        let (b, started) = mode.start(nodes, names, Arc::clone(&tb), manual()).unwrap();
        started.wait().unwrap();

        ta.advance_ns(2 * H_NS).unwrap();
        tb.advance_ns(5 * H_NS).unwrap();
        a.step(None).wait().unwrap();
        b.step(None).wait().unwrap();
        a.shutdown().wait().unwrap();
        b.shutdown().wait().unwrap();
        assert_eq!(events(&la, 0)[1], Event::Update(2.0 * H_NS as f64 / 1e9));
        assert_eq!(events(&lb, 0)[1], Event::Update(5.0 * H_NS as f64 / 1e9));
    });
}

#[test]
fn dedicated_and_pool_nodes_go_through_the_same_lifecycle() {
    let timer = virtual_timer();
    let Probes {
        log, nodes, names, ..
    } = probes(2);
    let config = RuntimeConfig {
        dedicated: Some(vec![true, false]),
        ..manual()
    };
    let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
    started.wait().unwrap();
    timer.advance_ns(H_NS).unwrap();
    rt.step(None).wait().unwrap();
    let states = rt.capture(|states| states, PAUSE).wait().unwrap();
    assert_eq!(states, [1, 1]);
    timer.advance_ns(H_NS).unwrap();
    rt.step(None).wait().unwrap();
    rt.shutdown().wait().unwrap();
    let dt = Event::Update(H_NS as f64 / 1e9);
    for id in 0..2 {
        let expected = [
            Event::Start,
            dt.clone(),
            Event::Pause,
            Event::StateDict,
            Event::Resume,
            dt.clone(),
            Event::Stop,
        ];
        assert_eq!(events(&log, id), expected);
    }
}
