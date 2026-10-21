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

    fn capture(&mut self) -> Result<(), HookError> {
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
        let (rt, started) = mode.start(manual(vec![node]), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        // Held while starting, so the start took no time; running afterwards.
        assert_eq!(*seen.lock().unwrap(), Some((true, false)));
        assert!(!timer.is_paused());
        assert_eq!(timer.time_ns(), t0);
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        shut_down(&rt).unwrap();
        // The first Δt is only the time after the start.
        assert_eq!(*dts.lock().unwrap(), [H_NS as f64 / 1e9]);
    });
}

#[test]
fn returns_while_a_node_is_still_updating_and_join_sees_the_end() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, mut nodes } = probes(1);
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        nodes[0].gate = Some((entered_tx, release_rx));
        let (rt, started) = mode.start(config(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        entered.recv_timeout(LONG).unwrap();

        // The node is blocked in its update, so a waiting stop would not return.
        rt.stop();
        assert!(!rt.join(Some(QUIET)).unwrap());
        assert!(!events(&log, 0).contains(&Event::Stop));

        release.send(()).unwrap();
        assert!(rt.join(Some(LONG)).unwrap());
        assert!(events(&log, 0).contains(&Event::Stop));
        assert!(rt.finished());
    });
}

#[test]
fn does_nothing_once_the_system_has_ended() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { log, nodes } = probes(2);
        let (rt, started) = mode.start(manual(nodes), Arc::clone(&timer)).unwrap();
        started.wait().unwrap();
        rt.stop();
        assert!(rt.join(Some(Duration::from_secs(5))).unwrap());
        rt.stop();
        shut_down(&rt).unwrap();
        for id in 0..2 {
            assert_eq!(events(&log, id), [Event::Start, Event::Stop]);
        }
    });
}

#[test]
fn each_runtime_measures_dt_on_its_own_timer() {
    each_mode(|mode| {
        let (ta, tb) = (virtual_timer(), virtual_timer());
        let Probes { log: la, nodes } = probes(1);
        let (a, started) = mode.start(manual(nodes), Arc::clone(&ta)).unwrap();
        started.wait().unwrap();
        let Probes { log: lb, nodes } = probes(1);
        let (b, started) = mode.start(manual(nodes), Arc::clone(&tb)).unwrap();
        started.wait().unwrap();

        ta.advance_ns(2 * H_NS).unwrap();
        tb.advance_ns(5 * H_NS).unwrap();
        a.step(None).wait().unwrap();
        b.step(None).wait().unwrap();
        shut_down(&a).unwrap();
        shut_down(&b).unwrap();
        assert_eq!(events(&la, 0)[1], Event::Update(2.0 * H_NS as f64 / 1e9));
        assert_eq!(events(&lb, 0)[1], Event::Update(5.0 * H_NS as f64 / 1e9));
    });
}

#[test]
fn dedicated_and_pool_nodes_go_through_the_same_lifecycle() {
    let timer = virtual_timer();
    let Probes { log, nodes, .. } = probes(2);
    let mut config = manual(nodes);
    config.nodes[0].dedicated_thread = true;
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    timer.advance_ns(H_NS).unwrap();
    rt.step(None).wait().unwrap();
    let states = rt.capture(|states| states, PAUSE).wait().unwrap();
    assert_eq!(states, [1, 1]);
    timer.advance_ns(H_NS).unwrap();
    rt.step(None).wait().unwrap();
    shut_down(&rt).unwrap();
    let dt = Event::Update(H_NS as f64 / 1e9);
    for id in 0..2 {
        let expected = [
            Event::Start,
            dt.clone(),
            Event::Pause,
            Event::Capture,
            Event::Resume,
            dt.clone(),
            Event::Stop,
        ];
        assert_eq!(events(&log, id), expected);
    }
}

#[test]
fn the_clock_stops_at_the_stop_request() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let run_clock = clock(&timer);
        let Probes { nodes, .. } = probes(2);
        let (rt, started) =
            Runtime::start(mode_config(mode, manual(nodes)), Arc::clone(&run_clock)).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        let stop_ns = timer.time_ns();
        shut_down(&rt).unwrap();
        timer.advance_ns(H_NS).unwrap();
        // The time of the run stays at the request, after every update.
        assert_eq!(run_clock.now_ns(), stop_ns);
        assert!(rt.last_update_ns().iter().all(|&ns| ns <= stop_ns));
    });
}

#[test]
fn the_clock_stops_when_a_failure_is_caught() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let run_clock = clock(&timer);
        let Probes { mut nodes, .. } = probes(1);
        // The update moves the time by H_NS and then fails.
        nodes[0].advance = Some((Arc::clone(&timer), 1));
        nodes[0].fail_on_update = Some((1, Failure::Error));
        let (rt, started) =
            Runtime::start(mode_config(mode, manual(nodes)), Arc::clone(&run_clock)).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        assert!(matches!(rt.step(None).wait(), Err(RuntimeError::Failed(_))));
        timer.advance_ns(H_NS).unwrap();
        assert_eq!(run_clock.now_ns(), 2 * H_NS);
    });
}
