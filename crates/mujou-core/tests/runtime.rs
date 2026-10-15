use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mujou_core::runtime::{
    HookError, NodeFailure, NodeHooks, Runtime, RuntimeConfig, RuntimeError,
};
use mujou_core::time::GlobalTimer;

const H_NS: u64 = 1_000_000;

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Start,
    Update(f64),
    Pause,
    Resume,
    Exception,
    Stop,
    StateDict,
}

#[derive(Debug)]
struct Boom;

impl fmt::Display for Boom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("boom")
    }
}

impl std::error::Error for Boom {}

/// Records its hook calls; fails (or panics) on the `n`-th update if asked.
struct Recorder {
    id: usize,
    log: Arc<Mutex<Vec<(usize, Event)>>>,
    updates: usize,
    fail_on_update: Option<usize>,
    panic_on_update: Option<usize>,
}

impl Recorder {
    fn record(&self, event: Event) {
        self.log.lock().unwrap().push((self.id, event));
    }
}

impl NodeHooks for Recorder {
    type State = usize;

    fn start(&mut self) -> Result<(), HookError> {
        self.record(Event::Start);
        Ok(())
    }

    fn update(&mut self, dt: f64) -> Result<(), HookError> {
        self.record(Event::Update(dt));
        self.updates += 1;
        if self.panic_on_update == Some(self.updates) {
            panic!("panic in node {}", self.id);
        }
        if self.fail_on_update == Some(self.updates) {
            return Err(HookError {
                hook: "on_update",
                source: Arc::new(Boom),
            });
        }
        Ok(())
    }

    fn pause(&mut self) -> Result<(), HookError> {
        self.record(Event::Pause);
        Ok(())
    }

    fn resume(&mut self) -> Result<(), HookError> {
        self.record(Event::Resume);
        Ok(())
    }

    fn exception(&mut self) -> Result<(), HookError> {
        self.record(Event::Exception);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), HookError> {
        self.record(Event::Stop);
        Ok(())
    }

    fn state_dict(&mut self) -> Result<usize, HookError> {
        self.record(Event::StateDict);
        Ok(self.updates)
    }
}

/// A virtual timer of its own: `cargo test` runs the tests in parallel threads.
fn virtual_timer() -> &'static GlobalTimer {
    let timer = Box::leak(Box::new(GlobalTimer::new()));
    timer.use_virtual_clock(true);
    timer
}

struct Setup {
    log: Arc<Mutex<Vec<(usize, Event)>>>,
    nodes: Vec<Recorder>,
    names: Vec<String>,
}

fn recorders(n: usize) -> Setup {
    let log = Arc::new(Mutex::new(Vec::new()));
    let nodes = (0..n)
        .map(|id| Recorder {
            id,
            log: Arc::clone(&log),
            updates: 0,
            fail_on_update: None,
            panic_on_update: None,
        })
        .collect();
    let names = (0..n).map(|i| format!("n{i}")).collect();
    Setup { log, nodes, names }
}

fn manual() -> RuntimeConfig {
    RuntimeConfig {
        manual: true,
        ..RuntimeConfig::default()
    }
}

fn events_of(log: &Mutex<Vec<(usize, Event)>>, id: usize) -> Vec<Event> {
    let log = log.lock().unwrap();
    log.iter()
        .filter(|(i, _)| *i == id)
        .map(|(_, e)| e.clone())
        .collect()
}

mod manual_mode {
    use super::*;

    #[test]
    fn steps_in_the_given_order_with_the_advanced_dt() {
        let timer = virtual_timer();
        let Setup { log, nodes, names } = recorders(3);
        let t0 = timer.time_ns();
        let (rt, started) = Runtime::start(nodes, names, timer, manual(), None).unwrap();
        started.wait().unwrap();
        for _ in 0..2 {
            timer.advance_ns(H_NS).unwrap();
            rt.step(Some(&[2, 0, 1])).wait().unwrap();
        }
        rt.shutdown().wait().unwrap();

        let order: Vec<usize> = log.lock().unwrap().iter().map(|(i, _)| *i).collect();
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
            assert_eq!(events_of(&log, id), expected);
        }
        assert_eq!(rt.last_update_ns(), vec![t0 + 2 * H_NS; 3]);
    }

    #[test]
    fn a_step_without_time_passing_does_not_update() {
        let timer = virtual_timer();
        let Setup { log, nodes, names } = recorders(1);
        let (rt, started) = Runtime::start(nodes, names, timer, manual(), None).unwrap();
        started.wait().unwrap();
        rt.step(None).wait().unwrap();
        rt.shutdown().wait().unwrap();
        assert_eq!(events_of(&log, 0), [Event::Start, Event::Stop]);
    }

    #[test]
    fn step_needs_the_manual_mode() {
        let timer = virtual_timer();
        let Setup { nodes, names, .. } = recorders(1);
        let config = RuntimeConfig::default();
        let (rt, started) = Runtime::start(nodes, names, timer, config, None).unwrap();
        started.wait().unwrap();
        assert!(matches!(rt.step(None).wait(), Err(RuntimeError::NotManual)));
        rt.shutdown().wait().unwrap();
    }
}

mod capture {
    use super::*;

    #[test]
    fn builds_from_every_state_while_the_time_is_frozen() {
        let timer = virtual_timer();
        let Setup { log, nodes, names } = recorders(2);
        let (rt, started) = Runtime::start(nodes, names, timer, manual(), None).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        let (states, paused) = rt
            .capture(None, move |states| (states, timer.is_paused()))
            .wait()
            .unwrap();
        assert_eq!(states, vec![1, 1]);
        assert!(paused);
        assert!(!timer.is_paused());
        rt.shutdown().wait().unwrap();
        for id in 0..2 {
            let events = events_of(&log, id);
            assert_eq!(
                &events[2..5],
                &[Event::Pause, Event::StateDict, Event::Resume]
            );
        }
    }
}

mod failure {
    use super::*;

    fn assert_exception_then_stop_once(events: &[Event]) {
        let exception = events.iter().position(|e| *e == Event::Exception);
        let stop = events.iter().position(|e| *e == Event::Stop);
        assert_eq!(events.iter().filter(|e| **e == Event::Exception).count(), 1);
        assert_eq!(events.iter().filter(|e| **e == Event::Stop).count(), 1);
        let (exception, stop) = (exception.unwrap(), stop.unwrap());
        assert!(exception < stop);
        // Nothing but `stop` after `exception`.
        assert_eq!(events.len(), stop + 1);
    }

    #[test]
    fn an_update_error_ends_every_node_with_exception_then_stop() {
        let timer = virtual_timer();
        let Setup {
            log,
            mut nodes,
            names,
        } = recorders(3);
        nodes[1].fail_on_update = Some(3);
        let (rt, started) = Runtime::start(nodes, names, timer, manual(), None).unwrap();
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
        assert_eq!((failure.node, failure.hook), (1, "on_update"));
        assert!(rt.is_finished());
        for id in 0..3 {
            assert_exception_then_stop_once(&events_of(&log, id));
        }
        // Later requests report the same failure.
        assert!(matches!(rt.shutdown().wait(), Err(RuntimeError::Failed(_))));
        assert!(matches!(rt.join(None), Err(RuntimeError::Failed(_))));
    }

    #[test]
    fn a_panic_is_a_failure() {
        let timer = virtual_timer();
        let Setup {
            log,
            mut nodes,
            names,
        } = recorders(2);
        nodes[0].panic_on_update = Some(1);
        let (rt, started) = Runtime::start(nodes, names, timer, manual(), None).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        let result = rt.step(None).wait();
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!((failure.node, failure.hook), (0, "on_update"));
        assert!(failure.error.to_string().contains("panic in node 0"));
        for id in 0..2 {
            assert_exception_then_stop_once(&events_of(&log, id));
        }
    }

    #[test]
    fn captures_the_state_on_the_frozen_timer_and_resumes_it_at_the_end() {
        let timer = virtual_timer();
        let Setup {
            mut nodes, names, ..
        } = recorders(2);
        nodes[0].fail_on_update = Some(2);
        type Captured = (BTreeMap<usize, usize>, usize, bool);
        let captured: Arc<Mutex<Option<Captured>>> = Arc::default();
        let capture = {
            let captured = Arc::clone(&captured);
            Box::new(move |states, failure: &NodeFailure| {
                *captured.lock().unwrap() = Some((states, failure.node, timer.is_paused()));
            })
        };
        let (rt, started) = Runtime::start(nodes, names, timer, manual(), Some(capture)).unwrap();
        started.wait().unwrap();
        for _ in 0..2 {
            timer.advance_ns(H_NS).unwrap();
            let _ = rt.step(None).wait();
        }
        assert!(rt.join(Some(Duration::from_secs(5))).is_err());
        let (states, node, paused) = captured.lock().unwrap().take().unwrap();
        assert_eq!(states, BTreeMap::from([(0, 2), (1, 1)]));
        assert_eq!(node, 0);
        assert!(paused);
        assert!(!timer.is_paused());
    }
}
