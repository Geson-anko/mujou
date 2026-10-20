use mujou_core::time::TimerState;

use super::*;

fn restoring() -> RuntimeConfig {
    RuntimeConfig {
        timer_state: Some(TimerState {
            time_ns: 7 * H_NS,
            scale: 1.0,
        }),
        ..manual()
    }
}

#[test]
fn the_time_is_not_restored_under_a_running_runtime() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let Probes { nodes, names, .. } = probes(1);
        let (running, started) = mode
            .start(nodes, names, Arc::clone(&timer), manual())
            .unwrap();
        started.wait().unwrap();
        let before = timer.time_ns();

        let Probes { log, nodes, names } = probes(1);
        let result = mode.start(nodes, names, Arc::clone(&timer), restoring());
        assert!(matches!(result, Err(RuntimeError::InvalidArgument(_))));
        // Nothing started, and the running runtime's time did not move.
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(timer.time_ns(), before);
        assert!(!timer.is_paused());

        // Once it has finished, the time can be restored.
        running.shutdown().wait().unwrap();
        let Probes { nodes, names, .. } = probes(1);
        let (restored, started) = mode
            .start(nodes, names, Arc::clone(&timer), restoring())
            .unwrap();
        started.wait().unwrap();
        assert_eq!(timer.time_ns(), 7 * H_NS);
        restored.shutdown().wait().unwrap();
    });
}

#[test]
fn a_runtime_on_another_timer_does_not_block_a_restore() {
    each_mode(|mode| {
        let (ta, tb) = (virtual_timer(), virtual_timer());
        let Probes { nodes, names, .. } = probes(1);
        let (running, started) = mode.start(nodes, names, Arc::clone(&ta), manual()).unwrap();
        started.wait().unwrap();
        let before = ta.time_ns();

        let config = RuntimeConfig {
            timer_state: Some(TimerState {
                time_ns: 7 * H_NS,
                scale: 1.0,
            }),
            ..manual()
        };
        let Probes { nodes, names, .. } = probes(1);
        let (restored, started) = mode.start(nodes, names, Arc::clone(&tb), config).unwrap();
        started.wait().unwrap();
        assert_eq!(tb.time_ns(), 7 * H_NS);
        assert_eq!(ta.time_ns(), before);
        running.shutdown().wait().unwrap();
        restored.shutdown().wait().unwrap();
    });
}
