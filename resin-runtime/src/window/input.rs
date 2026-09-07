//! Per-window button snapshots. Callbacks accumulate transitions until that
//! window polls, including taps that begin and end within one poll.
use glfw_sys as sys;

pub(super) const DOWN: u32 = 1;
const PRESSED: u32 = 2;
const RELEASED: u32 = 4;

#[derive(Clone)]
struct Snapshot {
    keys: [u32; sys::GLFW_KEY_LAST as usize + 1],
    buttons: [u32; sys::GLFW_MOUSE_BUTTON_LAST as usize + 1],
    scroll: (f64, f64),
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            keys: [0; sys::GLFW_KEY_LAST as usize + 1],
            buttons: [0; sys::GLFW_MOUSE_BUTTON_LAST as usize + 1],
            scroll: (0.0, 0.0),
        }
    }
}

#[derive(Default)]
pub(super) struct Input {
    pending: Snapshot,
    current: Snapshot,
}

impl Input {
    pub fn key(&mut self, key: i32, action: i32) {
        if let Ok(index) = usize::try_from(key)
            && let Some(state) = self.pending.keys.get_mut(index)
        {
            transition(state, action);
        }
    }

    pub fn mouse_button(&mut self, button: i32, action: i32) {
        if let Ok(index) = usize::try_from(button)
            && let Some(state) = self.pending.buttons.get_mut(index)
        {
            transition(state, action);
        }
    }

    pub fn scroll(&mut self, x: f64, y: f64) {
        self.pending.scroll.0 += x;
        self.pending.scroll.1 += y;
    }

    pub fn commit(&mut self) {
        self.current.clone_from(&self.pending);
        for state in self
            .pending
            .keys
            .iter_mut()
            .chain(self.pending.buttons.iter_mut())
        {
            *state &= DOWN;
        }
        self.pending.scroll = (0.0, 0.0);
    }

    pub fn key_state(&self, key: i32) -> u32 {
        usize::try_from(key)
            .ok()
            .and_then(|i| self.current.keys.get(i))
            .copied()
            .unwrap_or(0)
    }

    pub fn mouse_button_state(&self, button: i32) -> u32 {
        usize::try_from(button)
            .ok()
            .and_then(|i| self.current.buttons.get(i))
            .copied()
            .unwrap_or(0)
    }

    pub fn scroll_delta(&self) -> (f64, f64) {
        self.current.scroll
    }
}

fn transition(state: &mut u32, action: i32) {
    match action {
        sys::GLFW_PRESS => {
            *state |= DOWN | PRESSED;
        }
        sys::GLFW_RELEASE => {
            *state = (*state & !DOWN) | RELEASED;
        }
        _ => {} // Repeats do not create another physical press edge.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_taps_repeats_and_holds_survive_poll_boundaries() {
        let mut input = Input::default();
        input.key(sys::GLFW_KEY_A, sys::GLFW_PRESS);
        input.key(sys::GLFW_KEY_A, sys::GLFW_RELEASE);
        input.mouse_button(sys::GLFW_MOUSE_BUTTON_LEFT, sys::GLFW_PRESS);
        assert_eq!(input.key_state(sys::GLFW_KEY_A), 0);
        input.commit();
        assert_eq!(input.key_state(sys::GLFW_KEY_A), PRESSED | RELEASED);
        assert_eq!(
            input.mouse_button_state(sys::GLFW_MOUSE_BUTTON_LEFT),
            DOWN | PRESSED
        );
        input.mouse_button(sys::GLFW_MOUSE_BUTTON_LEFT, sys::GLFW_REPEAT);
        input.commit();
        assert_eq!(input.key_state(sys::GLFW_KEY_A), 0);
        assert_eq!(input.mouse_button_state(sys::GLFW_MOUSE_BUTTON_LEFT), DOWN);
        input.mouse_button(sys::GLFW_MOUSE_BUTTON_LEFT, sys::GLFW_RELEASE);
        input.commit();
        assert_eq!(
            input.mouse_button_state(sys::GLFW_MOUSE_BUTTON_LEFT),
            RELEASED
        );
    }

    #[test]
    fn scroll_accumulates_and_queries_do_not_consume_the_snapshot() {
        let mut input = Input::default();
        input.scroll(0.5, 2.0);
        input.scroll(-1.0, -0.5);
        input.commit();
        assert_eq!(input.scroll_delta(), (-0.5, 1.5));
        assert_eq!(input.scroll_delta(), (-0.5, 1.5));
        input.commit();
        assert_eq!(input.scroll_delta(), (0.0, 0.0));
        input.key(-1, sys::GLFW_PRESS);
        input.mouse_button(8, sys::GLFW_PRESS);
        input.commit();
        assert_eq!(input.key_state(-1), 0);
        assert_eq!(input.key_state(i32::MAX), 0);
        assert_eq!(input.mouse_button_state(8), 0);
    }
}
