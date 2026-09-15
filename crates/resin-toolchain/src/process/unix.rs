//! A private session contains Ninja's separately grouped jobs as well as their children.
use std::{
    io,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::process::{Child, Command};

pub(super) fn prepare(command: &mut Command) {
    // Only async-signal-safe operations run between fork and exec.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

pub(super) struct Tree {
    // Workers retain this identity gate. Shutdown drains any in-flight session scan
    // before clearing it and reaping the leader, so delayed workers cannot signal a
    // subsequently reused PID.
    session: Arc<Mutex<libc::pid_t>>,
}

impl Tree {
    pub(super) fn attach(child: &Child) -> io::Result<Self> {
        let session = child
            .id()
            .ok_or_else(|| io::Error::other("native process exited before supervision"))?;
        Ok(Self {
            session: Arc::new(Mutex::new(session as libc::pid_t)),
        })
    }

    pub(super) async fn wait_exit(&self, _child: &mut Child) -> io::Result<()> {
        let session = *self.session.lock().expect("native session gate");
        while !exited(session)? {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Ok(())
    }

    pub(super) fn interrupt(&self) {
        // Let Ninja forward interruption to its jobs and reap them before forcing exit.
        let session = self.session.lock().expect("native session gate");
        if *session != 0 {
            unsafe {
                libc::kill(-*session, libc::SIGTERM);
            }
        }
    }

    pub(super) async fn terminate(&self) {
        let session = self.session.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let session = session.lock().expect("native session gate");
            if *session != 0 {
                kill_session(*session);
            }
        })
        .await;
    }

    pub(super) async fn finished(&mut self) {
        loop {
            let session = self.session.clone();
            let finished = tokio::task::spawn_blocking(move || {
                let mut session = session.lock().expect("native session gate");
                if *session == 0 || session_processes(*session).is_empty() {
                    *session = 0;
                    true
                } else {
                    false
                }
            })
            .await
            .unwrap_or(false);
            if finished {
                return;
            }
            self.terminate().await;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    pub(super) fn terminate_sync(&mut self) {
        let mut session = self.session.lock().expect("native session gate");
        if *session != 0 {
            loop {
                kill_session(*session);
                if session_processes(*session).is_empty() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            *session = 0;
        }
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        self.terminate_sync();
    }
}

fn exited(process: libc::pid_t) -> io::Result<bool> {
    let mut status: libc::siginfo_t = unsafe { std::mem::zeroed() };
    let result = unsafe {
        libc::waitid(
            libc::P_PID,
            process as libc::id_t,
            &mut status,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if result >= 0 {
        return Ok(unsafe { status.si_pid() } != 0);
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::Interrupted {
        Ok(false)
    } else {
        Err(error)
    }
}

fn kill_session(session: libc::pid_t) {
    // Signal individual session members too: Ninja puts each compiler in a new group.
    // Descendants retain this session even when their immediate parent exits.
    for process in session_processes(session) {
        unsafe {
            libc::kill(process, libc::SIGKILL);
        }
    }
    unsafe {
        libc::kill(-session, libc::SIGKILL);
    }
}

#[cfg(target_os = "linux")]
fn session_processes(session: libc::pid_t) -> Vec<libc::pid_t> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let process: libc::pid_t = entry.file_name().to_str()?.parse().ok()?;
            let text = std::fs::read_to_string(entry.path().join("stat")).ok()?;
            let fields = text
                .rsplit_once(')')?
                .1
                .split_whitespace()
                .collect::<Vec<_>>();
            // Dead grandchildren are reaped by their parent or init; they cannot write files.
            (fields.first().copied() != Some("Z")
                && fields.get(3)?.parse::<libc::pid_t>().ok()? == session)
                .then_some(process)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn session_processes(session: libc::pid_t) -> Vec<libc::pid_t> {
    let count = unsafe { libc::proc_listallpids(std::ptr::null_mut(), 0) };
    if count <= 0 {
        return vec![];
    }
    let mut processes = vec![0i32; count as usize + 64];
    let count = unsafe {
        libc::proc_listallpids(
            processes.as_mut_ptr().cast(),
            (processes.len() * std::mem::size_of::<libc::pid_t>()) as i32,
        )
    };
    processes.truncate(count.max(0) as usize);
    processes
        .into_iter()
        .filter(|&pid| {
            if unsafe { libc::getsid(pid) } != session {
                return false;
            }
            let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
            let result = unsafe {
                libc::proc_pidinfo(
                    pid,
                    libc::PROC_PIDTBSDINFO,
                    0,
                    (&mut info as *mut libc::proc_bsdinfo).cast(),
                    std::mem::size_of_val(&info) as i32,
                )
            };
            result > 0 && info.pbi_status != libc::SZOMB as u32
        })
        .collect()
}
