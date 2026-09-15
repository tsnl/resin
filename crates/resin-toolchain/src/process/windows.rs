//! Windows jobs own the complete native process tree. The child starts suspended,
//! joins a kill-on-close job, and only then resumes, preventing descendant escape.
use std::{
    io, mem,
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle},
    ptr,
    time::Duration,
};
use tokio::process::{Child, Command};
use windows_sys::Win32::{
    Foundation::{HANDLE, INVALID_HANDLE_VALUE},
    System::{
        Diagnostics::ToolHelp::{
            CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
        },
        JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
            QueryInformationJobObject, SetInformationJobObject, TerminateJobObject,
        },
        Threading::{CREATE_SUSPENDED, OpenThread, ResumeThread, THREAD_SUSPEND_RESUME},
    },
};

pub(super) fn prepare(command: &mut Command) {
    command.creation_flags(CREATE_SUSPENDED);
}

pub(super) struct Tree {
    job: OwnedHandle,
}

impl Tree {
    pub(super) fn attach(child: &Child) -> io::Result<Self> {
        let handle = unsafe { CreateJobObjectW(ptr::null(), ptr::null()) };
        if handle.is_null() {
            return Err(io::Error::last_os_error());
        }
        let tree = Self {
            job: unsafe { OwnedHandle::from_raw_handle(handle) },
        };
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let process = child
            .raw_handle()
            .ok_or_else(|| io::Error::other("native process exited before supervision"))?;
        if unsafe {
            SetInformationJobObject(
                tree.handle(),
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const _,
                mem::size_of_val(&limits) as u32,
            )
        } == 0
            || unsafe { AssignProcessToJobObject(tree.handle(), process) } == 0
        {
            return Err(io::Error::last_os_error());
        }
        resume(child.id().expect("suspended process has an ID"))?;
        Ok(tree)
    }

    fn handle(&self) -> HANDLE {
        self.job.as_raw_handle()
    }

    pub(super) async fn wait_exit(&self, child: &mut Child) -> io::Result<()> {
        child.wait().await.map(|_| ())
    }

    pub(super) fn interrupt(&self) {
        unsafe {
            TerminateJobObject(self.handle(), 1);
        }
    }

    pub(super) async fn terminate(&self) {
        self.interrupt();
    }

    pub(super) fn terminate_sync(&mut self) {
        self.interrupt();
        while self.active() {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    pub(super) async fn finished(&mut self) {
        while self.active() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn active(&self) -> bool {
        let mut state: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = unsafe { mem::zeroed() };
        let queried = unsafe {
            QueryInformationJobObject(
                self.handle(),
                JobObjectBasicAccountingInformation,
                &mut state as *mut _ as *mut _,
                mem::size_of_val(&state) as u32,
                ptr::null_mut(),
            )
        };
        queried != 0 && state.ActiveProcesses != 0
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        self.terminate_sync();
    }
}

fn resume(process: u32) -> io::Result<()> {
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    let snapshot = unsafe { OwnedHandle::from_raw_handle(snapshot) };
    let mut entry: THREADENTRY32 = unsafe { mem::zeroed() };
    entry.dwSize = mem::size_of_val(&entry) as u32;
    let mut found = unsafe { Thread32First(snapshot.as_raw_handle(), &mut entry) };
    while found != 0 {
        if entry.th32OwnerProcessID == process {
            let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) };
            if thread.is_null() {
                return Err(io::Error::last_os_error());
            }
            let thread = unsafe { OwnedHandle::from_raw_handle(thread) };
            if unsafe { ResumeThread(thread.as_raw_handle()) } == u32::MAX {
                return Err(io::Error::last_os_error());
            }
            return Ok(());
        }
        found = unsafe { Thread32Next(snapshot.as_raw_handle(), &mut entry) };
    }
    Err(io::Error::other(
        "suspended native process has no primary thread",
    ))
}
