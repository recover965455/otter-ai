//! Resolve the **otter-ai** on-disk configuration directory.
//!
//! The TypeScript ancestor `@earendil-works/pi-ai` stores credentials, model
//! catalogues, user-declared `models.json`, extension state etc. under
//! `~/.pi/agent/`.  For `otter-ai` we deliberately switch to `~/.otter` so
//! the two SDKs can coexist on the same machine without clobbering each
//! other's files.
//!
//! Resolution order (matches `dirs-next` / XDG conventions on every platform,
//! same semantics pi uses — just with a different leaf name):
//!
//! 1. If `OTTER_CONFIG_DIR` is set in the environment → use it verbatim.
//! 2. If `XDG_CONFIG_HOME` is set (Unix convention) → `$XDG_CONFIG_HOME/otter`.
//! 3. Otherwise → the user's home directory joined with `.otter`
//!    (`~/.otter` on Unix; `%USERPROFILE%\.otter` on Windows).
//!
//! Callers may want to append an **agent id** or **app subdir** below this
//! root (e.g. `config_dir()?.join("agent")`) so multiple binaries built on
//! top of otter-ai can still share the provider catalogue while keeping
//! their own state isolated.

use std::path::{Path, PathBuf};

/// Name of the environment variable that overrides everything else.
pub const OTTER_CONFIG_DIR_ENV: &str = "OTTER_CONFIG_DIR";

/// Leaf directory name used when falling back to `$HOME/.<APP>`.
pub const OTTER_HOME_DIRNAME: &str = ".otter";

/// XDG subdir (`$XDG_CONFIG_HOME/<this>`).
pub const OTTER_XDG_DIRNAME: &str = "otter";

/// Resolve the on-disk configuration directory for otter-ai.
///
/// * Ok(Some(path)) → resolved successfully; callers can read/write there.
/// * Ok(None) → no reasonable home directory could be located (e.g.
///   containerised environments without a `$HOME`).  Callers should fall
///   back to in-memory stores or surface a warning.
/// * Err → environment variable / path arithmetic went sideways (non-UTF8
///   bytes in path etc.).
pub fn config_dir() -> anyhow::Result<Option<PathBuf>> {
    // 1) explicit OTTER_CONFIG_DIR override wins.
    if let Ok(val) = std::env::var(OTTER_CONFIG_DIR_ENV) {
        let p = PathBuf::from(val);
        return Ok(Some(p));
    }

    // 2) XDG_CONFIG_HOME/otter — Unix convention; still respected on other
    //    platforms if the user explicitly set it, since we don't gate it by
    //    target family.
    if let Ok(val) = std::env::var("XDG_CONFIG_HOME") {
        if !val.is_empty() {
            let p = PathBuf::from(val).join(OTTER_XDG_DIRNAME);
            return Ok(Some(p));
        }
    }

    // 3) $HOME/.otter — portable fallback that works for 99 % of end users.
    if let Some(home) = home_dir() {
        Ok(Some(home.join(OTTER_HOME_DIRNAME)))
    } else {
        Ok(None)
    }
}

/// Convenience: join `relative` onto [`config_dir`] in one go.
///
/// Returns `Err` when [`config_dir`] itself fails, `Ok(None)` when
/// [`config_dir`] returns `None` (no home directory), otherwise the joined
/// path.
pub fn config_path<P: AsRef<Path>>(relative: P) -> anyhow::Result<Option<PathBuf>> {
    Ok(config_dir()?.map(|root| root.join(relative)))
}

/// Ensure the config directory (or a sub-path inside it) exists on disk.
///
/// This is a thin wrapper around [`std::fs::create_dir_all`] that returns
/// the final canonicalised directory (or `None` when we couldn't even find
/// a home directory).  Callers typically call this once at startup before
/// reading `auth.json` / `models-store.json` / `models.json`.
pub fn ensure_config_dir<P: AsRef<Path>>(subdir: P) -> anyhow::Result<Option<PathBuf>> {
    match config_path(subdir)? {
        Some(p) => {
            std::fs::create_dir_all(&p)?;
            Ok(Some(p))
        }
        None => Ok(None),
    }
}

// ---- home-dir resolution (no external dep, mirrors `dirs::home_dir` logic) ----

fn home_dir() -> Option<PathBuf> {
    // Unix: prefer $HOME (same as dirs / std env).
    #[cfg(not(windows))]
    {
        if let Ok(home) = std::env::var("HOME") {
            if !home.is_empty() {
                return Some(PathBuf::from(home));
            }
        }
        // Fallback: getpwuid_r — only on unix, no dep on `libc` crate, use a
        // tiny libc binding via extern block so we keep dependency count
        // untouched.
        home_from_passwd()
    }

    // Windows: USERPROFILE, then HOMEDRIVE+HOMEPATH, then HOME.
    #[cfg(windows)]
    {
        if let Ok(p) = std::env::var("USERPROFILE") {
            if !p.is_empty() {
                return Some(PathBuf::from(p));
            }
        }
        if let (Ok(drive), Ok(path)) = (std::env::var("HOMEDRIVE"), std::env::var("HOMEPATH")) {
            if !drive.is_empty() {
                return Some(PathBuf::from(format!("{}{}", drive, path)));
            }
        }
        if let Ok(home) = std::env::var("HOME") {
            if !home.is_empty() {
                return Some(PathBuf::from(home));
            }
        }
        None
    }
}

#[cfg(not(windows))]
fn home_from_passwd() -> Option<PathBuf> {
    // Minimum libc surface to call getpwuid_r(getuid(), ...) without taking
    // the `libc` crate as a dep.  Anything weird → just return None; the
    // caller has explicit env overrides available.
    #[repr(C)]
    struct Passwd {
        pw_name: *mut LibcChar,
        pw_passwd: *mut LibcChar,
        pw_uid: u32,
        pw_gid: u32,
        pw_change: isize,
        pw_class: *mut LibcChar,
        pw_gecos: *mut LibcChar,
        pw_dir: *mut LibcChar,
        pw_shell: *mut LibcChar,
        pw_expire: isize,
        pw_fields: i32,
    }
    type LibcChar = std::os::raw::c_char;
    type LibcUidT = u32;

    extern "C" {
        fn getuid() -> LibcUidT;
        fn getpwuid_r(
            uid: LibcUidT,
            pwd: *mut Passwd,
            buf: *mut LibcChar,
            buflen: usize,
            result: *mut *mut Passwd,
        ) -> i32;
    }

    // SAFETY: plain C POSIX calls.  All buffers are stack allocated; if
    // anything goes wrong (null pointers, empty string, ERANGE) we just
    // return None and callers fall back to explicit env vars.
    unsafe {
        let mut pwd = std::mem::MaybeUninit::<Passwd>::zeroed().assume_init();
        const BUFLEN: usize = 16384;
        let mut buf: [LibcChar; BUFLEN] = std::mem::MaybeUninit::zeroed().assume_init();
        let mut result: *mut Passwd = std::ptr::null_mut();
        let rc = getpwuid_r(getuid(), &mut pwd, buf.as_mut_ptr(), BUFLEN, &mut result);
        if rc != 0 || result.is_null() || (*result).pw_dir.is_null() {
            return None;
        }
        let cstr = std::ffi::CStr::from_ptr((*result).pw_dir);
        let bytes = cstr.to_bytes();
        if bytes.is_empty() {
            return None;
        }
        match std::str::from_utf8(bytes) {
            Ok(s) => Some(PathBuf::from(s.to_string())),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_override_is_respected() {
        let key = OTTER_CONFIG_DIR_ENV;
        // SAFETY: tests run in-process, override only within this block.
        std::env::set_var(key, "/tmp/otter-test-override");
        let got = config_dir().unwrap().unwrap();
        std::env::remove_var(key);
        assert_eq!(got, PathBuf::from("/tmp/otter-test-override"));
    }

    #[test]
    fn resolves_a_directory_when_home_exists() {
        // In CI / dev environments at least one of $HOME / $USERPROFILE is
        // always set, so config_dir() shouldn't be None.
        let got = config_dir().unwrap();
        assert!(got.is_some(), "expected to find a home dir, got None");
        let path = got.unwrap();
        assert!(
            path.ends_with(OTTER_HOME_DIRNAME) || path.ends_with(OTTER_XDG_DIRNAME),
            "expected config dir to end with .otter or /otter, got {}",
            path.display()
        );
    }

    #[test]
    fn config_path_appends_relative_segment() {
        let key = OTTER_CONFIG_DIR_ENV;
        std::env::set_var(key, "/tmp/otter-cfg");
        let p = config_path("agent/auth.json").unwrap().unwrap();
        std::env::remove_var(key);
        assert_eq!(p, PathBuf::from("/tmp/otter-cfg/agent/auth.json"));
    }
}
