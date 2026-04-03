use crate::{PinnedAllocFlags, PinnedHostBuffer, Result};
use std::collections::BTreeMap;
use std::sync::Mutex;

const SIZE_CLASS_BYTES: usize = 8 * 1024 * 1024;

struct PoolState {
    cached_bytes: usize,
    bins: BTreeMap<usize, Vec<PinnedHostBuffer<u8>>>,
}

impl Default for PoolState {
    fn default() -> Self {
        Self {
            cached_bytes: 0,
            bins: BTreeMap::new(),
        }
    }
}

pub struct PinnedPool {
    flags: PinnedAllocFlags,
    max_cached_bytes: usize,
    inner: Mutex<PoolState>,
}

impl PinnedPool {
    pub fn new(flags: PinnedAllocFlags, max_cached_bytes: usize) -> Self {
        Self {
            flags,
            max_cached_bytes,
            inner: Mutex::new(PoolState::default()),
        }
    }

    fn round_up(bytes: usize) -> usize {
        if bytes == 0 {
            SIZE_CLASS_BYTES
        } else {
            let classes = (bytes + SIZE_CLASS_BYTES - 1) / SIZE_CLASS_BYTES;
            classes * SIZE_CLASS_BYTES
        }
    }

    pub fn checkout(&self, min_bytes: usize) -> Result<PinnedHostBuffer<u8>> {
        let needed = Self::round_up(min_bytes);
        {
            let mut guard = self.inner.lock().expect("PinnedPool mutex poisoned");
            let candidate_cap = guard.bins.range(needed..).next().map(|(&cap, _)| cap);
            if let Some(cap) = candidate_cap {
                let mut should_remove = false;
                let popped = {
                    let list = guard
                        .bins
                        .get_mut(&cap)
                        .expect("capacity bucket must exist when selected");
                    let popped = list.pop();
                    if list.is_empty() {
                        should_remove = true;
                    }
                    popped
                };
                if let Some(mut buf) = popped {
                    guard.cached_bytes = guard.cached_bytes.saturating_sub(cap);
                    if should_remove {
                        guard.bins.remove(&cap);
                    }
                    unsafe {
                        buf.set_len(0);
                    }
                    return Ok(buf);
                }
            }
        }
        PinnedHostBuffer::with_capacity_elems(needed, self.flags)
    }

    pub fn checkin(&self, mut buf: PinnedHostBuffer<u8>) {
        if self.max_cached_bytes == 0 {
            return;
        }
        let capacity = buf.capacity_bytes();
        let mut guard = self.inner.lock().expect("PinnedPool mutex poisoned");
        if guard.cached_bytes + capacity > self.max_cached_bytes {
            return;
        }
        unsafe {
            buf.set_len(0);
        }
        guard.cached_bytes += capacity;
        guard.bins.entry(capacity).or_default().push(buf);
    }

    pub fn stats(&self) -> (usize, usize) {
        let guard = self.inner.lock().expect("PinnedPool mutex poisoned");
        let buffers = guard.bins.values().map(|v| v.len()).sum();
        (guard.cached_bytes, buffers)
    }
}
