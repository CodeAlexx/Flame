#![cfg(feature = "cuda")]

use anyhow::Result;

#[test]
fn pinned_alloc_and_async_copy() -> Result<()> {
    // Hard-fail when CUDA isn't available so the suite doesn't silently skip GPU coverage.
    let dev = cudarc::driver::CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    );

    let stream = dev
        .fork_default_stream()
        .map_err(|e| anyhow::anyhow!("fork_default_stream: {:?}", e))?;

    // Use the public API you added:
    use flame_core::{PinnedAllocFlags, PinnedHostBuffer, StagingDeviceBuf};

    let n = 1usize << 20; // 1M u32
    let mut host =
        PinnedHostBuffer::<u32>::with_capacity_elems(n, PinnedAllocFlags::WRITE_COMBINED)?;
    unsafe {
        let slice = std::slice::from_raw_parts_mut(host.as_mut_ptr(), n);
        for (i, x) in slice.iter_mut().enumerate() {
            *x = i as u32;
        }

        // Track logical length so async copy covers initialized region.
        host.set_len(n);
    }

    let devbuf = StagingDeviceBuf::<u32>::new(dev.clone(), n)?;
    devbuf.async_upload_from(&host, &stream)?;
    dev.synchronize()
        .map_err(|e| anyhow::anyhow!("synchronize: {:?}", e))?;
    Ok(())
}
