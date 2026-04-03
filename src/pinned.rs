use crate::{cuda::ffi, Error, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::{size_of, ManuallyDrop};
use std::ops::{BitOr, BitOrAssign};
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_HOST_ALLOC_DEFAULT: u32 = 0x00;
const CUDA_HOST_ALLOC_PORTABLE: u32 = 0x01;
const CUDA_HOST_ALLOC_WRITE_COMBINED: u32 = 0x04;
const CUDA_HOST_REGISTER_DEFAULT: u32 = 0x00;
const CUDA_HOST_REGISTER_PORTABLE: u32 = 0x01;

fn check_cuda(code: i32, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Cuda(format!("{context} (cuda error {code})")))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PinnedAllocFlags(u32);

impl PinnedAllocFlags {
    pub const DEFAULT: Self = Self(CUDA_HOST_ALLOC_DEFAULT);
    pub const PORTABLE: Self = Self(CUDA_HOST_ALLOC_PORTABLE);
    pub const WRITE_COMBINED: Self = Self(CUDA_HOST_ALLOC_WRITE_COMBINED);

    #[inline]
    pub fn bits(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn contains(self, other: PinnedAllocFlags) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub fn portable_write_combined() -> Self {
        Self::PORTABLE | Self::WRITE_COMBINED
    }
}

impl Default for PinnedAllocFlags {
    fn default() -> Self {
        Self::WRITE_COMBINED
    }
}

impl BitOr for PinnedAllocFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PinnedAllocFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// CUDA pinned host buffer with typed element access.
pub struct PinnedHostBuffer<T> {
    ptr: NonNull<T>,
    capacity: usize,
    len: usize,
    flags: PinnedAllocFlags,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for PinnedHostBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedHostBuffer<T> {}

impl<T> PinnedHostBuffer<T> {
    pub fn with_capacity_elems(capacity: usize, flags: PinnedAllocFlags) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::InvalidInput(
                "PinnedHostBuffer::with_capacity_elems requires capacity > 0".into(),
            ));
        }
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::InvalidInput("PinnedHostBuffer byte size overflow".into()))?;
        let raw = unsafe { ffi::flame_cuda_alloc_pinned_host(bytes, flags.bits()) };
        let ptr = NonNull::new(raw as *mut T)
            .ok_or_else(|| Error::Cuda(format!("cudaHostAlloc returned null ({} bytes)", bytes)))?;
        Ok(Self {
            ptr,
            capacity,
            len: 0,
            flags,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn capacity_bytes(&self) -> usize {
        self.capacity * size_of::<T>()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn len_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    #[inline]
    pub fn flags(&self) -> PinnedAllocFlags {
        self.flags
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_capacity_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.len_bytes()) }
    }

    #[inline]
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.capacity_bytes()) }
    }

    pub fn copy_from_slice(&mut self, src: &[T])
    where
        T: Copy,
    {
        if src.len() > self.capacity {
            panic!("PinnedHostBuffer::copy_from_slice overflow");
        }
        let dst = self.as_mut_capacity_slice();
        dst[..src.len()].copy_from_slice(src);
        unsafe {
            self.set_len(src.len());
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity, "PinnedHostBuffer::set_len overflow");
        self.len = len;
    }

    pub fn reinterpret<'a, U>(&'a self) -> Result<PinnedHostBufferView<'a, U>> {
        let bytes = self.len_bytes();
        if bytes % size_of::<U>() != 0 {
            return Err(Error::InvalidOperation(
                "PinnedHostBuffer::reinterpret requires aligned length".into(),
            ));
        }
        let len = bytes / size_of::<U>();
        let ptr = self.ptr.as_ptr() as *const U;
        let slice = unsafe { slice::from_raw_parts(ptr, len) };
        Ok(PinnedHostBufferView { slice })
    }

    pub fn reinterpret_mut<'a, U>(&'a mut self) -> Result<PinnedHostBufferViewMut<'a, U>> {
        let bytes = self.len_bytes();
        if bytes % size_of::<U>() != 0 {
            return Err(Error::InvalidOperation(
                "PinnedHostBuffer::reinterpret_mut requires aligned length".into(),
            ));
        }
        let len = bytes / size_of::<U>();
        let ptr = self.ptr.as_ptr() as *mut U;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, len) };
        Ok(PinnedHostBufferViewMut { slice })
    }

    pub fn into_reinterpret<U>(self) -> Result<PinnedHostBuffer<U>> {
        let total_bytes = self.capacity_bytes();
        if total_bytes % size_of::<U>() != 0 {
            return Err(Error::InvalidOperation(
                "PinnedHostBuffer::into_reinterpret capacity misaligned".into(),
            ));
        }
        let len_bytes = self.len_bytes();
        if len_bytes % size_of::<U>() != 0 {
            return Err(Error::InvalidOperation(
                "PinnedHostBuffer::into_reinterpret length misaligned".into(),
            ));
        }
        let this = ManuallyDrop::new(self);
        let ptr = this.ptr.cast::<U>();
        let flags = this.flags;
        let capacity = total_bytes / size_of::<U>();
        let len = len_bytes / size_of::<U>();
        Ok(PinnedHostBuffer {
            ptr,
            capacity,
            len,
            flags,
            _marker: PhantomData,
        })
    }
}

impl<T> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::flame_cuda_free_pinned_host(self.ptr.as_ptr() as *mut c_void);
        }
    }
}

impl<T> std::fmt::Debug for PinnedHostBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHostBuffer")
            .field("capacity", &self.capacity)
            .field("len", &self.len)
            .field("flags", &self.flags)
            .finish()
    }
}

pub struct PinnedHostBufferView<'a, T> {
    slice: &'a [T],
}

impl<'a, T> PinnedHostBufferView<'a, T> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.slice
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slice.len()
    }
}

impl<'a, T> std::ops::Deref for PinnedHostBufferView<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

pub struct PinnedHostBufferViewMut<'a, T> {
    slice: &'a mut [T],
}

impl<'a, T> PinnedHostBufferViewMut<'a, T> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.slice
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.slice
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slice.len()
    }
}

impl<'a, T> std::ops::Deref for PinnedHostBufferViewMut<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<'a, T> std::ops::DerefMut for PinnedHostBufferViewMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
    }
}

/// Device buffer used for staged host→device transfers.
pub struct StagingDeviceBuf<T> {
    device: Arc<CudaDevice>,
    slice: CudaSlice<T>,
    len: usize,
}

impl<T> std::fmt::Debug for StagingDeviceBuf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingDeviceBuf")
            .field("len", &self.len)
            .field("device", &self.device.ordinal())
            .finish()
    }
}

impl<T> StagingDeviceBuf<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub fn new(device: Arc<CudaDevice>, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidInput(
                "StagingDeviceBuf::new len must be > 0".into(),
            ));
        }
        let slice = device
            .alloc_zeros::<T>(len)
            .map_err(|_| Error::CudaDriver)?;
        Ok(Self { device, slice, len })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    #[inline]
    pub fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    #[inline]
    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }

    pub fn into_inner(self) -> CudaSlice<T> {
        self.slice
    }

    pub fn async_upload_from(&self, host: &PinnedHostBuffer<T>, stream: &CudaStream) -> Result<()> {
        let bytes = host.len_bytes();
        if bytes > self.len * size_of::<T>() {
            return Err(Error::InvalidInput(
                "StagingDeviceBuf::async_upload_from host larger than device buffer".into(),
            ));
        }
        if bytes == 0 {
            return Ok(());
        }
        let dst = (*self.slice.device_ptr() as u64) as usize as *mut c_void;
        let src = host.as_ptr() as *const c_void;
        let stream_ptr = stream.stream as *mut c_void;
        check_cuda(
            unsafe {
                ffi::flame_cuda_memcpy_async(
                    dst,
                    src,
                    bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream_ptr,
                )
            },
            "cudaMemcpyAsync H2D",
        )
    }
}

pub unsafe fn register_slice_as_pinned<T>(slice: &mut [T], portable: bool) -> Result<()> {
    let flags = if portable {
        CUDA_HOST_REGISTER_PORTABLE
    } else {
        CUDA_HOST_REGISTER_DEFAULT
    };
    check_cuda(
        ffi::flame_cuda_host_register(
            slice.as_mut_ptr() as *mut c_void,
            slice.len() * size_of::<T>(),
            flags,
        ),
        "cudaHostRegister",
    )
}

pub unsafe fn unregister_pinned(ptr: *mut u8) -> Result<()> {
    if ptr.is_null() {
        return Ok(());
    }
    check_cuda(
        ffi::flame_cuda_host_unregister(ptr as *mut c_void),
        "cudaHostUnregister",
    )
}

pub fn memcpy_async_host_to_device(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: *mut c_void,
) -> Result<()> {
    check_cuda(
        unsafe {
            ffi::flame_cuda_memcpy_async(dst, src, bytes, CUDA_MEMCPY_HOST_TO_DEVICE, stream)
        },
        "cudaMemcpyAsync H2D",
    )
}

pub fn memcpy_async_device_to_host(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: *mut c_void,
) -> Result<()> {
    check_cuda(
        unsafe {
            ffi::flame_cuda_memcpy_async(dst, src, bytes, CUDA_MEMCPY_DEVICE_TO_HOST, stream)
        },
        "cudaMemcpyAsync D2H",
    )
}
