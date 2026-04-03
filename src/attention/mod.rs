use cfg_if::cfg_if;

pub mod rope;

cfg_if! {
    if #[cfg(feature = "flash_attn")] {
        mod flash_ffi;
        mod flash_impl;
        mod sdpa;
        pub use flash_impl::attention_impl;
        pub use sdpa::{attend, sdpa, GeGLU};
    } else {
        mod sdpa;
        pub use sdpa::{attention_impl, attend, sdpa, GeGLU};
    }
}
