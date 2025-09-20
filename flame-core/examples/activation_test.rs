#![cfg(feature = "legacy_examples")]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]
#![cfg_attr(
    clippy,
    allow(
        clippy::unused_imports,
        clippy::useless_vec,
        clippy::needless_borrow,
        clippy::needless_clone
    )
)]

fn main() {
    // Minimal example gated by `dev-examples` feature.
    // Run with:
    //   cargo run -p flame-core --example activation_test --features dev-examples
    println!("activation_test: ok");
}
