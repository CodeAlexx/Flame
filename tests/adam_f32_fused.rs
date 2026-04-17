//! Phase 2 Adam F32-param fused kernel numerical equivalence test.
//!
//! Runs 10 Adam steps on F32 params via the public `Adam::step` API, which
//! routes through the new `adam_fused_f32param_f32grad_kernel`. Compares
//! against a pure-host decoupled-AdamW reference implemented with
//! `Vec<f32>` — the reference cannot share a bug with the fused kernel.
//!
//! # Note on the BF16-grad variant
//!
//! The task prompt lists a third config: F32 param + BF16 grad. The public
//! `Parameter::set_grad` unconditionally casts the incoming grad to F32
//! (see `src/parameter.rs:90-94`), so by the time `Adam::step` reads the
//! grad via `param.grad()`, it is always F32. The BF16-grad F32-param
//! kernel path is therefore unreachable via the public API and is covered
//! instead by an inline module-internal test in `src/adam.rs`.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{adam::AdamW, global_cuda_device, Parameter, Result, Shape, Tensor};

/// Host-side decoupled-AdamW reference. Not reliant on any flame-core op so
/// it cannot share a bug with the fused CUDA kernel.
struct HostAdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: u32,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl HostAdamW {
    fn new(n: usize, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    fn step(&mut self, param: &mut [f32], grad: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..param.len() {
            let g = grad[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            let mut p = param[i];
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            if self.weight_decay > 0.0 {
                p -= self.lr * self.weight_decay * p;
            }
            param[i] = p;
        }
    }
}

/// Deterministic seeded pseudo-random f32 generator (xorshift32).
struct Xor32(u32);
impl Xor32 {
    fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        // Map to [-1, 1)
        (x as i32 as f32) / (i32::MAX as f32)
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

fn run_case(n: usize, steps: usize, lr: f32, wd: f32, init_seed: u32, grad_seed: u32) -> Result<()> {
    let device = global_cuda_device();
    let shape = Shape::from_dims(&[n]);

    let mut rng_init = Xor32::new(init_seed);
    let mut rng_grad = Xor32::new(grad_seed);

    let init: Vec<f32> = rng_init.vec(n, 0.1);
    let grads: Vec<Vec<f32>> = (0..steps).map(|_| rng_grad.vec(n, 0.01)).collect();

    // Fused F32 path
    let tensor = Tensor::from_vec(init.clone(), shape.clone(), device.clone())?.requires_grad_(true);
    let param = Parameter::new(tensor);
    let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, wd);
    for g in &grads {
        let grad = Tensor::from_vec(g.clone(), shape.clone(), device.clone())?;
        param.set_grad(grad)?;
        opt.step(std::slice::from_ref(&param))?;
    }
    let fused_out = param.tensor()?.to_vec_f32()?;

    // Host reference
    let mut ref_param = init.clone();
    let mut host = HostAdamW::new(n, lr, 0.9, 0.999, 1e-8, wd);
    for g in &grads {
        host.step(&mut ref_param, g);
    }

    // F32 precision: tolerance 1e-5 max-abs.
    let mut max_abs = 0f32;
    for (a, b) in fused_out.iter().zip(ref_param.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs <= 1e-5,
        "F32 fused Adam diverges from host reference: max_abs={max_abs} (n={n}, steps={steps}, wd={wd})"
    );

    Ok(())
}

#[test]
fn adam_f32_fused_no_weight_decay() -> Result<()> {
    run_case(4096, 10, 1e-3, 0.0, 0xC0FFEE01, 0xBEEF0001)
}

#[test]
fn adam_f32_fused_with_weight_decay() -> Result<()> {
    run_case(4096, 10, 1e-3, 0.01, 0xC0FFEE02, 0xBEEF0002)
}

/// Extra: non-power-of-two size to catch any rounding bugs in grid calc.
#[test]
fn adam_f32_fused_odd_size() -> Result<()> {
    run_case(3333, 10, 5e-4, 0.005, 0xC0FFEE03, 0xBEEF0003)
}
