use crate::{
    cuda::utils::cuda_mem_get_info, cuda_ops_bf16, ops_ext::shape4, perf_telemetry, DType, Error,
    Result, Tensor,
};
use std::sync::{Arc, Mutex, OnceLock};

/// Configuration for the BF16 streaming attention kernel.
#[derive(Clone, Copy)]
pub struct StreamingAttnCfg<'a> {
    pub scale: f32,
    pub chunk_size: usize,
    pub causal: bool,
    pub mask: Option<&'a Tensor>,
}

impl<'a> Default for StreamingAttnCfg<'a> {
    fn default() -> Self {
        Self {
            scale: 1.0,
            chunk_size: 2048,
            causal: false,
            mask: None,
        }
    }
}

fn strides4(t: &Tensor) -> Result<(usize, usize, usize, usize)> {
    let dims = t.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "expected 4D tensor for strides, got {:?}",
            dims
        )));
    }
    let stride = t.stride();
    Ok((stride[0], stride[1], stride[2], stride[3]))
}

fn ensure_packed_layout(
    name: &str,
    dims: (usize, usize, usize, usize),
    strides: (usize, usize, usize, usize),
) -> Result<()> {
    let (_, h, s, d) = dims;
    let (sb, sh, ss, sd) = strides;
    let expected_sd = 1;
    let expected_ss = d;
    let expected_sh = s * d;
    let expected_sb = h * s * d;
    if sd != expected_sd {
        return Err(Error::InvalidInput(format!(
            "{name} stride hidden mismatch: expected {expected_sd}, got {sd}"
        )));
    }
    if ss != expected_ss {
        return Err(Error::InvalidInput(format!(
            "{name} stride sequence mismatch: expected {expected_ss}, got {ss}"
        )));
    }
    if sh != expected_sh {
        return Err(Error::InvalidInput(format!(
            "{name} stride head mismatch: expected {expected_sh}, got {sh}"
        )));
    }
    if sb != expected_sb {
        return Err(Error::InvalidInput(format!(
            "{name} stride batch mismatch: expected {expected_sb}, got {sb}"
        )));
    }
    Ok(())
}

fn streaming_workspace_bytes(chunk_rows: usize, k_len: usize, dv: usize) -> Option<u64> {
    if chunk_rows == 0 || k_len == 0 {
        return Some(0);
    }
    let chunk_rows = chunk_rows as u64;
    let k_len = k_len as u64;
    let dv = dv as u64;
    let floats_attn = chunk_rows.checked_mul(k_len)?;
    let dv_term = dv.checked_add(2u64)?;
    let floats_value = chunk_rows.checked_mul(dv_term)?;
    let total_floats = floats_attn.checked_add(floats_value)?;
    total_floats.checked_mul(std::mem::size_of::<f32>() as u64)
}

fn is_cuda_oom_message(msg: &str) -> bool {
    msg.contains("cudaMalloc failed")
        || msg.contains("cudaErrorMemoryAllocation")
        || msg.contains("cudaErrorMemoryValueTooLarge")
        || msg.contains("cudaErrorLaunchOutOfResources")
        || msg.contains("(OOM)")
}

fn classify_fallback_reason(err: &Error) -> Option<StreamingFallbackReason> {
    match err {
        Error::OutOfMemory(_) => Some(StreamingFallbackReason::Oom {
            error_code: -1,
            required_bytes: None,
            budget_bytes: None,
        }),
        Error::Cuda(msg) if is_cuda_oom_message(msg) => Some(StreamingFallbackReason::Oom {
            error_code: -1,
            required_bytes: None,
            budget_bytes: None,
        }),
        _ => None,
    }
}

const STREAM_ATTENTION_MIN_CHUNK: usize = 32;
const STREAM_ATTENTION_MAX_CHUNK: usize = 2048;
const STREAM_ATTENTION_DEFAULT_SCALE: f32 = 1.0;
const STREAM_ATTENTION_MIN_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
const STREAM_ATTENTION_DEFAULT_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
const STREAM_ATTENTION_HARD_CAP_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

#[derive(Clone, Debug)]
struct StreamingTilePlan {
    chunk: usize,
    workspace_bytes: u64,
    budget_bytes: u64,
}

fn streaming_chunk_candidates(q_len: usize) -> Vec<usize> {
    const PREFERRED: [usize; 13] = [
        2048, 1536, 1024, 768, 512, 384, 256, 192, 128, 96, 64, 48, 32,
    ];
    if q_len == 0 {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    for &cand in PREFERRED.iter() {
        if cand == 0 {
            continue;
        }
        let candidate = cand.min(q_len);
        if candidate == 0 {
            continue;
        }
        if candidate < STREAM_ATTENTION_MIN_CHUNK && q_len >= STREAM_ATTENTION_MIN_CHUNK {
            continue;
        }
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    if candidates.is_empty() {
        candidates.push(q_len.max(1));
    }
    candidates.sort_unstable_by(|a, b| b.cmp(a));
    candidates.dedup();
    candidates
}

fn plan_streaming_chunk(q_len: usize, k_len: usize, dv: usize) -> Result<StreamingTilePlan> {
    if q_len == 0 || k_len == 0 || dv == 0 {
        return Ok(StreamingTilePlan {
            chunk: q_len.max(1),
            workspace_bytes: 0,
            budget_bytes: 0,
        });
    }

    let forced_budget = std::env::var("FLAME_SDPA_PLANNER_BUDGET_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok());

    let mut budget = if let Some(bytes) = forced_budget {
        bytes
    } else {
        let (free_bytes_raw, _) = cuda_mem_get_info().unwrap_or((0, 0));
        if free_bytes_raw > 0 {
            ((free_bytes_raw as f64) * 0.75) as u64
        } else {
            STREAM_ATTENTION_DEFAULT_BUDGET_BYTES
        }
    };
    if forced_budget.is_none() {
        budget = budget.max(STREAM_ATTENTION_MIN_BUDGET_BYTES);
    }
    budget = budget.min(STREAM_ATTENTION_HARD_CAP_BYTES);

    let candidates = streaming_chunk_candidates(q_len);
    for &chunk in &candidates {
        let chunk_rows = chunk.min(q_len);
        if let Some(required) = streaming_workspace_bytes(chunk_rows, k_len, dv) {
            if required <= budget {
                return Ok(StreamingTilePlan {
                    chunk,
                    workspace_bytes: required,
                    budget_bytes: budget,
                });
            }
        }
    }

    let smallest = *candidates.last().unwrap_or(&1);
    let chunk_rows = smallest.min(q_len);
    let required = streaming_workspace_bytes(chunk_rows, k_len, dv)
        .ok_or_else(|| Error::InvalidInput("streaming attention planner overflow".into()))?;
    Err(Error::OutOfMemory(format!(
        "streaming attention planner: chunk {} requires {:.2} MiB workspace but budget is {:.2} MiB",
        smallest,
        required as f64 / BYTES_PER_MIB,
        budget as f64 / BYTES_PER_MIB,
    )))
}

#[derive(Clone, Debug)]
pub enum StreamingFallbackReason {
    Oom {
        error_code: i32,
        required_bytes: Option<u64>,
        budget_bytes: Option<u64>,
    },
    WorkspaceBudget {
        required_bytes: u64,
        budget_bytes: u64,
    },
}

#[derive(Clone, Debug)]
pub struct StreamingFallbackEvent {
    pub from_chunk: usize,
    pub to_chunk: usize,
    pub reason: StreamingFallbackReason,
}

#[derive(Clone, Debug, Default)]
pub struct StreamingAttnLaunchInfo {
    pub requested_chunk: usize,
    pub chosen_chunk: Option<usize>,
    pub retries: usize,
    pub fallbacks: Vec<StreamingFallbackEvent>,
    pub last_error: Option<i32>,
}

impl StreamingAttnLaunchInfo {
    fn new(requested_chunk: usize) -> Self {
        Self {
            requested_chunk,
            chosen_chunk: None,
            retries: 0,
            fallbacks: Vec::new(),
            last_error: None,
        }
    }

    fn chunk_path(&self, final_chunk: usize) -> String {
        let mut segments = Vec::new();
        if self.requested_chunk > 0 {
            segments.push(self.requested_chunk);
        }
        for ev in &self.fallbacks {
            if segments.last().copied() != Some(ev.from_chunk) {
                segments.push(ev.from_chunk);
            }
            if segments.last().copied() != Some(ev.to_chunk) {
                segments.push(ev.to_chunk);
            }
        }
        if segments.last().copied() != Some(final_chunk) {
            segments.push(final_chunk);
        }
        segments.dedup();
        let parts: Vec<String> = segments.into_iter().map(|c| c.to_string()).collect();
        format!("[{}]", parts.join("→"))
    }
}

impl StreamingFallbackReason {
    fn telemetry_fields(
        &self,
    ) -> (
        &'static str,
        Option<u64>,
        Option<u64>,
        Option<usize>,
        Option<i32>,
    ) {
        match self {
            StreamingFallbackReason::Oom {
                error_code,
                required_bytes,
                budget_bytes,
            } => (
                "oom",
                *required_bytes,
                *budget_bytes,
                None,
                Some(*error_code),
            ),
            StreamingFallbackReason::WorkspaceBudget {
                required_bytes,
                budget_bytes,
            } => (
                "workspace_budget",
                Some(*required_bytes),
                Some(*budget_bytes),
                None,
                None,
            ),
        }
    }
}

static LAST_STREAMING_ATTN_LAUNCH: OnceLock<Mutex<StreamingAttnLaunchInfo>> = OnceLock::new();

fn record_launch_info(info: StreamingAttnLaunchInfo) {
    let lock =
        LAST_STREAMING_ATTN_LAUNCH.get_or_init(|| Mutex::new(StreamingAttnLaunchInfo::default()));
    if let Ok(mut guard) = lock.lock() {
        *guard = info;
    }
}

pub fn streaming_attn_last_launch_info() -> StreamingAttnLaunchInfo {
    LAST_STREAMING_ATTN_LAUNCH
        .get()
        .and_then(|lock| lock.lock().ok().map(|guard| (*guard).clone()))
        .unwrap_or_default()
}

/// Launch the CUDA streaming attention kernel.
pub fn streaming_attn_bf16_fp32(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: StreamingAttnCfg<'_>,
) -> Result<Tensor> {
    if cfg.chunk_size == 0 {
        return Err(Error::InvalidInput(
            "streaming attention chunk_size must be > 0".into(),
        ));
    }

    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "streaming attention expects BF16 tensors".into(),
        ));
    }
    if q.device().ordinal() != k.device().ordinal() || q.device().ordinal() != v.device().ordinal()
    {
        return Err(Error::InvalidInput(
            "streaming attention requires tensors on the same device".into(),
        ));
    }

    let (b, h, s, dh) = shape4(q)?;
    let (_, _, s_k, dh_k) = shape4(k)?;
    let (_, _, s_v, dv) = shape4(v)?;

    if s != s_k || s != s_v {
        return Err(Error::InvalidInput("sequence lengths do not match".into()));
    }
    if dh != dh_k {
        return Err(Error::InvalidInput(
            "head dimensions for Q and K differ".into(),
        ));
    }

    ensure_packed_layout("q", (b, h, s, dh), strides4(q)?)?;
    ensure_packed_layout("k", (b, h, s, dh), strides4(k)?)?;
    ensure_packed_layout("v", (b, h, s, dv), strides4(v)?)?;

    if cfg.mask.is_some() {
        return Err(Error::Unsupported(
            "streaming_attn_bf16_fp32: mask support not yet implemented".into(),
        ));
    }

    let mut plan = plan_streaming_chunk(s, s_k, dv)?;

    let requested_chunk = cfg.chunk_size.max(1);
    let requested_clamped = requested_chunk
        .min(s.max(1))
        .min(STREAM_ATTENTION_MAX_CHUNK);
    let mut requested_required = None;
    if requested_clamped > 0 {
        requested_required = streaming_workspace_bytes(requested_clamped.min(s), s_k, dv);
        if let Some(required) = requested_required {
            if required <= plan.budget_bytes {
                plan.chunk = requested_clamped;
                plan.workspace_bytes = required;
            }
        }
    }

    let mut chunk_candidates = streaming_chunk_candidates(s);
    if !chunk_candidates.contains(&plan.chunk) {
        chunk_candidates.push(plan.chunk);
        chunk_candidates.sort_unstable_by(|a, b| b.cmp(a));
        chunk_candidates.dedup();
    }
    if chunk_candidates.is_empty() {
        return Err(Error::InvalidInput(
            "no valid chunk size candidates for streaming attention".into(),
        ));
    }

    let mut current_idx = chunk_candidates
        .iter()
        .position(|&c| c == plan.chunk)
        .unwrap_or(0);

    let mut info = StreamingAttnLaunchInfo::new(requested_clamped.max(1));
    if let Some(required) = requested_required {
        if required > plan.budget_bytes && requested_clamped > plan.chunk {
            info.fallbacks.push(StreamingFallbackEvent {
                from_chunk: requested_clamped,
                to_chunk: plan.chunk,
                reason: StreamingFallbackReason::WorkspaceBudget {
                    required_bytes: required,
                    budget_bytes: plan.budget_bytes,
                },
            });
            info.retries = info.fallbacks.len();
            if perf_telemetry::telemetry_enabled() {
                perf_telemetry::log_sdpa_fallback_event(
                    perf_telemetry::SdpaFallbackTelemetryRecord {
                        batch: b,
                        heads: h,
                        q_len: s,
                        k_len: s_k,
                        dh,
                        dv,
                        from_chunk: requested_clamped,
                        to_chunk: plan.chunk,
                        reason: "workspace_budget",
                        workspace_required: Some(required),
                        workspace_limit: Some(plan.budget_bytes),
                        free_mem_mb: None,
                        error_code: None,
                    },
                );
            }
        }
    }
    let mut last_err: Option<Error> = None;

    println!(
        "[streaming_attn_debug] plan={{chunk: {}, budget_mib: {:.2}, workspace_mib: {:.2}}} seq={} scale={} causal={}",
        plan.chunk,
        plan.budget_bytes as f64 / BYTES_PER_MIB,
        plan.workspace_bytes as f64 / BYTES_PER_MIB,
        s,
        cfg.scale,
        cfg.causal
    );

    while current_idx < chunk_candidates.len() {
        let chunk_candidate = chunk_candidates[current_idx];
        let chunk_rows = chunk_candidate.min(s);
        let required_bytes = streaming_workspace_bytes(chunk_rows, s_k, dv)
            .ok_or_else(|| Error::InvalidInput("streaming attention workspace overflow".into()))?;

        println!(
            "[streaming_attn_debug] attempt={} chunk={} required_mib={:.2}",
            current_idx,
            chunk_candidate,
            required_bytes as f64 / BYTES_PER_MIB
        );

        if required_bytes > plan.budget_bytes {
            let next_idx = current_idx + 1;
            if next_idx >= chunk_candidates.len() {
                info.chosen_chunk = Some(chunk_candidate);
                record_launch_info(info);
                return Err(Error::OutOfMemory(format!(
                    "streaming attention planner rejected chunk {}: requires {:.2} MiB workspace, budget {:.2} MiB",
                    chunk_candidate,
                    required_bytes as f64 / BYTES_PER_MIB,
                    plan.budget_bytes as f64 / BYTES_PER_MIB
                )));
            }
            let next_chunk = chunk_candidates[next_idx];
            let reason = StreamingFallbackReason::WorkspaceBudget {
                required_bytes,
                budget_bytes: plan.budget_bytes,
            };
            if perf_telemetry::telemetry_enabled() {
                let (label, workspace_required, workspace_limit, free_mb, error_code) =
                    reason.telemetry_fields();
                perf_telemetry::log_sdpa_fallback_event(
                    perf_telemetry::SdpaFallbackTelemetryRecord {
                        batch: b,
                        heads: h,
                        q_len: s,
                        k_len: s_k,
                        dh,
                        dv,
                        from_chunk: chunk_candidate,
                        to_chunk: next_chunk,
                        reason: label,
                        workspace_required,
                        workspace_limit,
                        free_mem_mb: free_mb,
                        error_code,
                    },
                );
            }
            info.fallbacks.push(StreamingFallbackEvent {
                from_chunk: chunk_candidate,
                to_chunk: next_chunk,
                reason,
            });
            info.retries = info.fallbacks.len();
            current_idx = next_idx;
            continue;
        }

        match cuda_ops_bf16::sdpa_stream_bf16_with_workspace(
            q,
            k,
            v,
            cfg.mask,
            chunk_candidate,
            cfg.causal,
            Some(cfg.scale),
            None,
        ) {
            Ok(result) => {
                info.chosen_chunk = Some(chunk_candidate);
                info.retries = info.fallbacks.len();
                record_launch_info(info);
                println!(
                    "[streaming_attn_debug] selected_chunk={} seq={} scale={}",
                    chunk_candidate, s, cfg.scale
                );
                return Ok(result);
            }
            Err(err) => {
                let next_idx = current_idx + 1;
                if next_idx >= chunk_candidates.len() {
                    info.chosen_chunk = Some(chunk_candidate);
                    record_launch_info(info);
                    return Err(err);
                }

                let next_chunk = chunk_candidates[next_idx];
                let mut reason =
                    classify_fallback_reason(&err).unwrap_or(StreamingFallbackReason::Oom {
                        error_code: -1,
                        required_bytes: None,
                        budget_bytes: None,
                    });
                if let StreamingFallbackReason::Oom {
                    required_bytes: recorded_required,
                    budget_bytes: recorded_budget,
                    ..
                } = &mut reason
                {
                    *recorded_required = Some(required_bytes);
                    *recorded_budget = Some(plan.budget_bytes);
                }

                if perf_telemetry::telemetry_enabled() {
                    let (label, workspace_required, workspace_limit, free_mb, error_code) =
                        reason.telemetry_fields();
                    perf_telemetry::log_sdpa_fallback_event(
                        perf_telemetry::SdpaFallbackTelemetryRecord {
                            batch: b,
                            heads: h,
                            q_len: s,
                            k_len: s_k,
                            dh,
                            dv,
                            from_chunk: chunk_candidate,
                            to_chunk: next_chunk,
                            reason: label,
                            workspace_required: workspace_required.or(Some(required_bytes)),
                            workspace_limit: workspace_limit.or(Some(plan.budget_bytes)),
                            free_mem_mb: free_mb,
                            error_code,
                        },
                    );
                }

                info.fallbacks.push(StreamingFallbackEvent {
                    from_chunk: chunk_candidate,
                    to_chunk: next_chunk,
                    reason,
                });
                info.retries = info.fallbacks.len();
                info.last_error = Some(-1);
                last_err = Some(err);
                current_idx = next_idx;
            }
        }
    }

    record_launch_info(info);
    Err(last_err.unwrap_or_else(|| {
        Error::OutOfMemory("streaming attention exhausted chunk candidates without success".into())
    }))
}

/// Launch the kernel on a tiny configuration to validate launch parameters.
pub fn streaming_attn_bf16_fp32_smoke_test(device: Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    let b = 1;
    let h = 1;
    let s = 128;
    let dh = 64;
    let dv = 64;
    let shape = crate::Shape::from_dims(&[b, h, s, dh]);

    let q = Tensor::zeros_dtype(shape.clone(), DType::BF16, device.clone())?;
    let k = Tensor::zeros_dtype(shape.clone(), DType::BF16, device.clone())?;
    let v = Tensor::zeros_dtype(shape.clone(), DType::BF16, device.clone())?;

    cuda_ops_bf16::sdpa_stream_bf16(
        &q,
        &k,
        &v,
        None,
        64,
        false,
        Some(STREAM_ATTENTION_DEFAULT_SCALE),
    )?;

    println!(
        "[streaming_attn_smoke_debug] launch succeeded for dims [B={},H={},S={},Dh={},Dv={}] chunk={}",
        b, h, s, dh, dv, 64
    );

    Ok(())
}
