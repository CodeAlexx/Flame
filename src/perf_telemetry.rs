use std::fs::{create_dir_all, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct SdpaTelemetryRecord {
    pub batch: usize,
    pub heads: usize,
    pub q_len: usize,
    pub k_len: usize,
    pub dh: usize,
    pub dv: usize,
    pub chunk_requested: usize,
    pub tuned_q_chunk: u64,
    pub tuned_k_chunk: u64,
    pub plan_source: u64,
    pub mask_heads: usize,
    pub has_mask: bool,
    pub causal: bool,
    pub workspace_bytes: u64,
    pub stream_id: u64,
    pub elapsed_ms: f64,
    pub tflops: f64,
    pub scale: f32,
    pub candidate_count: u64,
    pub best_time_ns: u64,
    pub autotune_env_forced: u64,
    pub autotune_clamped: u64,
    pub autotune_skipped: u64,
    pub autotune_fallback: u64,
    pub autotune_errors: u64,
    pub autotune_tuned: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_saved: u64,
    pub cache_loads: u64,
    pub cache_load_errors: u64,
    pub cache_entries: u64,
}

pub struct SdpaFallbackTelemetryRecord {
    pub batch: usize,
    pub heads: usize,
    pub q_len: usize,
    pub k_len: usize,
    pub dh: usize,
    pub dv: usize,
    pub from_chunk: usize,
    pub to_chunk: usize,
    pub reason: &'static str,
    pub workspace_required: Option<u64>,
    pub workspace_limit: Option<u64>,
    pub free_mem_mb: Option<usize>,
    pub error_code: Option<i32>,
}

static TELEMETRY_ENABLED: OnceLock<bool> = OnceLock::new();
static TELEMETRY_WRITER: OnceLock<Mutex<Option<BufWriter<std::fs::File>>>> = OnceLock::new();

pub fn telemetry_enabled() -> bool {
    *TELEMETRY_ENABLED.get_or_init(|| {
        std::env::var("FLAME_SDPA_TELEMETRY_PATH")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    })
}

fn init_writer() -> Option<BufWriter<std::fs::File>> {
    let path = std::env::var("FLAME_SDPA_TELEMETRY_PATH").ok()?;
    if path.is_empty() {
        return None;
    }
    let path = PathBuf::from(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent).ok()?;
        }
    }
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()?;
    Some(BufWriter::new(file))
}

fn with_writer<F>(mut f: F)
where
    F: FnMut(&mut BufWriter<std::fs::File>),
{
    let mutex = TELEMETRY_WRITER.get_or_init(|| Mutex::new(init_writer()));
    if let Ok(mut guard) = mutex.lock() {
        if guard.is_none() {
            *guard = init_writer();
        }
        if let Some(writer) = guard.as_mut() {
            f(writer);
        }
    }
}

pub fn log_sdpa_event(record: SdpaTelemetryRecord) {
    if !telemetry_enabled() {
        return;
    }
    with_writer(|writer| {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let json = format!(
            "{{\"op\":\"sdpa_stream_bf16\",\"timestamp_ns\":{},\"batch\":{},\"heads\":{},\"q_len\":{},\"k_len\":{},\"dh\":{},\"dv\":{},\"chunk_requested\":{},\"tuned_q_chunk\":{},\"tuned_k_chunk\":{},\"plan_source\":{},\"mask_heads\":{},\"has_mask\":{},\"causal\":{},\"workspace_bytes\":{},\"stream_id\":{},\"elapsed_ms\":{:.6},\"tflops\":{:.6},\"scale\":{:.6},\"candidate_count\":{},\"best_time_ns\":{},\"autotune_env_forced\":{},\"autotune_clamped\":{},\"autotune_skipped\":{},\"autotune_fallback\":{},\"autotune_errors\":{},\"autotune_tuned\":{},\"cache_hits\":{},\"cache_misses\":{},\"cache_saved\":{},\"cache_loads\":{},\"cache_load_errors\":{},\"cache_entries\":{}}}\n",
            timestamp_ns,
            record.batch,
            record.heads,
            record.q_len,
            record.k_len,
            record.dh,
            record.dv,
            record.chunk_requested,
            record.tuned_q_chunk,
            record.tuned_k_chunk,
            record.plan_source,
            record.mask_heads,
            record.has_mask as u8,
            record.causal as u8,
            record.workspace_bytes,
            record.stream_id,
            record.elapsed_ms,
            record.tflops,
            record.scale,
            record.candidate_count,
            record.best_time_ns,
            record.autotune_env_forced,
            record.autotune_clamped,
            record.autotune_skipped,
            record.autotune_fallback,
            record.autotune_errors,
            record.autotune_tuned,
            record.cache_hits,
            record.cache_misses,
            record.cache_saved,
            record.cache_loads,
            record.cache_load_errors,
            record.cache_entries
        );
        let _ = writer.write_all(json.as_bytes());
        let _ = writer.flush();
    });
}

pub fn log_sdpa_fallback_event(record: SdpaFallbackTelemetryRecord) {
    if !telemetry_enabled() {
        return;
    }
    let workspace_required = record
        .workspace_required
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string());
    let workspace_limit = record
        .workspace_limit
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string());
    let free_mem_mb = record
        .free_mem_mb
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string());
    let error_code = record
        .error_code
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string());

    with_writer(|writer| {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let json = format!(
            "{{\"op\":\"sdpa_stream_fallback\",\"timestamp_ns\":{},\"batch\":{},\"heads\":{},\"q_len\":{},\"k_len\":{},\"dh\":{},\"dv\":{},\"from_chunk\":{},\"to_chunk\":{},\"reason\":\"{}\",\"workspace_required_bytes\":{},\"workspace_limit_bytes\":{},\"free_mem_mb\":{},\"error_code\":{}}}\n",
            timestamp_ns,
            record.batch,
            record.heads,
            record.q_len,
            record.k_len,
            record.dh,
            record.dv,
            record.from_chunk,
            record.to_chunk,
            record.reason,
            workspace_required,
            workspace_limit,
            free_mem_mb,
            error_code
        );
        let _ = writer.write_all(json.as_bytes());
        let _ = writer.flush();
    });
}
