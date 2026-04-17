// Low-level CUDA FFI declarations for narrow kernels

extern "C" {
    pub fn cudaStreamCreate(stream: *mut *mut core::ffi::c_void) -> i32;
    pub fn cudaStreamDestroy(stream: *mut core::ffi::c_void) -> i32;

    pub fn cublasLtCreate(handle: *mut *mut core::ffi::c_void) -> i32;
    pub fn cublasLtDestroy(handle: *mut core::ffi::c_void) -> i32;

    pub fn flame_narrow_strided_launch(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        rank: i32,
        out_shape_host: *const i64,
        src_strides_host: *const i64,
        out_strides_host: *const i64,
        dim: i32,
        start: i64,
        elem_size: i64,
        n_elements: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn narrow_backward_scatter_add_launch(
        grad_out: *const core::ffi::c_void,
        grad_in: *mut core::ffi::c_void,
        rank: i32,
        out_shape_host: *const i64,
        in_strides_host: *const i64,
        out_strides_host: *const i64,
        dim: i32,
        start: i64,
        elem_size: i64,
        n_elements: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Launch helper for the (N, A, B, C) -> (N, B, A, C) permute kernel.
    /// Keeping this in FFI means Rust never needs to stage tensors on CPU to
    /// shuffle attention layouts.
    pub fn launch_permute0213_f32(
        src: *const f32,
        dst: *mut f32,
        N: i32,
        A: i32,
        B: i32,
        C: i32,
        stream: *mut core::ffi::c_void,
    );

    /// BF16 entry point mirrors the F32 signature but operates on u16 storage
    /// (each element is CUDA’s __nv_bfloat16).  We pass it as raw void pointers
    /// because Rust does not expose that intrinsic type directly.
    pub fn launch_permute0213_bf16(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        N: i32,
        A: i32,
        B: i32,
        C: i32,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_permute021_f32(
        src: *const f32,
        dst: *mut f32,
        N: i32,
        A: i32,
        B: i32,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_permute021_bf16(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        N: i32,
        A: i32,
        B: i32,
        stream: *mut core::ffi::c_void,
    );

    // Pinned host memory helpers
    pub fn flame_cuda_alloc_pinned_host(size: usize, flags: u32) -> *mut core::ffi::c_void;
    pub fn flame_cuda_free_pinned_host(ptr: *mut core::ffi::c_void) -> i32;
    /// kind: 1=H2D, 2=D2H, 3=D2D, default otherwise.
    pub fn flame_cuda_memcpy_async(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        size: usize,
        kind: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn flame_cuda_host_register(ptr: *mut core::ffi::c_void, size: usize, flags: u32) -> i32;
    pub fn flame_cuda_host_unregister(ptr: *mut core::ffi::c_void) -> i32;

    pub fn launch_sum_last_keepdim_bf16(
        x: *const core::ffi::c_void,
        y: *mut core::ffi::c_void,
        B: i32,
        M: i32,
        K: i32,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_sum_last_keepdim_bf16_to_f32(
        x: *const core::ffi::c_void,
        y: *mut core::ffi::c_void,
        B: i32,
        M: i32,
        K: i32,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_add_inplace_f32(
        dst: *mut f32,
        src: *const f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_add_inplace_bf16(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        n: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_mul_inplace_f32(
        dst: *mut f32,
        src: *const f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_mul_inplace_bf16(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        n: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_mul_scalar_f32(
        dst: *mut f32,
        src: *const f32,
        scalar: f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_mul_scalar_bf16(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        scalar: f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_add_scalar_f32(
        dst: *mut f32,
        src: *const f32,
        scalar: f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_add_scalar_bf16(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        scalar: f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_broadcast_f32(
        src: *const f32,
        dst: *mut f32,
        out_shape: *const i64,
        in_stride: *const i64,
        out_stride: *const i64,
        ndim: i32,
        total: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_broadcast_bf16(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        out_shape: *const i64,
        in_stride: *const i64,
        out_stride: *const i64,
        ndim: i32,
        total: i64,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_modulate_affine_bf16(
        dst: *mut core::ffi::c_void,
        shift: *const core::ffi::c_void,
        scale: *const core::ffi::c_void,
        batch: i32,
        tokens: i32,
        hidden: i32,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_gate_mul_bf16(
        dst: *mut core::ffi::c_void,
        gate: *const core::ffi::c_void,
        batch: i32,
        tokens: i32,
        hidden: i32,
        stream: *mut core::ffi::c_void,
    );

    pub fn launch_tile_bc_to_bhwc_f32(
        src: *const f32,
        dst: *mut f32,
        B: i32,
        H: i32,
        W: i32,
        C: i32,
        stream: *mut core::ffi::c_void,
    );

    pub fn flame_rope_apply_bf16_fp32(
        input: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        B: i32,
        H: i32,
        S: i32,
        Dh: i32,
        rope_dim: i32,
        base_theta: f32,
        pos_offset: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_silu_backward_bf16(
        grad_out: *const core::ffi::c_void,
        input: *const core::ffi::c_void,
        grad_in: *mut core::ffi::c_void,
        n: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_silu_backward_f32(
        grad_out: *const core::ffi::c_void,
        input: *const core::ffi::c_void,
        grad_in: *mut core::ffi::c_void,
        n: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_swiglu_backward_bf16(
        grad_out: *const core::ffi::c_void,
        gate: *const core::ffi::c_void,
        up: *const core::ffi::c_void,
        d_gate: *mut core::ffi::c_void,
        d_up: *mut core::ffi::c_void,
        n: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_rope_precomputed_bf16(
        x: *const core::ffi::c_void,
        cos_buf: *const core::ffi::c_void,
        sin_buf: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        B: i32,
        H: i32,
        S: i32,
        D: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_apply_causal_mask_fp32(
        scores: *mut f32,
        B: i32,
        H: i32,
        Q: i32,
        K: i32,
        q_offset: i32,
        k_offset: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_apply_attn_mask_fp32(
        scores: *mut f32,
        mask: *const u8,
        B: i32,
        H: i32,
        Q: i32,
        K: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_sdpa_add_mask_tile_fp32(
        logits: *mut f32,
        user_bool: *const u8,
        user_add: *const f32,
        BH: i32,
        q_t: i32,
        k_t: i32,
        q_abs_start: i32,
        k_abs_start: i32,
        user_bool_rank: i32,
        user_add_rank: i32,
        k_total: i32,
        bool_zero_is_mask: i32,
        causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_sdpa_softmax_from_lse_tile(
        logits: *const f32,
        lse_row: *const f32,
        probs_bf16: *mut core::ffi::c_void,
        BH: i32,
        q_t: i32,
        k_t: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_sdpa_lse_from_logits_tile(
        logits: *const f32,
        out_lse_row: *mut f32,
        BH: i32,
        q_t: i32,
        k_t: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_sdpa_lse_merge_rows(
        lse_row: *mut f32,
        tile_lse_row: *const f32,
        BH: i32,
        q_t: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_sdpa_dropout_bf16_inplace(
        probs_bf16: *mut core::ffi::c_void,
        p: f32,
        seed: u64,
        BH: i32,
        q_t: i32,
        k_t: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_geglu_pointwise_fp32(
        gated: *const f32,
        value: *const f32,
        out: *mut f32,
        n: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn gemm_bf16_fp32acc_stridedBatched(
        lt: *mut core::ffi::c_void,
        opA: i32,
        opB: i32,
        m: i32,
        n: i32,
        k: i32,
        A: *const core::ffi::c_void,
        lda: i64,
        strideA: i64,
        B: *const core::ffi::c_void,
        ldb: i64,
        strideB: i64,
        C: *mut core::ffi::c_void,
        ldc: i64,
        strideC: i64,
        batchCount: i32,
        alpha: f32,
        beta: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn streaming_attn_bf16_fp32_launch(
        Q: *const core::ffi::c_void,
        K: *const core::ffi::c_void,
        V: *const core::ffi::c_void,
        B: i32,
        H: i32,
        S: i32,
        Dh: i32,
        Dv: i32,
        qB: i64,
        qH: i64,
        qS: i64,
        qD: i64,
        kB: i64,
        kH: i64,
        kS: i64,
        kD: i64,
        vB: i64,
        vH: i64,
        vS: i64,
        vD: i64,
        O: *mut core::ffi::c_void,
        oB: i64,
        oH: i64,
        oS: i64,
        oD: i64,
        scale: f32,
        chunk_size: i32,
        causal: i32,
        mask: *const u8,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn streaming_attn_bf16_fp32_attrs(
        max_threads_per_block: *mut i32,
        static_shared_bytes: *mut i32,
        binary_version: *mut i32,
    ) -> i32;

    pub fn fc_upsample2d_nearest_bf16(
        input: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        batch: i32,
        channels: i32,
        h_in: i32,
        w_in: i32,
        h_out: i32,
        w_out: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn fc_upsample2d_nearest_f32(
        input: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        batch: i32,
        channels: i32,
        h_in: i32,
        w_in: i32,
        h_out: i32,
        w_out: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    // ── Fused inference kernels ──────────────────────────────────────

    /// GPU-side FP8 E4M3 → BF16 dequant: out[i] = bf16(fp8(in[i]) * scale)
    pub fn flame_fp8_to_bf16(
        input: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        scale: f32,
        n_elements: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// GPU-side BF16 → FP8 E4M3 quantize: out[i] = fp8(bf16(in[i]) * inv_scale)
    /// Caller provides inv_scale = 1.0 / scale where scale = absmax / 448.0.
    pub fn flame_bf16_to_fp8(
        input: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        inv_scale: f32,
        n_elements: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// GPU-side FP16 (IEEE half) → BF16 conversion.
    /// In-place safe (both 2 bytes per element).
    pub fn flame_f32_to_bf16(
        input: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        n: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_fp16_to_bf16(
        input: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        n_elements: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Flash attention forward: BF16 in/out, FP32 accumulation, online softmax.
    /// Q,K,V: [B*H, N, HD] BF16. O: [B*H, N, HD] BF16.
    /// LSE: [B*H, N_q] FP32, can be null. Outputs log-sum-exp per row for backward pass.
    /// head_dim must be 64, 96, or 128. Returns 0 on success.
    pub fn flame_flash_attention_bf16(
        Q: *const core::ffi::c_void,
        K: *const core::ffi::c_void,
        V: *const core::ffi::c_void,
        O: *mut core::ffi::c_void,
        LSE: *mut f32,
        batch_heads: i32,
        seq_len_q: i32,
        seq_len_kv: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Legacy BQ=32 WMMA forward kernel. Preserved for Phase 1 parity testing
    /// (`tests/fa2_parity.rs`) — same signature as `flame_flash_attention_bf16`.
    /// Will be deleted in Phase 3. Do not use in production paths.
    pub fn flame_flash_attention_bf16_wmma_legacy(
        Q: *const core::ffi::c_void,
        K: *const core::ffi::c_void,
        V: *const core::ffi::c_void,
        O: *mut core::ffi::c_void,
        LSE: *mut f32,
        batch_heads: i32,
        seq_len_q: i32,
        seq_len_kv: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Flash attention backward — wmma tensor core version.
    /// Computes dQ, dK, dV from Q, K, V, O, dO, and LSE.
    /// Q,K,V,O,dO: [B*H, N, HD] BF16. LSE: [B*H, N_q] FP32.
    /// dQ,dK,dV: [B*H, N, HD] BF16 output.
    /// head_dim must be 64, 96, or 128. Returns 0 on success.
    /// Flash attention backward — wmma tensor core version.
    /// dK, dV written directly as BF16 (register accumulation, no staging).
    /// dQ uses FP32 staging buffer (atomicAdd across KV-tile blocks).
    pub fn flame_flash_attention_backward_bf16(
        Q: *const core::ffi::c_void,
        K: *const core::ffi::c_void,
        V: *const core::ffi::c_void,
        O: *const core::ffi::c_void,
        dO: *const core::ffi::c_void,
        LSE: *const core::ffi::c_void,
        dQ: *mut core::ffi::c_void,
        dK: *mut core::ffi::c_void,
        dV: *mut core::ffi::c_void,
        batch_heads: i32,
        seq_len_q: i32,
        seq_len_kv: i32,
        head_dim: i32,
        stream: *mut core::ffi::c_void,
        dQ_f32: *mut f32,  // Pre-allocated FP32 staging [BH, N_q, HD], zeroed
    ) -> i32;

    /// Fused RMS norm + modulation: out = rms_norm(x, w) * (1+scale) + shift.
    /// Replaces fused_rms_norm + fused_modulate (2 kernels → 1).
    pub fn flame_fused_rms_norm_modulate_bf16(
        x: *const core::ffi::c_void,
        norm_weight: *const core::ffi::c_void,
        scale: *const core::ffi::c_void,
        shift: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Fused residual + gating: out = x + gate * attn_out.
    /// Replaces mul + add (2 kernels → 1).
    pub fn flame_fused_residual_gate_bf16(
        x: *const core::ffi::c_void,
        attn_out: *const core::ffi::c_void,
        gate: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        n_elements: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Fused RMS norm: BF16 in → BF16 out, with weight multiply.
    /// Replaces 6 kernel launches (cast + sq + mean + rsqrt + mul + mul) with 1.
    pub fn flame_fused_rms_norm_bf16(
        input: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Fused modulation: out = x * (1 + scale) + shift. All BF16.
    /// Replaces 4 kernel launches (add_scalar + cast + mul + add) with 1.
    pub fn flame_fused_modulate_bf16(
        x: *const core::ffi::c_void,
        scale: *const core::ffi::c_void,
        shift: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        n_elements: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Fused FP8 E4M3 dequant + transpose: [M,N] FP8 -> [N,M] BF16.
    /// One kernel launch, zero allocation. Output must be pre-allocated.
    pub fn flame_fused_dequant_transpose_bf16(
        input: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        scale: f32,
        m: i32,
        n: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Fused 3D linear via cublasLt: [B,N,Cin] @ [Cout,Cin]^T + bias = [B,N,Cout].
    /// No reshape kernels. Bias fused into GEMM epilogue.
    /// Replaces 4 launches (reshape + gemm + reshape + bias_add) with 1 cublasLt call.
    pub fn flame_linear3d_bf16(
        handle: *mut core::ffi::c_void,
        input: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        bias: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        batch_size: i32,
        seq_len: i32,
        in_features: i32,
        out_features: i32,
        workspace: *mut core::ffi::c_void,
        workspace_size: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Same as flame_linear3d_bf16 but takes the weight in standard PyTorch
    /// `[Cout, Cin]` row-major layout (no pre-transpose). cuBLASLt does the
    /// transpose inside the GEMM via TRANSA=T, eliminating the wasted
    /// transpose pass that flux1_dit's `linear_bias` was paying every call.
    pub fn flame_linear3d_bf16_native(
        handle: *mut core::ffi::c_void,
        input: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        bias: *const core::ffi::c_void,
        output: *mut core::ffi::c_void,
        batch_size: i32,
        seq_len: i32,
        in_features: i32,
        out_features: i32,
        workspace: *mut core::ffi::c_void,
        workspace_size: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Fused gated scatter-add for MoE unpermute:
    /// `accum[indices[t]] += expert_out[t] * gating[t]`, in-place.
    ///
    /// - `expert_out`: BF16 (T, D), per-token expert outputs in expert-major order
    /// - `gating`:     F32  (T,)    per-token gating weight
    /// - `indices`:    I32  (T,)    global token indices (out-of-range rows skipped)
    /// - `accum`:      F32  (N, D)  running accumulator, updated in-place
    ///
    /// Uses F32 `atomicAdd` since multiple t's may collide on the same row
    /// (MoE top-K with K>1). Returns cudaError_t (0 on success).
    pub fn flame_fused_gated_scatter_add_bf16(
        expert_out: *const core::ffi::c_void,
        gating: *const core::ffi::c_void,
        indices: *const core::ffi::c_void,
        accum: *mut core::ffi::c_void,
        T: i32,
        D: i32,
        N: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Grouped BF16 matmul for MoE: one kernel computes all E experts.
    ///
    /// Semantics mirror `torch.nn.functional.grouped_mm(x, w, offs=offsets)`:
    /// - `x`: (T, K) BF16, expert-major ordered tokens
    /// - `w`: (E, K, N) BF16, stacked weights (PyTorch layout)
    /// - `offsets`: (E,) i32, EXCLUSIVE cumulative end indices into x
    ///   (expert e covers rows [offsets[e-1] .. offsets[e]), with offsets[-1] := 0)
    /// - `y`: (T, N) BF16 output
    /// - `T_max`: max rows across experts (used to size grid.y)
    ///
    /// Accumulation is FP32. Returns 0 on success, cudaError_t otherwise.
    pub fn flame_grouped_mm_bf16(
        x: *const core::ffi::c_void,
        w: *const core::ffi::c_void,
        offsets: *const core::ffi::c_void,
        y: *mut core::ffi::c_void,
        T_max: i32,
        K: i32,
        N: i32,
        E: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    // ── CUDA Graph capture / replay ─────────────────────────────────
    //
    // These are CUDA Runtime API functions linked via -lcudart.
    // Used by `cuda_graph.rs` for recording the backward pass as a
    // replayable graph, eliminating per-kernel launch overhead.

    /// Begin capturing kernel launches on `stream` into a graph.
    /// `mode`: 0 = cudaStreamCaptureModeGlobal,
    ///         1 = cudaStreamCaptureModeThreadLocal,
    ///         2 = cudaStreamCaptureModeRelaxed.
    pub fn cudaStreamBeginCapture(stream: *mut core::ffi::c_void, mode: i32) -> i32;

    /// End capture on `stream` and return the captured graph in `*graph`.
    pub fn cudaStreamEndCapture(
        stream: *mut core::ffi::c_void,
        graph: *mut *mut core::ffi::c_void,
    ) -> i32;

    /// Instantiate a captured graph into an executable form.
    /// `error_node` and `log`/`log_size` are for debug diagnostics (may be null/0).
    pub fn cudaGraphInstantiate(
        graph_exec: *mut *mut core::ffi::c_void,
        graph: *mut core::ffi::c_void,
        error_node: *mut *mut core::ffi::c_void,
        log: *mut core::ffi::c_char,
        log_size: usize,
    ) -> i32;

    /// Launch a previously instantiated graph on `stream`.
    pub fn cudaGraphLaunch(graph_exec: *mut core::ffi::c_void, stream: *mut core::ffi::c_void)
        -> i32;

    /// Destroy a captured graph (not the executable).
    pub fn cudaGraphDestroy(graph: *mut core::ffi::c_void) -> i32;

    /// Destroy an instantiated (executable) graph.
    pub fn cudaGraphExecDestroy(graph_exec: *mut core::ffi::c_void) -> i32;

    /// Synchronize all pending work on `stream`.
    pub fn cudaStreamSynchronize(stream: *mut core::ffi::c_void) -> i32;
}
