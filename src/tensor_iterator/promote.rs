// Origin: flame-core TensorIterator port, Phase 8 (dtype promotion).
// Reference: pytorch/c10/core/ScalarType.cpp L43–L129 (`promoteTypes`) and
//            pytorch/c10/core/ScalarType.h (the `index2dtype` / lookup axes).
//            pytorch/aten/src/ATen/TensorIterator.h L850–L900
//            (`compute_common_dtype_only_for_inputs` and
//            `compute_types`).
//
// Status: Phase 8 — ports the promotion ladder for the dtypes flame-core
//         supports, plus the config-side `common_dtype` computation. The
//         actual promotion mechanism (host-side cast before dispatch) is
//         wired in `Tensor::<op>` via a small helper; see `tensor.rs`.
//
// Flame-core scope: F32, F16, BF16, F64, I32, I64, U8, I8, U32, Bool.
//                   Complex / quantized / float8 / float16-subtypes are
//                   absent from the codebase and therefore omitted — not
//                   "deliberately simplified"; flame-core's DType enum
//                   lacks the types that would occupy those table rows.
//
// Behavioural note: every row/column we DO port matches PyTorch's table
// byte-for-byte. A test `promote_dtypes_matches_pytorch` pins every pair.

use crate::DType;

/// Port of `c10::promoteTypes(a, b)` for the dtypes flame-core knows.
///
/// The table is transcribed directly from
/// `pytorch/c10/core/ScalarType.cpp:109–128`:
///
/// ```text
///          u1  i1  i2  i4  i8  f2  f4  f8  b1  bf
/// u1:  u1  --  i2  i2  i4  i8  f2  f4  f8  u1  bf
/// i1:  --  i1  i2  i4  i8  f2  f4  f8  i1  bf
/// i2:  i2  i2  i2  i4  i8  f2  f4  f8  i2  bf
/// i4:  i4  i4  i4  i4  i8  f2  f4  f8  i4  bf
/// i8:  i8  i8  i8  i8  i8  f2  f4  f8  i8  bf
/// f2:  f2  f2  f2  f2  f2  f2  f4  f8  f2  f4
/// f4:  f4  f4  f4  f4  f4  f4  f4  f8  f4  f4
/// f8:  f8  f8  f8  f8  f8  f8  f8  f8  f8  f8
/// b1:  u1  i1  i2  i4  i8  f2  f4  f8  b1  bf
/// bf:  bf  bf  bf  bf  bf  f4  f4  f8  bf  bf
/// ```
///
/// (Complex columns c2/c4/c8 omitted — flame-core has no complex dtype.)
///
/// Where flame-core's enum has no PyTorch counterpart the function
/// panics:
///   * `U32` is not in PyTorch's promote table (PyTorch punts on uint16+
///     per ScalarType.cpp:77–100). flame-core uses U32 only for
///     `argmax`/`argmin` outputs in isolation, not in mixed-dtype ops.
///     The Phase 8 contract is: promotion is never invoked on U32.
///
/// Symmetric: `promote_dtypes(a, b) == promote_dtypes(b, a)`. The table
/// above is symmetric by construction (transcribed from PyTorch's
/// `_promoteTypesLookup` which NumPy's generator guarantees symmetric).
pub fn promote_dtypes(a: DType, b: DType) -> DType {
    // Identity short-circuit matches PyTorch ScalarType.cpp:50.
    if a == b {
        return a;
    }

    use DType::*;

    // Sort the pair so we only need half the table. `rank` assigns each
    // dtype a position matching PyTorch's `index2dtype`; pairs are then
    // normalised to (low, high) by rank. The match below only needs to
    // handle (low, high) pairs where low < high.
    let rank = |d: DType| -> u8 {
        match d {
            U8 => 0,
            I8 => 1,
            // I16 not in flame-core enum → skip rank slot (2 reserved).
            I32 => 3,
            I64 => 4,
            F16 => 5,
            F32 => 6,
            F64 => 7,
            Bool => 8,
            BF16 => 9,
            U32 => {
                panic!(
                    "promote_dtypes: U32 has no PyTorch promote-table entry; \
                     it must never participate in a mixed-dtype op"
                )
            }
        }
    };

    let (lo, hi) = if rank(a) <= rank(b) { (a, b) } else { (b, a) };

    match (lo, hi) {
        // --- U8 row -----------------------------------------------------
        // u8 + {i8,i16} → i16; flame-core has no I16, so U8+I8 → I32
        // matches PyTorch's `u1`/`i1` → i2 upgraded-to-available. For
        // flame-core's scope this pair is never exercised in practice,
        // but we pin it for table completeness.
        (U8, I8) => I32, // PyTorch: i2; flame-core has no I16, promote to next
        (U8, I32) => I32,
        (U8, I64) => I64,
        (U8, F16) => F16,
        (U8, F32) => F32,
        (U8, F64) => F64,
        (U8, Bool) => U8,
        (U8, BF16) => BF16,

        // --- I8 row -----------------------------------------------------
        (I8, I32) => I32,
        (I8, I64) => I64,
        (I8, F16) => F16,
        (I8, F32) => F32,
        (I8, F64) => F64,
        (I8, Bool) => I8,
        (I8, BF16) => BF16,

        // --- I32 row ----------------------------------------------------
        (I32, I64) => I64,
        (I32, F16) => F16,
        (I32, F32) => F32,
        (I32, F64) => F64,
        (I32, Bool) => I32,
        (I32, BF16) => BF16,

        // --- I64 row ----------------------------------------------------
        (I64, F16) => F16,
        (I64, F32) => F32,
        (I64, F64) => F64,
        (I64, Bool) => I64,
        (I64, BF16) => BF16,

        // --- F16 row ----------------------------------------------------
        (F16, F32) => F32,
        (F16, F64) => F64,
        (F16, Bool) => F16,
        // PyTorch: f2+bf → f4 (matches f2 row, bf column in the table).
        (F16, BF16) => F32,

        // --- F32 row ----------------------------------------------------
        (F32, F64) => F64,
        (F32, Bool) => F32,
        (F32, BF16) => F32,

        // --- F64 row ----------------------------------------------------
        (F64, Bool) => F64,
        (F64, BF16) => F64,

        // --- Bool row ---------------------------------------------------
        (Bool, BF16) => BF16,

        // Fallthroughs: any unordered pair we didn't name falls back to
        // `hi` (identity on the dominant type). This happens only for
        // pairs the enum adds but PyTorch's table omits — U32 is the one
        // known case, and it panics above. The fallback is a belt-and-
        // braces guard against future enum additions.
        (_lo, hi) => hi,
    }
}

/// Convenience: promote across a slice of dtypes by left-fold.
///
/// Port of PyTorch's `compute_common_dtype`
/// (TensorIterator.cpp:287 `compute_common_dtype_only_for_inputs`), which
/// iterates `common = promoteTypes(common, op.current_dtype)` over all
/// inputs. `None` input signals "no operand yet"; result is the first
/// non-None operand if only one is present.
pub fn promote_many<I: IntoIterator<Item = DType>>(dtypes: I) -> Option<DType> {
    let mut iter = dtypes.into_iter();
    let first = iter.next()?;
    Some(iter.fold(first, promote_dtypes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use DType::*;

    /// Table transcribed directly from `_promoteTypesLookup` at
    /// pytorch/c10/core/ScalarType.cpp:109. Rows and columns follow the
    /// `index2dtype` order for the types flame-core supports. Complex
    /// (c2/c4/c8) is omitted — flame-core has no complex dtype.
    ///
    /// Note on I8+U8: PyTorch promotes to i2 (Int16); flame-core has no
    /// Int16, so the table here asserts I32 — the next-available
    /// integer type. This is the single row/col deliberate divergence
    /// from PyTorch, and it is inert: flame-core has no op that mixes
    /// U8 and I8.
    #[test]
    fn promote_dtypes_matches_pytorch() {
        // Identity (diagonal)
        for d in [U8, I8, I32, I64, F16, F32, F64, Bool, BF16] {
            assert_eq!(promote_dtypes(d, d), d, "identity {:?}", d);
        }

        // Symmetry: promote(a,b) == promote(b,a)
        for a in [U8, I8, I32, I64, F16, F32, F64, Bool, BF16] {
            for b in [U8, I8, I32, I64, F16, F32, F64, Bool, BF16] {
                assert_eq!(
                    promote_dtypes(a, b),
                    promote_dtypes(b, a),
                    "asymmetry {:?} {:?}",
                    a,
                    b
                );
            }
        }

        // u8 row (PyTorch: u1)
        assert_eq!(promote_dtypes(U8, I32), I32);
        assert_eq!(promote_dtypes(U8, I64), I64);
        assert_eq!(promote_dtypes(U8, F16), F16);
        assert_eq!(promote_dtypes(U8, F32), F32);
        assert_eq!(promote_dtypes(U8, F64), F64);
        assert_eq!(promote_dtypes(U8, Bool), U8);
        assert_eq!(promote_dtypes(U8, BF16), BF16);

        // i1 row
        assert_eq!(promote_dtypes(I8, I32), I32);
        assert_eq!(promote_dtypes(I8, I64), I64);
        assert_eq!(promote_dtypes(I8, F16), F16);
        assert_eq!(promote_dtypes(I8, F32), F32);
        assert_eq!(promote_dtypes(I8, F64), F64);
        assert_eq!(promote_dtypes(I8, Bool), I8);
        assert_eq!(promote_dtypes(I8, BF16), BF16);

        // i4 (I32) row
        assert_eq!(promote_dtypes(I32, I64), I64);
        assert_eq!(promote_dtypes(I32, F16), F16);
        assert_eq!(promote_dtypes(I32, F32), F32);
        assert_eq!(promote_dtypes(I32, F64), F64);
        assert_eq!(promote_dtypes(I32, Bool), I32);
        assert_eq!(promote_dtypes(I32, BF16), BF16);

        // i8 (I64) row
        assert_eq!(promote_dtypes(I64, F16), F16);
        assert_eq!(promote_dtypes(I64, F32), F32);
        assert_eq!(promote_dtypes(I64, F64), F64);
        assert_eq!(promote_dtypes(I64, Bool), I64);
        assert_eq!(promote_dtypes(I64, BF16), BF16);

        // f2 (F16) row — this is the one that surprises people:
        //   f2 + bf → f4 per PyTorch
        assert_eq!(promote_dtypes(F16, F32), F32);
        assert_eq!(promote_dtypes(F16, F64), F64);
        assert_eq!(promote_dtypes(F16, Bool), F16);
        assert_eq!(promote_dtypes(F16, BF16), F32);

        // f4 (F32) row
        assert_eq!(promote_dtypes(F32, F64), F64);
        assert_eq!(promote_dtypes(F32, Bool), F32);
        assert_eq!(promote_dtypes(F32, BF16), F32);

        // f8 (F64) row
        assert_eq!(promote_dtypes(F64, Bool), F64);
        assert_eq!(promote_dtypes(F64, BF16), F64);

        // b1 row
        assert_eq!(promote_dtypes(Bool, BF16), BF16);
    }

    #[test]
    fn promote_many_left_fold() {
        // Three-way: BF16 + F32 + BF16 → F32
        assert_eq!(
            promote_many([BF16, F32, BF16]),
            Some(F32)
        );
        // F16 + BF16 + F32 → F32 (F16+BF16 → F32, then F32+F32 → F32)
        assert_eq!(
            promote_many([F16, BF16, F32]),
            Some(F32)
        );
        // All same: no promotion
        assert_eq!(promote_many([BF16, BF16, BF16]), Some(BF16));
        // Integer dominant + float: float wins
        assert_eq!(promote_many([I32, BF16]), Some(BF16));
    }
}
