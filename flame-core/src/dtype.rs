// EXTRACTED FROM: candle-core/src/dtype.rs
// REASON: Standard dtype definitions we need for compatibility

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    F64,
    U8,
    U32,
    I64,
    I8,  // INT8 for Sage Attention
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::U8 | Self::I8 => 1,
            Self::U32 => 4,
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F64 => "f64",
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::I8 => "i8",
        }
    }
}

impl Default for DType {
    fn default() -> Self {
        Self::F32
    }
}