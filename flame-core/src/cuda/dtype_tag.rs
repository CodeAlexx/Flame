use crate::DType;

#[repr(i32)]
pub enum DTypeTag { F32 = 0, F16 = 1, BF16 = 2, I32 = 3 }

pub const _: [(); 0] = [(); !(DTypeTag::F32 as i32 == 0
    && DTypeTag::F16 as i32 == 1
    && DTypeTag::BF16 as i32 == 2
    && DTypeTag::I32 as i32 == 3) as usize];

pub fn dtype_to_tag(dt: DType) -> i32 {
    match dt {
        DType::F32 => DTypeTag::F32 as i32,
        DType::F16 => DTypeTag::F16 as i32,
        DType::BF16 => DTypeTag::BF16 as i32,
        DType::I32 => DTypeTag::I32 as i32,
        _ => DTypeTag::F32 as i32,
    }
}
