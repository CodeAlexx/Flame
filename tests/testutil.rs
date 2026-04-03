use flame_core::Error;

pub fn assert_mixed_dtype_err(e: &Error) {
    let msg = e.to_string().to_lowercase();
    assert!(
        msg.contains("dtype") && msg.contains("mismatch"),
        "expected a dtype mismatch error; got: {msg}"
    );
}
