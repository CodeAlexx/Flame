fn main() {
    if std::env::var("CARGO_FEATURE_CAPI").is_ok() {
        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("flame.h");
        cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .generate()
            .expect("cbindgen generation failed")
            .write_to_file(out);
    }
}

