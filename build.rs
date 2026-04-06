fn main() {
    println!(
        "cargo:rustc-env={}=dist",
        datacode_sdk::module_dist::DIST_RUSTC_ENV
    );
    println!("cargo:rerun-if-changed=build.rs");
}
