//! Integration-level checks for backend registry and auto-selection policy.

use ml::BackendRegistry;

#[test]
fn available_devices_always_includes_cpu() {
    let r = BackendRegistry::detect();
    let d = r.available_devices();
    assert!(
        d.iter().any(|&x| x == "cpu"),
        "expected cpu in {:?}",
        d
    );
}

#[test]
fn auto_select_is_cpu_when_registry_has_no_gpu() {
    let r = BackendRegistry {
        cpu: true,
        metal: false,
        cuda: false,
    };
    assert_eq!(r.auto_select(), "cpu");
}

#[test]
#[cfg(target_os = "macos")]
fn auto_select_prefers_metal_on_macos_when_flagged() {
    let r = BackendRegistry {
        cpu: true,
        metal: true,
        cuda: false,
    };
    assert_eq!(r.auto_select(), "metal");
}

#[test]
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn auto_select_prefers_cuda_when_flagged() {
    let r = BackendRegistry {
        cpu: true,
        metal: false,
        cuda: true,
    };
    assert_eq!(r.auto_select(), "cuda");
}
