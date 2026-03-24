// Backend registry for runtime device discovery
// Provides two-level checking: compile-time (cargo features) + runtime (device availability)

/// Registry для обнаружения доступных backend'ов во время выполнения
#[derive(Debug, Clone)]
pub struct BackendRegistry {
    pub cpu: bool,
    pub metal: bool,
    pub cuda: bool,
}

impl BackendRegistry {
    /// Обнаружить доступные backend'ы при старте интерпретатора
    pub fn detect() -> Self {
        Self {
            cpu: true,  // CPU всегда доступен
            metal: Self::detect_metal(),
            cuda: Self::detect_cuda(),
        }
    }

    /// Обнаружить Metal backend (двухуровневая проверка)
    fn detect_metal() -> bool {
        #[cfg(feature = "metal")]
        {
            // Compile-time: код есть (metal feature включает gpu)
            // Runtime: проверяем доступность устройства
            candle_core::Device::new_metal(0).is_ok()
        }

        #[cfg(not(feature = "metal"))]
        {
            // Compile-time: кода нет
            false
        }
    }

    /// Обнаружить CUDA backend (двухуровневая проверка)
    fn detect_cuda() -> bool {
        #[cfg(feature = "cuda")]
        {
            // Compile-time: код есть (cuda feature включает gpu)
            // Runtime: проверяем доступность устройства
            candle_core::Device::new_cuda(0).is_ok()
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Compile-time: кода нет
            false
        }
    }

    /// Получить список доступных устройств для пользователя
    pub fn available_devices(&self) -> Vec<&'static str> {
        let mut devices = vec!["cpu"];
        if self.metal {
            devices.push("metal");
        }
        if self.cuda {
            devices.push("cuda");
        }
        devices
    }

    /// Автоматически выбрать лучшее доступное устройство (должно совпадать с [`crate::Device::default`]).
    /// macOS: Metal при наличии, иначе CPU. Linux/Windows: CUDA при наличии, иначе CPU.
    pub fn auto_select(&self) -> &'static str {
        #[cfg(target_os = "macos")]
        {
            if self.metal {
                "metal"
            } else {
                "cpu"
            }
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            if self.cuda {
                "cuda"
            } else {
                "cpu"
            }
        }
        #[cfg(not(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )))]
        {
            let _ = (self.metal, self.cuda);
            "cpu"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BackendRegistry;

    #[test]
    fn auto_select_cpu_when_no_gpu_backends() {
        let r = BackendRegistry {
            cpu: true,
            metal: false,
            cuda: false,
        };
        assert_eq!(r.auto_select(), "cpu");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn auto_select_macos_prefers_metal_when_available() {
        let r = BackendRegistry {
            cpu: true,
            metal: true,
            cuda: false,
        };
        assert_eq!(r.auto_select(), "metal");
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn auto_select_linux_windows_prefers_cuda_when_available() {
        let r = BackendRegistry {
            cpu: true,
            metal: false,
            cuda: true,
        };
        assert_eq!(r.auto_select(), "cuda");
    }
}
