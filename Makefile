# ML-Datacode-lib — сборка libml (cdylib) по каналам: CPU, Metal (macOS), CUDA (Linux/Windows).
# Каждый канал использует свой --target-dir, чтобы release-артефакты не перезаписывали друг друга.

.PHONY: help install build-cpu build-metal build-cuda build-all dcmodule clean clean-dist

ROOT := $(CURDIR)
DIST_DIR := $(ROOT)/dist

TARGET_CPU := $(ROOT)/target/ml-cpu
TARGET_METAL := $(ROOT)/target/ml-metal
TARGET_CUDA := $(ROOT)/target/ml-cuda

UNAME_S := $(shell uname -s 2>/dev/null || echo Unknown)
ifeq ($(UNAME_S),Darwin)
LIBML := libml.dylib
else ifeq ($(UNAME_S),Linux)
LIBML := libml.so
else
# MINGW64_NT-*, MSYS*, Git Bash, MSVC окружения без uname — по умолчанию Windows DLL
LIBML := ml.dll
endif

help:
	@echo "ML (libml) — каналы сборки"
	@echo ""
	@echo "  make install       — установка submodules"
	@echo "  make build-cpu     — CPU only (без Candle GPU): dist/ml-cpu/$(LIBML)"
	@echo "  make build-metal   — Metal (только macOS):     dist/ml-metal/$(LIBML)"
	@echo "  make build-cuda    — CUDA (Linux/Windows):     dist/ml-cuda/$(LIBML)"
	@echo "  make build-all     — build-cpu + подходящие GPU-сборки для этой ОС"
	@echo "  make dcmodule      — релизный пакет dist/ml-<версия>-<target>.dcmodule (через dpm pack)"
	@echo ""
	@echo "Артефакты копируются в $(DIST_DIR)/ml-{cpu,metal,cuda}/"
	@echo "Промежуточные target-каталоги: target/ml-{cpu,metal,cuda}/"

install: 
	@echo "Installing submodules..."
	git submodule update --init --recursive datacode_sdk
	@echo "Submodules installed"

build-cpu : install
	cargo build --release --target-dir "$(TARGET_CPU)"
	@mkdir -p "$(DIST_DIR)/ml-cpu"
	@cp "$(TARGET_CPU)/release/$(LIBML)" "$(DIST_DIR)/ml-cpu/"
	@echo "OK: $(DIST_DIR)/ml-cpu/$(LIBML)"

build-metal: install
	@if [ "$(UNAME_S)" != "Darwin" ]; then \
		echo "Пропуск build-metal: Metal только на macOS"; \
		exit 0; \
	fi
	cargo build --release --features metal --target-dir "$(TARGET_METAL)"
	@mkdir -p "$(DIST_DIR)/ml-metal"
	@cp "$(TARGET_METAL)/release/$(LIBML)" "$(DIST_DIR)/ml-metal/"
	@echo "OK: $(DIST_DIR)/ml-metal/$(LIBML)"

build-cuda: install
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
		echo "Пропуск build-cuda: CUDA не собирается на macOS (используйте Metal или CPU)"; \
		exit 0; \
	fi
	cargo build --release --features cuda --target-dir "$(TARGET_CUDA)"
	@mkdir -p "$(DIST_DIR)/ml-cuda"
	@cp "$(TARGET_CUDA)/release/$(LIBML)" "$(DIST_DIR)/ml-cuda/"
	@echo "OK: $(DIST_DIR)/ml-cuda/$(LIBML)"

build-all: build-cpu
ifeq ($(UNAME_S),Darwin)
	@$(MAKE) build-metal
else
	@$(MAKE) build-cuda
endif

# Релизный бандл для registry artifacts: dist/ml-<version>-<target>.dcmodule + .sha256
DPM ?= dpm
ML_VERSION := $(shell sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml | head -1)
HOST_TARGET := $(shell rustc -vV 2>/dev/null | sed -n 's/^host: //p')
DCMODULE := $(DIST_DIR)/ml-$(ML_VERSION)-$(HOST_TARGET).dcmodule

# ABI, под которую собран libml (см. datacode_sdk); переопределяется: make dcmodule ABI_MINOR=8
ABI_MAJOR ?= 1
ABI_MINOR ?= 7
DCMODULE_STAGE := $(ROOT)/target/dcmodule-stage

dcmodule: install
ifeq ($(UNAME_S),Darwin)
	cargo build --release --features metal --target-dir "$(TARGET_METAL)"
	$(eval LIB_SRC := $(TARGET_METAL)/release/$(LIBML))
else
	cargo build --release --features cuda --target-dir "$(TARGET_CUDA)" || cargo build --release --target-dir "$(TARGET_CPU)"
	$(eval LIB_SRC := $(shell [ -f "$(TARGET_CUDA)/release/$(LIBML)" ] && echo "$(TARGET_CUDA)" || echo "$(TARGET_CPU)")/release/$(LIBML))
endif
	@rm -rf "$(DCMODULE_STAGE)" && mkdir -p "$(DCMODULE_STAGE)/lib/$(HOST_TARGET)" "$(DIST_DIR)"
	@cp "$(LIB_SRC)" "$(DCMODULE_STAGE)/lib/$(HOST_TARGET)/"
	@printf '{"schema_version":1,"name":"ml","version":"%s","abi_version":{"major":%s,"minor":%s},\n "targets":{"%s":"lib/%s/%s"}}\n' \
		"$(ML_VERSION)" "$(ABI_MAJOR)" "$(ABI_MINOR)" "$(HOST_TARGET)" "$(HOST_TARGET)" "$(LIBML)" > "$(DCMODULE_STAGE)/manifest.json"
	"$(DPM)" pack "$(DCMODULE_STAGE)" -o "$(DCMODULE)"
	@(shasum -a 256 "$(DCMODULE)" 2>/dev/null || sha256sum "$(DCMODULE)") | tee "$(DCMODULE).sha256"

clean-dist:
	rm -rf "$(DIST_DIR)"

clean:
	rm -rf "$(TARGET_CPU)" "$(TARGET_METAL)" "$(TARGET_CUDA)"
