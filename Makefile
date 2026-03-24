# ML-Datacode-lib — сборка libml (cdylib) по каналам: CPU, Metal (macOS), CUDA (Linux/Windows).
# Каждый канал использует свой --target-dir, чтобы release-артефакты не перезаписывали друг друга.

.PHONY: help build-cpu build-metal build-cuda build-all clean clean-dist

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
	@echo "  make build-cpu     — CPU only (без Candle GPU): dist/ml-cpu/$(LIBML)"
	@echo "  make build-metal   — Metal (только macOS):     dist/ml-metal/$(LIBML)"
	@echo "  make build-cuda    — CUDA (Linux/Windows):     dist/ml-cuda/$(LIBML)"
	@echo "  make build-all     — build-cpu + подходящие GPU-сборки для этой ОС"
	@echo ""
	@echo "Артефакты копируются в $(DIST_DIR)/ml-{cpu,metal,cuda}/"
	@echo "Промежуточные target-каталоги: target/ml-{cpu,metal,cuda}/"

build-cpu:
	cargo build --release --target-dir "$(TARGET_CPU)"
	@mkdir -p "$(DIST_DIR)/ml-cpu"
	@cp "$(TARGET_CPU)/release/$(LIBML)" "$(DIST_DIR)/ml-cpu/"
	@echo "OK: $(DIST_DIR)/ml-cpu/$(LIBML)"

build-metal:
	@if [ "$(UNAME_S)" != "Darwin" ]; then \
		echo "Пропуск build-metal: Metal только на macOS"; \
		exit 0; \
	fi
	cargo build --release --features metal --target-dir "$(TARGET_METAL)"
	@mkdir -p "$(DIST_DIR)/ml-metal"
	@cp "$(TARGET_METAL)/release/$(LIBML)" "$(DIST_DIR)/ml-metal/"
	@echo "OK: $(DIST_DIR)/ml-metal/$(LIBML)"

build-cuda:
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

clean-dist:
	rm -rf "$(DIST_DIR)"

clean:
	rm -rf "$(TARGET_CPU)" "$(TARGET_METAL)" "$(TARGET_CUDA)"
