# qgemm-mx build
#
# Plain make rather than CMake: the dev box has no cmake and no passwordless
# sudo, and the build is small enough not to need it. Add CMake later only if
# an upstream integration requires it.
#
#   make                 build everything for the local arch (default sm_86)
#   make ARCH=sm_90      build for Hopper
#   make probe           just the device/bandwidth probes
#   make run-probe       build and run both probes
#   make test-cpu        host unit tests (g++, no GPU)
#   make clean

CUDA_HOME ?= /usr/local/cuda-12.6
NVCC      := $(CUDA_HOME)/bin/nvcc

# sm_86 = RTX 30-series (dev box). sm_90 = H100 (authoritative results).
# sm_89 = Ada. Override on the command line, never edit this in place.
ARCH ?= sm_86

BUILD := build
INC   := -Iinclude

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr \
             -Xcompiler -Wall $(INC)
LDLIBS    := -lcublas

# Host-only tests (g++, no GPU / no nvcc required).
CXX      ?= g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -Iinclude
CPU_TESTS := $(BUILD)/test_formats $(BUILD)/test_pack_roundtrip \
             $(BUILD)/test_ref_gemm

PROBES  := $(BUILD)/device_props $(BUILD)/bandwidth
BENCHES := $(BUILD)/bench_cublas $(BUILD)/bench_stream_ideal
ALL     := $(PROBES) $(BENCHES)

.PHONY: all probe bench bench_cublas bench_stream run-probe run-stream clean \
        arch-info test-cpu

all: $(ALL)

probe: $(PROBES)
bench: $(BENCHES)
bench_cublas: $(BUILD)/bench_cublas
bench_stream: $(BUILD)/bench_stream_ideal

$(BUILD):
	@mkdir -p $(BUILD)

# CPU unit tests: formats math, pack/unpack roundtrip, structured probes.
test-cpu: $(CPU_TESTS)
	@echo "--- test_formats ---"
	@$(BUILD)/test_formats
	@echo "--- test_pack_roundtrip ---"
	@$(BUILD)/test_pack_roundtrip
	@echo "--- test_ref_gemm ---"
	@$(BUILD)/test_ref_gemm
	@echo "All CPU tests passed."

$(BUILD)/test_formats: tests/test_formats.cpp include/qgemm/formats.cuh \
                       include/qgemm/pack.hpp include/qgemm/shapes.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/test_pack_roundtrip: tests/test_pack_roundtrip.cpp \
                              include/qgemm/pack.hpp \
                              include/qgemm/correctness.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/test_ref_gemm: tests/test_ref_gemm.cpp include/qgemm/ref_gemm.hpp \
                        include/qgemm/pack.hpp include/qgemm/correctness.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/device_props: src/probe/device_props.cu include/qgemm/*.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/bandwidth: src/probe/bandwidth.cu include/qgemm/*.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/bench_cublas: src/bench/bench_cublas.cu include/qgemm/*.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/bench_stream_ideal: src/bench/bench_stream_ideal.cu include/qgemm/*.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Run both probes and capture the environment alongside them. Do this first on
# every new machine; the bandwidth figure is the project's denominator.
run-probe: $(PROBES)
	@bash scripts/env_capture.sh
	@echo
	@$(BUILD)/device_props
	@echo
	@$(BUILD)/bandwidth 256

# Pure weight-stream bound. Requires MEASURED_BW (GB/s) from ./bandwidth.
#   make run-stream MEASURED_BW=124.2
run-stream: $(BUILD)/bench_stream_ideal
	@if [ -z "$(MEASURED_BW)" ]; then \
	  echo "usage: make run-stream MEASURED_BW=<read-only GB/s from ./bandwidth>"; \
	  exit 2; \
	fi
	$(BUILD)/bench_stream_ideal 0 $(MEASURED_BW)

arch-info:
	@echo "CUDA_HOME = $(CUDA_HOME)"
	@echo "ARCH      = $(ARCH)"
	@$(NVCC) --version | tail -2

clean:
	rm -rf $(BUILD)
