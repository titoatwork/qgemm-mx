# qgemm-mx build
#
# Plain make rather than CMake: the dev box has no cmake and no passwordless
# sudo, and the build is small enough not to need it. Add CMake later only if
# an upstream integration requires it.
#
#   make                 build everything for the local arch (default sm_86)
#   make ARCH=sm_90      build for Hopper
#   make probe           just the device/bandwidth probes
#   make bench           all benches (cublas, stream, naive, deq, qgemv)
#   make bench_cublas    FP16 cuBLAS baseline
#   make bench_stream    packed-weight stream ideal
#   make bench_naive     dummy nibble-expand + cuBLAS (traffic-shaped floor)
#   make bench_deq       real pack.hpp dequant + cuBLAS FP16 (sanity floor)
#   make bench_qgemv     fused-ref qgemv (correct-slow; not a SOTA claim)
#   make run-probe       build and run both probes + env_capture
#   make run-stream MEASURED_BW=<gbps>
#   make test-cpu        host unit tests (g++, no GPU)
#   make test-gpu        fused-ref GPU correctness (nvcc)
#   make local-peak      scripts/run_local_peak.sh
#   make sanitize        compute-sanitizer memcheck on test_qgemv_gpu
#   make arch-info       print CUDA_HOME, ARCH, nvcc
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
             $(BUILD)/test_ref_gemm $(BUILD)/test_occupancy \
             $(BUILD)/test_contract

PROBES  := $(BUILD)/device_props $(BUILD)/bandwidth
BENCHES := $(BUILD)/bench_cublas $(BUILD)/bench_stream_ideal \
           $(BUILD)/bench_naive_dequant $(BUILD)/bench_qgemv_ref \
           $(BUILD)/bench_dequant_then_cublas $(BUILD)/bench_waterfall
ALL     := $(PROBES) $(BENCHES)

.PHONY: all probe bench bench_cublas bench_stream bench_naive bench_qgemv \
        bench_deq run-probe run-stream clean arch-info test-cpu test-gpu \
        local-peak sanitize

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
	@echo "--- test_occupancy ---"
	@$(BUILD)/test_occupancy
	@echo "--- test_contract ---"
	@$(BUILD)/test_contract
	@echo "All CPU tests passed."

$(BUILD)/test_contract: tests/test_contract.cpp include/qgemm/kernel_contract.hpp \
                        include/qgemm/occupancy.hpp include/qgemm/shapes.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/test_occupancy: tests/test_occupancy.cpp include/qgemm/occupancy.hpp \
                         include/qgemm/shapes.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/test_qgemv_gpu: tests/test_qgemv_gpu.cu include/qgemm/qgemv_ref.cuh \
                         include/qgemm/dequant_device.cuh include/qgemm/pack.hpp \
                         include/qgemm/timing.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/test_qgemv_probes: tests/test_qgemv_probes.cu include/qgemm/qgemv_ref.cuh \
                            include/qgemm/dequant_device.cuh include/qgemm/pack.hpp \
                            include/qgemm/correctness.hpp include/qgemm/ref_gemm.hpp \
                            include/qgemm/timing.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/test_bittrick_splitk: tests/test_bittrick_splitk.cu \
                               include/qgemm/qgemv_ref.cuh \
                               include/qgemm/dequant_device.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# GPU correctness for the fused-ref kernel (sm_86 lab). Not a perf claim.
$(BUILD)/test_mx_decode: tests/test_mx_decode.cu include/qgemm/dequant_device.cuh \
                         include/qgemm/timing.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

test-gpu: $(BUILD)/test_qgemv_gpu $(BUILD)/test_qgemv_probes \
          $(BUILD)/test_bittrick_splitk $(BUILD)/test_mx_decode
	@echo "--- test_qgemv_gpu ---"
	@$(BUILD)/test_qgemv_gpu
	@echo "--- test_qgemv_probes ---"
	@$(BUILD)/test_qgemv_probes
	@echo "--- test_bittrick_splitk ---"
	@$(BUILD)/test_bittrick_splitk
	@echo "--- test_mx_decode ---"
	@$(BUILD)/test_mx_decode
	@echo "GPU fused-ref tests passed."

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

$(BUILD)/bench_naive_dequant: src/bench/bench_naive_dequant.cu include/qgemm/*.cuh \
                              include/qgemm/shapes.hpp | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/bench_qgemv_ref: src/bench/bench_qgemv_ref.cu include/qgemm/*.cuh \
                          include/qgemm/*.hpp | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/bench_dequant_then_cublas: src/bench/bench_dequant_then_cublas.cu \
                                    include/qgemm/*.cuh include/qgemm/*.hpp | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/bench_waterfall: src/bench/bench_waterfall.cu include/qgemm/*.cuh \
                          include/qgemm/*.hpp | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

bench_naive: $(BUILD)/bench_naive_dequant
bench_qgemv: $(BUILD)/bench_qgemv_ref
bench_deq: $(BUILD)/bench_dequant_then_cublas
bench_waterfall: $(BUILD)/bench_waterfall

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

# CPU+GPU tests, occupancy table, optional short benches if binaries exist.
local-peak:
	@bash scripts/run_local_peak.sh

# compute-sanitizer memcheck; no-op if build/test_qgemv_gpu is missing.
sanitize:
	@bash scripts/sanitize_qgemv.sh

arch-info:
	@echo "CUDA_HOME = $(CUDA_HOME)"
	@echo "ARCH      = $(ARCH)"
	@$(NVCC) --version | tail -2

clean:
	rm -rf $(BUILD)
