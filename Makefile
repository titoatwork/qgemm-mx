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

PROBES  := $(BUILD)/device_props $(BUILD)/bandwidth
ALL     := $(PROBES)

.PHONY: all probe run-probe clean arch-info

all: $(ALL)

probe: $(PROBES)

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/device_props: src/probe/device_props.cu include/qgemm/*.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/bandwidth: src/probe/bandwidth.cu include/qgemm/*.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Run both probes and capture the environment alongside them. Do this first on
# every new machine; the bandwidth figure is the project's denominator.
run-probe: $(PROBES)
	@bash scripts/env_capture.sh
	@echo
	@$(BUILD)/device_props
	@echo
	@$(BUILD)/bandwidth 256

arch-info:
	@echo "CUDA_HOME = $(CUDA_HOME)"
	@echo "ARCH      = $(ARCH)"
	@$(NVCC) --version | tail -2

clean:
	rm -rf $(BUILD)
