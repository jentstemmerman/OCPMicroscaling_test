# Simple Makefile fallback if CMake is unavailable
# Usage:
#   make (builds library + example + test)
#   make run (runs example)
#   make test (builds and runs test)
#   make clean

# Prefer nvcc from an installed CUDA toolkit to avoid PATH mismatches (e.g., stale cudafe++)
# Users can still override by invoking `make NVCC=/path/to/nvcc`
NVCC ?= /usr/local/cuda/bin/nvcc
GPU_ARCH ?= sm_80
CXXFLAGS ?= -O3 -std=c++17 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Wpedantic
INCLUDES = -Iinclude
AR ?= ar
ARFLAGS = rcs

SRC_LIB = src/fp6.cu
OBJ_LIB = build/fp6.o
LIB = build/libmx_fp6.a

EXAMPLE_SRC = src/main.cu
EXAMPLE_BIN = build/example

TEST_SRC = tests/test_fp6.cu
TEST_BIN = build/test_fp6

.PHONY: all dirs clean run test

all: dirs $(LIB) $(EXAMPLE_BIN) $(TEST_BIN)

dirs:
	@mkdir -p build

$(OBJ_LIB): $(SRC_LIB) | dirs
	$(NVCC) $(CXXFLAGS) -arch=$(GPU_ARCH) $(INCLUDES) -c $< -o $@

$(LIB): $(OBJ_LIB)
	$(AR) $(ARFLAGS) $@ $^

$(EXAMPLE_BIN): $(EXAMPLE_SRC) $(LIB)
	$(NVCC) $(CXXFLAGS) -arch=$(GPU_ARCH) $(INCLUDES) $< $(LIB) -o $@

$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(NVCC) $(CXXFLAGS) -arch=$(GPU_ARCH) $(INCLUDES) $< $(LIB) -o $@

run: $(EXAMPLE_BIN)
	$(EXAMPLE_BIN)

test: $(TEST_BIN)
	$(TEST_BIN)

clean:
	rm -rf build
