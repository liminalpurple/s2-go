# Builds libs2go.a: s2.cpp, its ggml and the cgo wrapper, merged into one archive so its ggml
# can't be mixed up with another ggml's archives of the same names (such as llama-go's)
#
# BUILD_TYPE=cublas builds the CUDA backend; CUDA_ARCHITECTURES picks the GPU architectures to
# compile for (e.g. 86 for an RTX 3090), as llama-go does

BUILD_TYPE ?=
CUDA_ARCHITECTURES ?=

BUILD_DIR := build
CMAKE_ARGS := -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DS2_BUILD_SHARED_LIBRARIES=ON

ifeq ($(BUILD_TYPE),cublas)
    CMAKE_ARGS += -DS2_CUDA=ON
    ifneq ($(CUDA_ARCHITECTURES),)
        CMAKE_ARGS += -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHITECTURES)
    endif
endif

CXXFLAGS := -O3 -std=c++17 -fPIC -I./s2.cpp/include -I./s2.cpp/ggml/include

.PHONY: all clean FORCE

all: libs2go.a

FORCE:

# CMake rebuilds whatever changed in s2.cpp
$(BUILD_DIR)/libs2_static.a: FORCE
	cmake -S s2.cpp -B $(BUILD_DIR) $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) --config Release --target s2_static --parallel

# The wrapper lives in csrc/ so cgo doesn't compile it again: the Go package only needs wrapper.h.
# The compiler writes wrapper.d listing every header wrapper.cpp includes, so changes to it or to
# s2.cpp's headers rebuild the wrapper
$(BUILD_DIR)/wrapper.o: csrc/wrapper.cpp wrapper.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c csrc/wrapper.cpp -o $@

-include $(BUILD_DIR)/wrapper.d

libs2go.a: $(BUILD_DIR)/libs2_static.a $(BUILD_DIR)/wrapper.o
	rm -f $@
	{ echo "CREATE $@"; \
	  for lib in $(BUILD_DIR)/libs2_static.a $$(find $(BUILD_DIR)/ggml -name 'libggml*.a' | sort); do \
	    echo "ADDLIB $$lib"; \
	  done; \
	  echo "ADDMOD $(BUILD_DIR)/wrapper.o"; echo "SAVE"; echo "END"; } | $(AR) -M
	ranlib $@

clean:
	rm -rf $(BUILD_DIR) libs2go.a
