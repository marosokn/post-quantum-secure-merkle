NVCC     ?= nvcc
ARCH     ?= sm_86
CXXFLAGS := -O3 -std=c++17 -Iinclude -Isrc
NVFLAGS  := -arch=$(ARCH) $(CXXFLAGS) --use_fast_math -Xcompiler -fPIC

LIB := libpqmerkle.so
SRC := $(wildcard src/*.cu)
OBJ := $(SRC:src/%.cu=build/%.o)

.PHONY: all clean
all: $(LIB)

build:
	mkdir -p build

build/%.o: src/%.cu | build
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(LIB): $(OBJ)
	$(NVCC) $(NVFLAGS) -shared $^ -o $@

clean:
	rm -rf build $(LIB)
