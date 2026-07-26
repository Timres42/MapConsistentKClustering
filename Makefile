CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall
CPPFLAGS ?= -Isrc/include

SRC_DIR := src
BUILD_DIR := build

all: $(BUILD_DIR)/propagation

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/propagation: \
	$(SRC_DIR)/propagation.cpp \
	$(SRC_DIR)/include/core.hpp \
	$(SRC_DIR)/include/heuristics.hpp \
	$(SRC_DIR)/include/knapsack.hpp \
	$(SRC_DIR)/include/form.hpp \
	$(SRC_DIR)/include/fid.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC_DIR)/propagation.cpp -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean