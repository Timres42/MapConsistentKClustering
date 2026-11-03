CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall
CPPFLAGS ?= -Isrc/include

SRC_DIR := src
BUILD_DIR := build

# ---- Gurobi (optional; needed for --opt and the baseline --heuristic optimal) --
# Enabled automatically when GUROBI_HOME points to a Gurobi installation, e.g.
#   export GUROBI_HOME=/Library/gurobi1300/macos_universal2   (macOS)
#   export GUROBI_HOME=/opt/gurobi1300/linux64                 (Linux)
# or per call: make GUROBI_HOME=...  Disable with: make USE_GUROBI=0
# Uses Gurobi's C API (libgurobiXYZ), not the C++ wrapper library.
GUROBI_HOME ?=
USE_GUROBI ?= $(if $(GUROBI_HOME),1,0)

ifeq ($(USE_GUROBI),1)
  ifeq ($(GUROBI_HOME),)
    $(error USE_GUROBI=1 but GUROBI_HOME is not set)
  endif
  # Library name, e.g. gurobi130 for Gurobi 13.0 (auto-detected; override with GUROBI_LIB=gurobi120).
  GUROBI_LIB ?= $(patsubst lib%,%,$(basename $(notdir $(firstword \
                  $(wildcard $(GUROBI_HOME)/lib/libgurobi[0-9][0-9][0-9].so \
                             $(GUROBI_HOME)/lib/libgurobi[0-9][0-9][0-9].dylib)))))
  ifeq ($(GUROBI_LIB),)
    $(error No libgurobiXYZ.so/.dylib found in $(GUROBI_HOME)/lib - set GUROBI_LIB=gurobiXYZ)
  endif
  GUROBI_CPPFLAGS := -DUSE_GUROBI -I$(GUROBI_HOME)/include
  GUROBI_LDLIBS := -L$(GUROBI_HOME)/lib -l$(GUROBI_LIB) -Wl,-rpath,$(GUROBI_HOME)/lib
endif

# Records the Gurobi configuration so that switching it on/off triggers a rebuild.
BUILD_CONFIG := $(BUILD_DIR)/.build_config
BUILD_CONFIG_LINE := $(GUROBI_CPPFLAGS) $(GUROBI_LDLIBS)

HEADERS := \
	$(SRC_DIR)/include/core.hpp \
	$(SRC_DIR)/include/heuristics.hpp \
	$(SRC_DIR)/include/knapsack.hpp \
	$(SRC_DIR)/include/form.hpp \
	$(SRC_DIR)/include/fid.hpp \
	$(SRC_DIR)/include/opt.hpp

all: $(BUILD_DIR)/propagation

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_CONFIG): FORCE | $(BUILD_DIR)
	@echo '$(BUILD_CONFIG_LINE)' | cmp -s - $@ || echo '$(BUILD_CONFIG_LINE)' > $@

$(BUILD_DIR)/propagation: $(SRC_DIR)/propagation.cpp $(HEADERS) $(BUILD_CONFIG) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(GUROBI_CPPFLAGS) $(CXXFLAGS) $(SRC_DIR)/propagation.cpp -o $@ $(GUROBI_LDLIBS)
	@if [ "$(USE_GUROBI)" = "1" ]; then echo "Built with Gurobi ($(GUROBI_LIB)): --opt and --heuristic optimal available"; \
	 else echo "Built without Gurobi: --opt and --heuristic optimal unavailable (set GUROBI_HOME to enable)"; fi

# Brute-force check of the exact opt algorithm on small random instances (needs Gurobi).
$(BUILD_DIR)/test_opt: tests/test_opt.cpp $(HEADERS) $(BUILD_CONFIG) | $(BUILD_DIR)
	@if [ "$(USE_GUROBI)" != "1" ]; then echo "test-opt needs Gurobi (set GUROBI_HOME)"; exit 1; fi
	$(CXX) $(CPPFLAGS) $(GUROBI_CPPFLAGS) $(CXXFLAGS) tests/test_opt.cpp -o $@ $(GUROBI_LDLIBS)

test-opt: $(BUILD_DIR)/test_opt
	./$(BUILD_DIR)/test_opt

clean:
	rm -rf $(BUILD_DIR)

FORCE:

.PHONY: all clean test-opt FORCE
