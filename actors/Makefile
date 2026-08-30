CXX      ?= g++
TARGET   := test_actor
SRC      := test.cc

MFBO_TARGET := mfbo_demo
MFBO_SRC    := mfbo_demo.cc

# Installed CAF prefix (headers in $(CAF_PREFIX)/include, libs in $(CAF_PREFIX)/$(CAF_LIBDIR)).
# test.cc uses the CAF >=1.0 API (anon_mail, self->println, callable self->state()),
# which the system CAF at /usr/local/caf (0.18.6) does not have. The 1.1.0 build
# shipped under ~/labs/summa/.../install/caf was built elsewhere with a newer glibc
# and won't run here, so we rebuilt it locally (core+io only) into caf-hundsdorfer.
# Override on the command line to point elsewhere, e.g.:
#   make CAF_PREFIX=/usr/local/caf CAF_LIBDIR=lib
CAF_PREFIX ?= $(HOME)/labs/summa/Summa-Actors/utils/dependencies/install/caf-hundsdorfer
CAF_LIBDIR ?= lib

CXXFLAGS := -std=c++17 -O2 -pthread -Wall -Wextra
LDFLAGS  := -pthread
LDLIBS   := -lcaf_io -lcaf_core

ifneq ($(CAF_PREFIX),)
CXXFLAGS += -I$(CAF_PREFIX)/include
LDFLAGS  += -L$(CAF_PREFIX)/$(CAF_LIBDIR) -Wl,-rpath,$(CAF_PREFIX)/$(CAF_LIBDIR)
endif

.PHONY: all run clean mfbo run-mfbo

all: $(TARGET) $(MFBO_TARGET)

$(TARGET): $(SRC) config.h
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

# mfbo_demo has no CAF dependency (plain C++17 + libm), so it builds without
# CAF_PREFIX/LDLIBS.
mfbo: $(MFBO_TARGET)

$(MFBO_TARGET): $(MFBO_SRC) mfbo.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra $(MFBO_SRC) -o $(MFBO_TARGET)

run-mfbo: $(MFBO_TARGET)
	./$(MFBO_TARGET)

clean:
	rm -f $(TARGET) $(MFBO_TARGET)
