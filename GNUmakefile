# GNUmakefile for PowerSpectrumExtractor
# Builds a standalone AMReX program that reads plot files and computes
# the power spectrum of a named component (default: "R").
#
# Usage:
#   make                   (builds with default settings)
#   make DEBUG=TRUE        (adds debug flags)
#   make USE_MPI=FALSE     (serial build)
#
# The AMREX_HOME variable must point to an AMReX source tree.
# It defaults to a sibling 'amrex' directory relative to GRTeclyn-workspace,
# but can be overridden on the command line:
#   make AMREX_HOME=/path/to/amrex

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AMREX_HOME ?= $(realpath ../../Codes/amrex)

# ---------------------------------------------------------------------------
# Build settings  (match the GRTeclyn defaults)
# ---------------------------------------------------------------------------
DEBUG     = FALSE
PRECISION = DOUBLE

USE_MPI   = TRUE
USE_OMP   = TRUE
USE_FFT   = TRUE

USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

USE_HDF5  = FALSE

TINY_PROFILE     = FALSE
BL_NO_FORT       = TRUE
AMREX_NO_PROBINIT = TRUE

COMP    = intel-llvm  # override on command line if using gcc: make COMP=gnu
CXXSTD  = c++17
DIM     = 3

# Shared build-object directory (keeps the source tree clean)
TMP_BUILD_DIR ?= $(realpath .)/tmp_build_dir

# Executable name
EBASE = PowerSpectrumExtractor

# ---------------------------------------------------------------------------
# AMReX build infrastructure
# ---------------------------------------------------------------------------
include $(AMREX_HOME)/Tools/GNUMake/Make.defs

# Source files for this tool
include ./Make.package

# AMReX packages needed: Base (MultiFab, PlotFileUtil, ParmParse …) + FFT
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/FFT/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
