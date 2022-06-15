# Config for a basic DM only simulation using TreePM gravity
# Basic code operation

    PERIODIC  # periodic box for cosmology
    NTYPES=2  # particle types (type 0 always reserved for gas?) type 1 - DM
    RANDOMIZE_DOMAINCENTER  # random shifts to reduce force error correlations (should always use)
    LEAN  # Aggressive memory saving for DM only sims

# Gravity options

    SELFGRAVITY
    HIERARCHICAL_GRAVITY

# TreePM options

    PMGRID=512  # changed from 384 (power of 2 better)
    TREEPM_NOTIMESPLIT  # time-integrates long-range and short-range gravity on same step
    ASMTH=1.5  # default: 1.25

# Softening types

    NSOFTCLASSES=1

# Floating point accuracy

    POSITIONS_IN_32BIT
    DOUBLEPRECISION=2  # mixed internal precision
    IDS_32BIT

# Group finding - better to do as postprocessing (probably?)

    FOF
    FOF_LINKLENGTH=0.2
    FOF_GROUP_MIN_LEN=32
    SUBFIND

# Miscellaneous code options

    POWERSPEC_ON_OUTPUT

# IC generation via N-GenIC

    NGENIC=1024  # FFT grid size for ICs
    NGENIC_2LPT  # ICs via 2nd order LPT
    CREATE_GRID

# Debug

    ENABLE_HEALTHTEST  # MPI info and performance at start-up
