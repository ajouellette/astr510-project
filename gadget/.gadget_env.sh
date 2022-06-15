conda deactivate

# GADGET libs
export GSL_HOME=${HOME}/lib/gsl/2.7

export FFTW_HOME=${HOME}/lib/fftw/3.3.9

#export HDF5_HOME=${HOME}/lib/hdf5/1.10.1
export HDF5_HOME=${HOME}/lib/hdf5/1.8

#export MPICH_HOME=${HOME}/lib/mpich/3.0.2
export OPENMPI_HOME=${HOME}/lib/openmpi/4.1.2

#export HWLOC_HOME=${HOME}/lib/hwloc/2.2.0

export LD_LIBRARY_PATH=${GSL_HOME}/lib:${FFTW_HOME}/lib:${HDF5_HOME}/lib:${OPENMPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${GSL_HOME}/bin:${FFTW_HOME}/bin:${HDF5_HOME}/bin:${OPENMPI_HOME}/bin:$PATH

# UCX settings (try to avoid MPI instability)
export UCX_TLS=self,sm,ud
export UCX_UD_MLX5_RX_QUEUE_LEN=16384
