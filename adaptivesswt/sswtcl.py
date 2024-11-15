import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
except ImportError as e:
    logger.error('PyOpenCL is not installed. Please install it with "pip install pyopencl"')
    raise e

try:
    platforms = cl.get_platforms()
except cl.LogicError as e:
    logger.error('No OpenCL platforms found!')
    raise e

ctx = cl.Context(
    dev_type=cl.device_type.ALL,
    properties=[(cl.context_properties.PLATFORM, platforms[0])])

queue = cl.CommandQueue(ctx)

#################################
# Frequency aggregation program #
#################################
_freq_agregate_prg = cl.Program(ctx,
        """
        #define PYOPENCL_DEFINE_CDOUBLE
        #include <pyopencl-complex.h>
        __kernel void agregate(
            __global const double *deltaFreqs, __global const double *borderFreqs,
            __global const double *aScale,  __global const double *wab, __global const cdouble_t *tr_matr,
            __global cdouble_t *sst, __global int *width, __global int *height)
        {
            int wd = *width;
            int hg = *height;
            int r_gid = get_global_id(0);
            int c_gid = get_global_id(1);
            int idx = c_gid + wd*r_gid;
            for(int w=0; w<hg; w++){
                if (wab[c_gid+w*wd] >= borderFreqs[r_gid] && wab[c_gid+w*wd] < borderFreqs[r_gid+1]){
                    sst[idx] = cdouble_add(sst[idx], cdouble_mulr(tr_matr[c_gid+w*wd] , aScale[w] / deltaFreqs[r_gid]));
                }
            }
        }
        """)

################################
# Frequency extraction program #
################################
_freq_extract_prg = cl.Program(ctx,
        """
        #define PYOPENCL_DEFINE_CDOUBLE
        #include <pyopencl-complex.h>
        __kernel void extract(
            __global const double *deltaFreqs, __global const double *borderFreqs,
            __global const double *aScale,  __global const double *wab, __global const cdouble_t *tr_matr,
            __global cdouble_t *sst, __global int *width, __global int *height)
        {
            int wd = *width;
            int hg = *height;
            int r_gid = get_global_id(0);
            int c_gid = get_global_id(1);
            int idx = c_gid + wd*r_gid;
            if (wab[idx] >= borderFreqs[r_gid] && wab[idx] < borderFreqs[r_gid+1]){
                sst[idx] = tr_matr[idx];
            }
        }
        """)

############################
# Time aggregation program #
############################
_time_agregate_prg = cl.Program(ctx,
        """
        #define PYOPENCL_DEFINE_CDOUBLE
        #include <pyopencl-complex.h>
        __kernel void t_agregate(
            __global const double *time, __global const double *tab, __global const cdouble_t *tr_matr,
            __global cdouble_t *tsst, __global int *width, __global int *height)
        {
            int wd = *width;
            int hg = *height;
            int r_gid = get_global_id(0);
            int c_gid = get_global_id(1);
            int idx = c_gid + wd*r_gid;
            int row_start = r_gid*wd;
            for(int t=0; t<wd; t++){
                if (tab[row_start+t] >= time[c_gid] && tab[row_start+t] < time[c_gid+1]){
                    tsst[idx] = cdouble_add(tsst[idx], tr_matr[row_start+t]);
                }
            }
        }
        """)


mf = cl.mem_flags


################################
# Frequency aggregation kernel #
################################
try:
    _freq_agregate_prg.build()
except Exception:
    logger.error('Error!: %s', _freq_agregate_prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
    raise

freq_agregate_knl = _freq_agregate_prg.agregate  # Use this Kernel object for repeated calls

def _freq_agregate_cl(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                      aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                      sst: np.ndarray) -> np.ndarray:
    deltaFreqs_dev = cl_array.to_device(queue, deltaFreqs)
    borderFreqs_dev = cl_array.to_device(queue, borderFreqs)
    aScale_dev = cl_array.to_device(queue, aScale)
    wab_dev = cl_array.to_device(queue, wab)
    tr_matr_dev = cl_array.to_device(queue, tr_matr)
    sst_dev = cl_array.to_device(queue, sst)

    width_dev = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(sst_dev.shape[1])  # type: ignore  # since shape is a tuple
        )
    height_dev = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(sst_dev.shape[0])  # type: ignore  # since shape is a tuple
        )
    freq_agregate_knl(queue, sst.shape, None, deltaFreqs_dev.data, borderFreqs_dev.data,
        aScale_dev.data, wab_dev.data, tr_matr_dev.data, sst_dev.data,
        width_dev, height_dev)

    queue.finish()
    sst = sst_dev.get()
    return sst


###############################
# Frequency extraction kernel #
###############################
try:
    _freq_extract_prg.build()
except Exception:
    logger.error('Error!: %s', _freq_extract_prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
    raise
freq_extract_knl = _freq_extract_prg.extract  # Use this Kernel object for repeated calls

def _freq_extract_cl(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                     aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                     set_tr: np.ndarray) -> np.ndarray:
    deltaFreqs_dev = cl_array.to_device(queue, deltaFreqs)
    borderFreqs_dev = cl_array.to_device(queue, borderFreqs)
    aScale_dev = cl_array.to_device(queue, aScale)
    wab_dev = cl_array.to_device(queue, wab)
    tr_matr_dev = cl_array.to_device(queue, tr_matr)
    set_dev = cl_array.to_device(queue, set_tr)

    width_dev = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(set_dev.shape[1])  # type: ignore  # since shape is a tuple
        )
    height_dev = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(set_dev.shape[0])  # type: ignore  # since shape is a tuple
        )
    freq_extract_knl(queue, set_tr.shape, None, deltaFreqs_dev.data, borderFreqs_dev.data,
        aScale_dev.data, wab_dev.data, tr_matr_dev.data, set_dev.data,
        width_dev, height_dev)

    queue.finish()
    set = set_dev.get()
    return set


###########################
# Time aggregation kernel #
###########################
try:
    _time_agregate_prg.build()
except Exception:
    logger.error('Error!: %s', _time_agregate_prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
    raise

time_agregate_knl = _time_agregate_prg.t_agregate  # Use this Kernel object for repeated calls

def _time_agregate_cl(time: np.ndarray, tab: np.ndarray, tr_matr: np.ndarray,
                      tsst: np.ndarray) -> np.ndarray:
    time_dev = cl_array.to_device(queue, time)
    tab_dev = cl_array.to_device(queue, tab)
    tr_matr_dev = cl_array.to_device(queue, tr_matr)
    tsst_dev = cl_array.to_device(queue, tsst)

    width_dev = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(tsst_dev.shape[1])  # type: ignore  # since shape is a tuple
        )
    height_dev = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(tsst_dev.shape[0])  # type: ignore  # since shape is a tuple
        )
    time_agregate_knl(queue, tsst.shape, None, time_dev.data,
        tab_dev.data, tr_matr_dev.data, tsst_dev.data,
        width_dev, height_dev)

    queue.finish()
    tsst = tsst_dev.get()
    return tsst
