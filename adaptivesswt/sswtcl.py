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
            /*
            if (wab[idx] >= borderFreqs[r_gid] && wab[idx] < borderFreqs[r_gid+1]){
                sst[idx] = tr_matr[idx];
            }*/
            for(int w=0; w<hg; w++){
                if (wab[c_gid+w*wd] >= borderFreqs[r_gid] && wab[c_gid+w*wd] < borderFreqs[r_gid+1]){
                    sst[idx] = cdouble_add(sst[idx], cdouble_mulr(tr_matr[c_gid+w*wd] , aScale[w] / deltaFreqs[r_gid]));
                }
            }
        }
        """)

mf = cl.mem_flags

try:
    _freq_agregate_prg.build()
except Exception:
    print("Error:")
    print(_freq_agregate_prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
    raise
freq_agregate_knl = _freq_agregate_prg.agregate  # Use this Kernel object for repeated calls

def _freq_agregate_cl(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                    aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                    sst: np.ndarray):
    deltaFreqs_dev = cl_array.to_device(queue, deltaFreqs)
    borderFreqs_dev = cl_array.to_device(queue, borderFreqs)
    aScale_dev = cl_array.to_device(queue, aScale)
    wab_dev = cl_array.to_device(queue, wab)
    tr_matr_dev = cl_array.to_device(queue, tr_matr)
    sst_dev = cl_array.to_device(queue, sst)

    width_dev = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(sst_dev.shape[1])
        )
    height_dev = cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(sst_dev.shape[0])
        )
    freq_agregate_knl(queue, sst.shape, None, deltaFreqs_dev.data, borderFreqs_dev.data,
        aScale_dev.data, wab_dev.data, tr_matr_dev.data, sst_dev.data,
        width_dev, height_dev)

    queue.finish()
    sst = sst_dev.get()
    return sst
