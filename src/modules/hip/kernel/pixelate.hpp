#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduction.hpp"

// Kernel to compute the pixelate effect
template <typename T>
__global__ void pixelate_pln_hip_tensor(T *srcPtr,
                                        uint3 srcStridesNCH,
                                        T *dstPtr,
                                        uint3 dstStridesNCH,
                                        int channelsDst,
                                        int blockSize,
                                        uint2 tileSize,
                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y);
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    __shared__ float block_data[SMEM_LENGTH_X][SMEM_LENGTH_X];
    float *blockRow = &block_data[hipThreadIdx_y][0];
    if (id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight || id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth)
    {
        return;
    }

    // Load the block data
    if ((hipThreadIdx_x % blockSize) == 0)
    {
        for(int i = 0; i < tileSize.x; i++)
            block_data[hipThreadIdx_y][hipThreadIdx_x + i] = srcPtr[srcIdx + i];
        __syncthreads();
    }

    for (int threadMax = tileSize.x / 2; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            blockRow[hipThreadIdx_x] += blockRow[hipThreadIdx_x + threadMax];
    }
    __syncthreads();

    if ((hipThreadIdx_x % blockSize) == 0)
    {
        for (int threadMax =  tileSize.x / 2, increment = tileSize.x * 4; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                blockRow[0] += blockRow[increment];
            __syncthreads();
        }
    }
    // __syncthreads();
    float average = ceil((block_data[0][0]) / 64);
    d_float8 average_f8;
    average_f8.f4[0] = (float4)average;
    average_f8.f4[1] = (float4)average;
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &average_f8);
}

template <typename T>
RppStatus hip_exec_pixelate_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u kernelSize,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = dstDescPtr->strides.hStride;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    kernelSize = (kernelSize / 8) * 8;

    uint2 tileSize;
    tileSize.x = tileSize.y = kernelSize;

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
            hipLaunchKernelGGL(pixelate_pln_hip_tensor<T>,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(tileSize.x / 8, tileSize.y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               kernelSize,
                               tileSize,
                               roiTensorPtrSrc);
    }
    return RPP_SUCCESS;
}