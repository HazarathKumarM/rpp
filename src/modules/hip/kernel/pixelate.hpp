#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduction.hpp"

// template <typename T>
// __global__ void pixelate_pln_hip_tensor(T *srcPtr,
//                                         uint3 srcStridesNCH,
//                                         T *dstPtr,
//                                         uint3 dstStridesNCH,
//                                         int channelsDst,
//                                         uint blockSize,
//                                         RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * blockSize;
//     int id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y) * blockSize;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//       return;
//     }

//     if(id_x == 0)
//       printf("\n in pixelate kernel");

//     uint dstIdx = (id_z * srcStridesNCH.x) + (id_y * srcStridesNCH.z) + (id_x);

//     float sum = 0.0f;
//     int alignedLength = (blockSize / 8) * blockSize;
//     for (int j = 0; j < blockSize; j++)
//     {
//       for(int i = 0; i < alignedLength; i += 8)
//       {
//           printf("\n in vector loop");
//         uint wid_x = fmaxf(roiTensorPtrSrc[id_z].xywhROI.xy.x, fminf((id_x + i), roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
//         uint wid_y = fmaxf(roiTensorPtrSrc[id_z].xywhROI.xy.y, fminf((id_y + j), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));
//         uint window_idx = (id_z * srcStridesNCH.x) + (wid_y * srcStridesNCH.z) + (wid_x);

//         d_float8 window_data_f8;
//         rpp_hip_load8_and_unpack_to_float8(srcPtr + window_idx, &window_data_f8);

//         window_data_f8.f4[0] += window_data_f8.f4[1];
//         sum += (window_data_f8.f1[0] +  window_data_f8.f1[1] +  window_data_f8.f1[2] +  window_data_f8.f1[3]);
//       }
//       for(int i = alignedLength; i < blockSize; i++)
//       {
//         if(j == 0)
//           printf("\n in vector loop");
//         uint wid_x = fmaxf(roiTensorPtrSrc[id_z].xywhROI.xy.x, fminf((id_x + i), roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
//         uint wid_y = fmaxf(roiTensorPtrSrc[id_z].xywhROI.xy.y, fminf((id_y + j), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));
//         uint window_idx = (id_z * srcStridesNCH.x) + (wid_y * srcStridesNCH.z) + (wid_x);
//         sum += (float)*(srcPtr + window_idx);
//       }
//     }

//     float average = sum / (blockSize * blockSize);
//     d_float8 average_f8;
//     average_f8.f4[0] = (float4)average;
//     average_f8.f4[1] = (float4)average;

//     for (int j = 0; j < blockSize; j++)
//     {
//       for(int i = 0; i < alignedLength; i+=8){
//         uint wid_x = id_x + i;
//         uint wid_y = id_y + j;
//         uint window_idx = (id_z * srcStridesNCH.x) + (wid_y * srcStridesNCH.z) + (wid_x);
//         rpp_hip_pack_float8_and_store8(dstPtr + window_idx, &average_f8);
//       }
//       for(int i = alignedLength; i < blockSize; i++)
//       {
//         uint wid_x = id_x + i;
//         uint wid_y = id_y + j;
//         uint window_idx = (id_z * srcStridesNCH.x) + (wid_y * srcStridesNCH.z) + (wid_x);
//         *(dstPtr+window_idx) = (T)average;
//       }
//     }
// }

__device__ void compute_average(Rpp8u* block, float &avg, int blockSize)
{
    float sum  = 0;
    for (int i = 0; i < blockSize * blockSize; i++)
    {
        sum += block[i];
    }
    avg = sum / (blockSize * blockSize);
}

// Kernel to compute the pixelate effect
template <typename T>
__global__ void pixelate_pln_hip_tensor(T *srcPtr,
                                        uint3 srcStridesNCH,
                                        T *dstPtr,
                                        uint3 dstStridesNCH,
                                        int channelsDst,
                                        int blockSize,
                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y);
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    __shared__ float block_data[8][8];
    float *partialRSumRowPtr_smem = &block_data[hipThreadIdx_y][0];

    if (id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight || id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth)
    {
        return;
    }  

    // Load the block data
    block_data[hipThreadIdx_y][hipThreadIdx_x] = srcPtr[srcIdx];
    __syncthreads();

    for (int threadMax = 4; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialRSumRowPtr_smem[hipThreadIdx_x] += partialRSumRowPtr_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 4, increment = 32; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialRSumRowPtr_smem[0] += partialRSumRowPtr_smem[increment];
            __syncthreads();
        }
    }
    __syncthreads();
    
    rpp_hip_pixel_check_and_store(ceil(block_data[0][0] / 64), &dstPtr[dstIdx]);
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

    dim3 blockDim(8, 8, 1); // Adjust based on your GPU capabilities
    dim3 gridDim((globalThreads_x + blockDim.x - 1) / blockDim.x,
                 (globalThreads_y + blockDim.y - 1) / blockDim.y,
                 globalThreads_z);

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
            hipLaunchKernelGGL(pixelate_pln_hip_tensor<T>,
                               dim3(ceil((float)globalThreads_x/8), ceil((float)globalThreads_y/8), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(8, 8, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               kernelSize,
                               roiTensorPtrSrc);
    }
    return RPP_SUCCESS;
}