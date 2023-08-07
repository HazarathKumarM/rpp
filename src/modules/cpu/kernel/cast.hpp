/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_common.hpp"

template <typename T, typename U>
RppStatus cast_host_tensor(T *srcPtr,
                           RpptDescPtr srcDescPtr,
                           U *dstPtr,
                           RpptDescPtr dstDescPtr,
                           RppLayoutParams layoutParams,
                           rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        T *srcPtrImage;
        U *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        if(std::is_same<T, Rpp8u>::value && (std::is_same<U, Rpp16f>::value || std::is_same<U, Rpp32f>::value))
            std::transform(srcPtrImage, srcPtrImage + srcDescPtr->strides.nStride, dstPtrImage, [](T val) { return static_cast<U>(val) / 255.0; });
        else if(std::is_same<T, Rpp8u>::value && std::is_same<U, Rpp8s>::value)
            std::transform(srcPtrImage, srcPtrImage + srcDescPtr->strides.nStride, dstPtrImage,
                           [](T val) { return static_cast<U>(val) - 128; });
        else
            std::copy(srcPtrImage, srcPtrImage + srcDescPtr->strides.nStride, dstPtrImage);
    }
    return RPP_SUCCESS;
}
