#include <rpp.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include "optical_flow_vectors.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

#define FARNEBACK_FRAME_WIDTH 960                       // Farneback algorithm frame width
#define FARNEBACK_FRAME_HEIGHT 540                      // Farneback algorithm frame height
#define FARNEBACK_OUTPUT_FRAME_SIZE 518400u             // 960 * 540
#define FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE 1036800u   // 960 * 540 * 2
#define FARNEBACK_OUTPUT_RGB_SIZE 1555200u              // 960 * 540 * 3
#define HUE_CONVERSION_FACTOR 0.0019607843f             // ((1 / 360.0) * (180 / 255.0))

const RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
const RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::BGRtype;
const RpptRoiType roiType = RpptRoiType::XYWH;
const RpptAngleType angleType = RpptAngleType::DEGREES;

void rpp_tensor_initialize_descriptor(RpptDescPtr descPtr,
                                      RpptLayout layout,
                                      RpptDataType dataType,
                                      Rpp32u offsetInBytes,
                                      RppSize_t numDims,
                                      Rpp32u n,
                                      Rpp32u h,
                                      Rpp32u w,
                                      Rpp32u c,
                                      Rpp32u nStride,
                                      Rpp32u hStride,
                                      Rpp32u wStride,
                                      Rpp32u cStride)
{
    descPtr->layout = layout;
    descPtr->dataType = dataType;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->numDims = numDims;
    descPtr->n = n;
    descPtr->h = h;
    descPtr->w = w;
    descPtr->c = c;
    descPtr->strides.nStride = nStride;
    descPtr->strides.hStride = hStride;
    descPtr->strides.wStride = wStride;
    descPtr->strides.cStride = cStride;
}

void rpp_tensor_initialize_descriptor_generic(RpptGenericDescPtr descPtr,
                                              RpptLayout layout,
                                              RpptDataType dataType,
                                              Rpp32u offsetInBytes,
                                              RppSize_t numDims,
                                              Rpp32u dim0,
                                              Rpp32u dim1,
                                              Rpp32u dim2,
                                              Rpp32u dim3,
                                              Rpp32u stride0,
                                              Rpp32u stride1,
                                              Rpp32u stride2,
                                              Rpp32u stride3)
{
    descPtr->layout = layout;
    descPtr->dataType = dataType;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->numDims = numDims;
    descPtr->dims[0] = dim0;
    descPtr->dims[1] = dim1;
    descPtr->dims[2] = dim2;
    descPtr->dims[3] = dim3;
    descPtr->strides[0] = stride0;
    descPtr->strides[1] = stride1;
    descPtr->strides[2] = stride2;
    descPtr->strides[3] = stride3;
}

void rpp_tensor_initialize_roi_xywh(RpptROIPtr roiPtr, Rpp32u x, Rpp32u y, Rpp32u roiWidth, Rpp32u roiHeight)
{
    roiPtr->xywhROI.xy.x = x;
    roiPtr->xywhROI.xy.y = y;
    roiPtr->xywhROI.roiWidth = roiWidth;
    roiPtr->xywhROI.roiHeight = roiHeight;
}

void rpp_tensor_create_strided(Rpp8u *frame_u8, Rpp8u *srcRGB, RpptDescPtr srcDescPtrRGB)
{
    Rpp8u *frameTemp_u8 = frame_u8;
    Rpp8u *srcRGBTemp = srcRGB + srcDescPtrRGB->offsetInBytes;
    Rpp32u elementsInRow = srcDescPtrRGB->c * srcDescPtrRGB->w;
    for (int i = 0; i < srcDescPtrRGB->h; i++)
    {
        memcpy(srcRGBTemp, frameTemp_u8, elementsInRow);
        srcRGBTemp += srcDescPtrRGB->strides.hStride;
        frameTemp_u8 += elementsInRow;
    }
}

template <typename T>
void rpp_tensor_print(string tensorName, T *tensor, RpptDescPtr desc)
{
    std::cout << "\n" << tensorName << ":\n";
    T *tensorTemp;
    tensorTemp = tensor + desc->offsetInBytes;
    for (int n = 0; n < desc->n; n++)
    {
        std::cout << "[ ";
        for (int h = 0; h < desc->h; h++)
        {
            for (int w = 0; w < desc->w; w++)
            {
                for (int c = 0; c < desc->c; c++)
                {
                    std::cout << (Rpp32f)*tensorTemp << ", ";
                    tensorTemp++;
                }
            }
            std::cout << ";\n";
        }
        std::cout << " ]\n\n";
    }
}

template <typename T>
void rpp_tensor_write_to_file(string tensorName, T *tensor, RpptDescPtr desc)
{
    ofstream outputFile (tensorName + ".csv");

    outputFile << "\n" << tensorName << ":\n";
    T *tensorTemp;
    tensorTemp = tensor + desc->offsetInBytes;
    for (int n = 0; n < desc->n; n++)
    {
        outputFile << "[ ";
        for (int h = 0; h < desc->h; h++)
        {
            for (int w = 0; w < desc->w; w++)
            {
                for (int c = 0; c < desc->c; c++)
                {
                    outputFile << (Rpp32f)*tensorTemp << ", ";
                    tensorTemp++;
                }
            }
            outputFile << ";\n";
        }
        outputFile << " ]\n\n";
    }
}

void rpp_tensor_write_to_images(string tensorName, Rpp8u *tensor, RpptDescPtr desc, RpptImagePatch *imgSizes)
{
    Rpp8u outputImage_u8[desc->strides.nStride];
    Rpp8u *tensorTemp, *outputImageTemp_u8;
    tensorTemp = tensor + desc->offsetInBytes;
    outputImageTemp_u8 = outputImage_u8;

    for (int n = 0; n < desc->n; n++)
    {
        Rpp32u elementsInRow = desc->c * imgSizes[n].width;
        for (int i = 0; i < imgSizes[n].height; i++)
        {
            memcpy(outputImageTemp_u8, tensorTemp, elementsInRow);
            tensorTemp += desc->strides.hStride;
            outputImageTemp_u8 += elementsInRow;
        }

        cv::Mat outputImage_mat;
        outputImage_mat = (desc->c == 1) ? Mat(imgSizes[n].height, imgSizes[n].width, CV_8UC1, outputImage_u8) : Mat(imgSizes[n].height, imgSizes[n].width, CV_8UC3, outputImage_u8);
        imwrite(tensorName + "_" + std::to_string(n) + ".jpg", outputImage_mat);
    }
}

void rpp_optical_flow_hip(string inputVideoFileName)
{
    // initialize map to track time for every stage at each iteration
    unordered_map<string, vector<double>> timers;

    // initialize video capture with opencv video
    VideoCapture capture(inputVideoFileName);
    if (!capture.isOpened())
    {
        // error in opening the video file
        cout << "\nUnable to open file!";
        return;
    }

    // get video properties
    double fps = capture.get(CAP_PROP_FPS);                     // input video fps
    int numOfFrames = int(capture.get(CAP_PROP_FRAME_COUNT));   // input video number of frames
    int frameWidth = int(capture.get(CAP_PROP_FRAME_WIDTH));    // input video frame width
    int frameHeight = int(capture.get(CAP_PROP_FRAME_HEIGHT));  // input video frame height
    int bitRate = int(capture.get(CAP_PROP_BITRATE));           // input video bitrate

    // declare rpp tensor descriptors
    RpptDesc srcDescRGB, srcResizedDescRGB, src1Desc, src2Desc;
    RpptDescPtr srcDescPtrRGB, srcResizedDescPtrRGB, src1DescPtr, src2DescPtr;
    RpptGenericDesc motionVectorPlnDesc, motionVectorPkdDesc, motionVectorCompPlnDesc;
    RpptGenericDescPtr motionVectorPlnDescPtr, motionVectorPkdDescPtr, motionVectorCompPlnDescPtr;

    // initialize rpp tensor descriptor for srcRGB frames
    srcDescPtrRGB = &srcDescRGB;
    rpp_tensor_initialize_descriptor(srcDescPtrRGB, RpptLayout::NHWC, RpptDataType::U8, 0, 4, 1, frameHeight, frameWidth, 3, frameHeight * frameWidth * 3, frameWidth * 3, 3, 1);

    // initialize rpp tensor descriptor for srcResizedRGB frames (same descriptor as srcRGB except farneback width/height and stride changes)
    srcResizedDescPtrRGB = &srcResizedDescRGB;
    rpp_tensor_initialize_descriptor(srcResizedDescPtrRGB, RpptLayout::NHWC, RpptDataType::U8, 0, 4, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, 3, FARNEBACK_OUTPUT_RGB_SIZE, FARNEBACK_FRAME_WIDTH * 3, 3, 1);

    // initialize rpp tensor descriptors for src greyscale frames
    src1DescPtr = &src1Desc;
    rpp_tensor_initialize_descriptor(src1DescPtr, RpptLayout::NCHW, RpptDataType::U8, 0, 4, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, 1, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_FRAME_WIDTH, 1, FARNEBACK_OUTPUT_FRAME_SIZE);
    src2DescPtr = &src2Desc;
    src2Desc = src1Desc;

    // set rpp tensor buffer sizes in bytes for srcRGB, srcResizedRGB, src1 and src2
    unsigned long long sizeInBytesSrcRGB, sizeInBytesSrcResizedRGB, sizeInBytesSrc1, sizeInBytesSrc2;
    sizeInBytesSrcRGB = (srcDescPtrRGB->n * srcDescPtrRGB->strides.nStride) + srcDescPtrRGB->offsetInBytes;
    sizeInBytesSrcResizedRGB = (srcResizedDescPtrRGB->n * srcResizedDescPtrRGB->strides.nStride) + srcResizedDescPtrRGB->offsetInBytes;
    sizeInBytesSrc1 = (src1DescPtr->n * src1DescPtr->strides.nStride) + src1DescPtr->offsetInBytes;
    sizeInBytesSrc2 = (src2DescPtr->n * src2DescPtr->strides.nStride) + src2DescPtr->offsetInBytes;

    // allocate rpp 8u host and hip buffers for srcRGB, srcResizedRGB, src1 and src2
    Rpp8u *srcRGB = (Rpp8u *)calloc(sizeInBytesSrcRGB, 1);
    Rpp8u *srcResizedRGB = (Rpp8u *)calloc(sizeInBytesSrcResizedRGB, 1);
    Rpp8u *src1 = (Rpp8u *)calloc(sizeInBytesSrc1, 1);
    Rpp8u *src2 = (Rpp8u *)calloc(sizeInBytesSrc2, 1);
    Rpp8u *d_srcRGB, *d_srcResizedRGB, *d_src1, *d_src2;
    hipMalloc(&d_srcRGB, sizeInBytesSrcRGB);
    hipMalloc(&d_srcResizedRGB, sizeInBytesSrcResizedRGB);
    hipMalloc(&d_src1, sizeInBytesSrc1);
    hipMalloc(&d_src2, sizeInBytesSrc2);

    // allocate and initialize rpp roi and imagePatch buffers for resize ops on pinned memory
    RpptROI *roiTensorPtrSrcRGB, *roiTensorPtrSrcResized;
    RpptImagePatch *src1ImgSizes;
    hipHostMalloc(&roiTensorPtrSrcRGB, srcDescPtrRGB->n * sizeof(RpptROI));
    hipHostMalloc(&roiTensorPtrSrcResized, srcDescPtrRGB->n * sizeof(RpptROI));
    hipHostMalloc(&src1ImgSizes, srcDescPtrRGB->n * sizeof(RpptImagePatch));
    rpp_tensor_initialize_roi_xywh(roiTensorPtrSrcRGB, 0, 0, srcDescPtrRGB->w, srcDescPtrRGB->h);
    rpp_tensor_initialize_roi_xywh(roiTensorPtrSrcResized, 0, 0, FARNEBACK_FRAME_WIDTH, FARNEBACK_FRAME_HEIGHT);
    src1ImgSizes[0].width = src1DescPtr->w;
    src1ImgSizes[0].height = src1DescPtr->h;

    // read the first frame
    cv::Mat frame, previousFrame;
    capture >> frame;

    // copy frame into rpp hip buffer
    // rpp_tensor_create_strided(frame.data, srcRGB, srcDescPtrRGB);    // not required since farneback frame size is already multiple of 8
    hipMemcpy(d_srcRGB, frame.data, sizeInBytesSrcRGB, hipMemcpyHostToDevice);

    // create rpp handle for hip with stream and batch size
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, srcDescPtrRGB->n);

    // -------------------- stage output dump check --------------------
    // rpp_tensor_write_to_file("srcRGB", srcRGB, srcDescPtrRGB);
    // RpptImagePatch srcRGBImgSizes;
    // srcRGBImgSizes.height = srcDescPtrRGB->h;
    // srcRGBImgSizes.width = srcDescPtrRGB->w;
    // rpp_tensor_write_to_images("srcRGB", srcRGB, srcDescPtrRGB, &srcRGBImgSizes);

    // resize frame
    rppt_resize_gpu(d_srcRGB, srcDescPtrRGB, d_srcResizedRGB, srcResizedDescPtrRGB, src1ImgSizes, interpolationType, roiTensorPtrSrcRGB, roiType, handle);
    hipDeviceSynchronize();

    // -------------------- stage output dump check --------------------
    // hipMemcpy(srcResizedRGB, d_srcResizedRGB, sizeInBytesSrcResizedRGB, hipMemcpyDeviceToHost);
    // rpp_tensor_write_to_file("srcResizedRGB", srcResizedRGB, srcResizedDescPtrRGB);
    // rpp_tensor_write_to_images("srcResizedRGB", srcResizedRGB, srcResizedDescPtrRGB, src1ImgSizes);

    // convert to gray
    rppt_color_to_greyscale_gpu(d_srcResizedRGB, srcResizedDescPtrRGB, d_src1, src1DescPtr, srcSubpixelLayout, handle);
    hipDeviceSynchronize();

    // -------------------- stage output dump check --------------------
    // hipMemcpy(src1, d_src1, sizeInBytesSrc1, hipMemcpyDeviceToHost);
    // rpp_tensor_write_to_file("src1", src1, src1DescPtr);
    // rpp_tensor_write_to_images("src1", src1, src1DescPtr, src1ImgSizes);

    // initialize rpp generic tensor descriptor for packed motion vector
    motionVectorPkdDescPtr = &motionVectorPkdDesc;
    rpp_tensor_initialize_descriptor_generic(motionVectorPkdDescPtr, RpptLayout::NHWC, RpptDataType::F32, 0, 4, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, 2, FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE, FARNEBACK_FRAME_WIDTH * 2, 2, 1);

    // initialize rpp generic tensor descriptor for planar motion vector
    motionVectorPlnDescPtr = &motionVectorPlnDesc;
    rpp_tensor_initialize_descriptor_generic(motionVectorPlnDescPtr, RpptLayout::NCHW, RpptDataType::F32, 0, 4, 1, 2, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, FARNEBACK_OUTPUT_FRAME_SIZE * 4, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_FRAME_WIDTH, 1);

    // initialize rpp generic tensor descriptor to extract one component of motion vector
    motionVectorCompPlnDescPtr = &motionVectorCompPlnDesc;
    rpp_tensor_initialize_descriptor_generic(motionVectorCompPlnDescPtr, RpptLayout::NCHW, RpptDataType::F32, 0, 4, 1, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_FRAME_WIDTH, 1);

    // allocate hip motion vector buffers
    Rpp32f *d_motionVectorsCartesianF32, *d_motionVectorsPolarF32, *d_motionVectorsPolarF32Comp1, *d_motionVectorsPolarF32Comp2, *d_motionVectorsPolarF32Comp3;
    hipMalloc(&d_motionVectorsCartesianF32, FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE * sizeof(Rpp32f));
    hipMalloc(&d_motionVectorsPolarF32, FARNEBACK_OUTPUT_FRAME_SIZE * 4 * sizeof(Rpp32f));
    d_motionVectorsPolarF32Comp1 = d_motionVectorsPolarF32 + FARNEBACK_OUTPUT_FRAME_SIZE;
    d_motionVectorsPolarF32Comp2 = d_motionVectorsPolarF32Comp1 + FARNEBACK_OUTPUT_FRAME_SIZE;
    d_motionVectorsPolarF32Comp3 = d_motionVectorsPolarF32Comp2 + FARNEBACK_OUTPUT_FRAME_SIZE;
    hipMemcpy(d_motionVectorsCartesianF32, motionVectorsCartesian, FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE * sizeof(Rpp32f), hipMemcpyHostToDevice);

    // preinitialize saturation channel portion of the buffer for HSV and reuse on every iteration in post-processing
    Rpp32f saturationChannel[FARNEBACK_OUTPUT_FRAME_SIZE];
    std::fill(&saturationChannel[0], &saturationChannel[FARNEBACK_OUTPUT_FRAME_SIZE - 1], 1.0f);
    hipMemcpy(d_motionVectorsPolarF32Comp2, saturationChannel, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyHostToDevice);

    // allocate post-processing buffer for imageMinMax
    Rpp32f *imageMinMaxArr;
    Rpp32u imageMinMaxArrLength = 2 * srcDescPtrRGB->n;
    hipHostMalloc(&imageMinMaxArr, imageMinMaxArrLength * sizeof(Rpp32f));

    // declare cpu outputs for optical flow
    // cv::Mat hsv[3], angle, bgr;      // ------- revisit

    // declare gpu outputs for optical flow
    // cv::cuda::GpuMat gpuMagnitude, gpuNormalizedMagnitude, gpuAngle;      // ------- revisit
    // cv::cuda::GpuMat gpuHSV[3], gpuMergedHSV, gpuHSV_8u, gpuBGR;      // ------- revisit

    // set saturation to 1
    // hsv[1] = cv::Mat::ones(frame.size(), CV_32F);      // ------- revisit
    // gpuHSV[1].upload(hsv[1]);      // ------- revisit

    while (true)
    {
        // start full pipeline timer
        auto start_full_time = high_resolution_clock::now();

        // start reading timer
        auto start_read_time = high_resolution_clock::now();

        // capture frame-by-frame
        capture >> frame;

        if (frame.empty())
            break;

        // copy frame into rpp hip buffer
        // rpp_tensor_create_strided(frame.data, srcRGB, srcDescPtrRGB);    // not required since farneback frame size is already multiple of 8
        hipMemcpy(d_srcRGB, frame.data, sizeInBytesSrcRGB, hipMemcpyHostToDevice);
        hipDeviceSynchronize();

        // end reading timer
        auto end_read_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["reading"].push_back(duration_cast<microseconds>(end_read_time - start_read_time).count() / 1000.0);

        // start pre-process timer
        auto start_pre_time = high_resolution_clock::now();

        // resize frame
        rppt_resize_gpu(d_srcRGB, srcDescPtrRGB, d_srcResizedRGB, srcResizedDescPtrRGB, src1ImgSizes, interpolationType, roiTensorPtrSrcRGB, roiType, handle);
        // rppt_resize_host(srcRGB, srcDescPtrRGB, srcResizedRGB, srcResizedDescPtrRGB, src1ImgSizes, interpolationType, roiTensorPtrSrcRGB, roiType, handle);
        hipDeviceSynchronize();

        // convert to gray
        rppt_color_to_greyscale_gpu(d_srcResizedRGB, srcResizedDescPtrRGB, d_src2, src2DescPtr, srcSubpixelLayout, handle);
        // rppt_color_to_greyscale_host(srcResizedRGB, srcResizedDescPtrRGB, src2, src2DescPtr, srcSubpixelLayout, handle);
        hipDeviceSynchronize();

        // end pre-process timer
        auto end_pre_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["pre-process"].push_back(duration_cast<microseconds>(end_pre_time - start_pre_time).count() / 1000.0);

        // start optical flow timer
        auto start_of_time = high_resolution_clock::now();

        // create optical flow instance
        // Ptr<cuda::FarnebackOpticalFlow> ptr_calc = cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, 0);
        // calculate optical flow
        // cv::cuda::GpuMat gpu_flow;
        // ptr_calc->calc(gpuPrevious, gpu_current, gpu_flow);

        // end optical flow timer
        auto end_of_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["optical flow"].push_back(duration_cast<microseconds>(end_of_time - start_of_time).count() / 1000.0);

        // start post-process timer
        auto start_post_time = high_resolution_clock::now();

        // -------------------- stage output dump check --------------------
        // Rpp32f motionVectorsCartesianCPU[FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE];
        // hipMemcpy(motionVectorsCartesianCPU, d_motionVectorsCartesianF32, FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsCartesianCPU", motionVectorsCartesianCPU, (RpptDescPtr)motionVectorPkdDescPtr);

        // convert from cartesian to polar coordinates
        rppt_cartesian_to_polar_gpu(d_motionVectorsCartesianF32, motionVectorPkdDescPtr, d_motionVectorsPolarF32, motionVectorPlnDescPtr, angleType, roiTensorPtrSrcResized, roiType, handle);
        hipDeviceSynchronize();

        // -------------------- stage output dump check --------------------
        // Rpp32f motionVectorsPolarCPU[FARNEBACK_OUTPUT_FRAME_SIZE];
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPU", motionVectorsPolarCPU, (RpptDescPtr)motionVectorCompPlnDescPtr);
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp1, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPUComp1", motionVectorsPolarCPU, (RpptDescPtr)motionVectorCompPlnDescPtr);
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp2, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPUComp2", motionVectorsPolarCPU, (RpptDescPtr)motionVectorCompPlnDescPtr);
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp3, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPUComp3", motionVectorsPolarCPU, (RpptDescPtr)motionVectorCompPlnDescPtr);

        // find min and max of polar magnitude
        rppt_image_min_max_gpu(d_motionVectorsPolarF32, motionVectorCompPlnDescPtr, imageMinMaxArr, imageMinMaxArrLength, roiTensorPtrSrcResized, roiType, handle);
        hipDeviceSynchronize();

        // normalize polar magnitude from 0 to 1
        rppt_normalize_minmax_gpu(d_motionVectorsPolarF32, motionVectorCompPlnDescPtr, d_motionVectorsPolarF32Comp3, motionVectorCompPlnDescPtr, imageMinMaxArr, imageMinMaxArrLength, 0.0f, 1.0f, roiTensorPtrSrcResized, roiType, handle);
        hipDeviceSynchronize();

        // normalize polar angle from 0 to 1
        rppt_multiply_scalar_gpu(d_motionVectorsPolarF32Comp1, motionVectorCompPlnDescPtr, d_motionVectorsPolarF32Comp1, motionVectorCompPlnDescPtr, HUE_CONVERSION_FACTOR, roiTensorPtrSrcResized, roiType, handle);
        hipDeviceSynchronize();

        // -------------------- stage output dump check --------------------
        Rpp32f motionVectorsPolarCPU[FARNEBACK_OUTPUT_FRAME_SIZE];
        hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp1, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        rpp_tensor_write_to_file("motionVectorsPolarCPUComp1", motionVectorsPolarCPU, (RpptDescPtr)motionVectorCompPlnDescPtr);
        hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp2, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        rpp_tensor_write_to_file("motionVectorsPolarCPUComp2", motionVectorsPolarCPU, (RpptDescPtr)motionVectorCompPlnDescPtr);
        hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp3, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        rpp_tensor_write_to_file("motionVectorsPolarCPUComp3", motionVectorsPolarCPU, (RpptDescPtr)motionVectorCompPlnDescPtr);

        // end post pipeline timer
        auto end_post_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["post-process"].push_back(duration_cast<microseconds>(end_post_time - start_post_time).count() / 1000.0);

        // end full pipeline timer
        auto end_full_time = high_resolution_clock::now();

        // add elapsed iteration time
        timers["full pipeline"].push_back(duration_cast<microseconds>(end_full_time - start_full_time).count() / 1000.0);

        // visualization
        // imshow("original", frame);
        // imshow("result", bgr);
        // int keyboard = waitKey(1);
        // if (keyboard == 27)
        //     break;
        hipMemcpy(src2, d_src2, sizeInBytesSrc2, hipMemcpyDeviceToHost);
        rpp_tensor_write_to_file("src2", src2, src2DescPtr);
        rpp_tensor_write_to_images("src2", src2, src2DescPtr, src1ImgSizes);

        break;
    }

    capture.release();
    destroyAllWindows();

    // destroy rpp handle and deallocate all buffers
    rppDestroyGPU(handle);
    hipFree(&d_srcRGB);
    hipFree(&d_srcResizedRGB);
    hipFree(&d_src1);
    hipFree(&d_src2);
    hipFree(&d_motionVectorsCartesianF32);
    hipFree(&d_motionVectorsPolarF32);
    hipHostFree(&roiTensorPtrSrcRGB);
    hipHostFree(&roiTensorPtrSrcResized);
    hipHostFree(&src1ImgSizes);
    free(srcRGB);
    free(srcResizedRGB);
    free(src1);
    free(src2);

    // display video file properties to user
    cout << "\nInput Video File - " << inputVideoFileName;
    cout << "\nFPS - " << fps;
    cout << "\nNumber of Frames - " << numOfFrames;
    cout << "\nFrame Width - " << frameWidth;
    cout << "\nFrame Height - " << frameHeight;
    cout << "\nBit Rate - " << bitRate;

    // elapsed time at each stage
    cout << "\n\nElapsed time:";
    for (auto const& timer : timers)
        cout << "\n- " << timer.first << " : " << std::accumulate(timer.second.begin(), timer.second.end(), 0.0) << " milliseconds";

    // calculate frames per second
    float opticalFlowFPS  = (numOfFrames - 1) / std::accumulate(timers["optical flow"].begin(),  timers["optical flow"].end(),  0.0);
    float fullPipelineFPS = (numOfFrames - 1) / std::accumulate(timers["full pipeline"].begin(), timers["full pipeline"].end(), 0.0);
    cout << "\n\nInput video FPS : " << fps;
    cout << "\nOptical flow FPS : " << opticalFlowFPS;
    cout << "\nFull pipeline FPS : " << fullPipelineFPS;
    cout << "\n";
}

int main(int argc, const char** argv)
{
    // handle inputs
    const int ARG_COUNT = 2;
    if (argc != ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_hip_optical_flow <input video file>\n");
        return -1;
    }
    string inputVideoFileName;
    inputVideoFileName = argv[1];

    // query and fix max batch size
    const auto cpuThreadCount = std::thread::hardware_concurrency();
    cout << "\n\nCPU Threads = " << cpuThreadCount;

    int device, deviceCount;
    hipGetDevice(&device);
    hipGetDeviceCount(&deviceCount);
    cout << "\nDevice = " << device;
    cout << "\nDevice Count = " << deviceCount;
    cout << "\n";

    // run optical flow
    cout << "\n\nProcessing RPP optical flow on " << inputVideoFileName << " with HIP backend...\n\n";
    rpp_optical_flow_hip(inputVideoFileName);

    return 0;
}
