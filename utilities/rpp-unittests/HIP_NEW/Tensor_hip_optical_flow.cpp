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

using namespace cv;
using namespace std;
using namespace std::chrono;

#define FARNEBACK_FRAME_WIDTH 960
#define FARNEBACK_FRAME_HEIGHT 540
#define FARNEBACK_OUTPUT_FRAME_SIZE (FARNEBACK_FRAME_WIDTH * FARNEBACK_FRAME_HEIGHT)

const RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
const RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::BGRtype;
const RpptRoiType roiType = RpptRoiType::XYWH;

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

void rpp_tensor_print(string tensorName, Rpp8u *tensor, RpptDescPtr desc)
{
    std::cout << "\n" << tensorName << ":\n";
    Rpp8u *tensorTemp;
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
                    std::cout << (Rpp32u)*tensorTemp << ", ";
                    tensorTemp++;
                }
            }
            std::cout << ";\n";
        }
        std::cout << " ]\n\n";
    }
}

void rpp_tensor_write_to_file(string tensorName, Rpp8u *tensor, RpptDescPtr desc)
{
    ofstream outputFile (tensorName + ".csv");

    outputFile << "\n" << tensorName << ":\n";
    Rpp8u *tensorTemp;
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
                    outputFile << (Rpp32u)*tensorTemp << ", ";
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

        Mat outputImage_mat;
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

    // initialize rpp tensor descriptor for srcRGB frames
    srcDescPtrRGB = &srcDescRGB;
    srcDescPtrRGB->layout = RpptLayout::NHWC;
    srcDescPtrRGB->dataType = RpptDataType::U8;
    srcDescPtrRGB->numDims = 4;
    srcDescPtrRGB->offsetInBytes = 64;
    srcDescPtrRGB->n = 1;
    srcDescPtrRGB->h = frameHeight;
    srcDescPtrRGB->w = frameWidth; // -------------------- to add the / 8 * 8 + 8?
    srcDescPtrRGB->c = 3;
    srcDescPtrRGB->strides.nStride = srcDescPtrRGB->c * srcDescPtrRGB->w * srcDescPtrRGB->h;
    srcDescPtrRGB->strides.hStride = srcDescPtrRGB->c * srcDescPtrRGB->w;
    srcDescPtrRGB->strides.wStride = srcDescPtrRGB->c;
    srcDescPtrRGB->strides.cStride = 1;

    // initialize rpp tensor descriptor for srcResizedRGB frames (same descriptor as srcRGB except farneback width/height and stride changes)
    srcResizedDescPtrRGB = &srcResizedDescRGB;
    srcResizedDescRGB = srcDescRGB;
    srcResizedDescPtrRGB->h = FARNEBACK_FRAME_HEIGHT;
    srcResizedDescPtrRGB->w = FARNEBACK_FRAME_WIDTH; // -------------------- to add the / 8 * 8 + 8?
    srcResizedDescPtrRGB->strides.nStride = srcResizedDescPtrRGB->c * srcResizedDescPtrRGB->w * srcResizedDescPtrRGB->h;
    srcResizedDescPtrRGB->strides.hStride = srcResizedDescPtrRGB->c * srcResizedDescPtrRGB->w;

    // initialize rpp tensor descriptors for src greyscale frames
    src1DescPtr = &src1Desc;
    src1DescPtr->layout = RpptLayout::NCHW;
    src1DescPtr->dataType = RpptDataType::U8;
    src1DescPtr->numDims = 4;
    src1DescPtr->offsetInBytes = 64;
    src1DescPtr->n = srcDescPtrRGB->n;
    src1DescPtr->c = 1;
    src1DescPtr->h = FARNEBACK_FRAME_HEIGHT;
    src1DescPtr->w = FARNEBACK_FRAME_WIDTH; // -------------------- to add the / 8 * 8 + 8?
    src1DescPtr->strides.nStride = src1DescPtr->c * src1DescPtr->w * src1DescPtr->h;
    src1DescPtr->strides.cStride = src1DescPtr->w * src1DescPtr->h;
    src1DescPtr->strides.hStride = src1DescPtr->w;
    src1DescPtr->strides.wStride = 1;
    src2DescPtr = &src2Desc;
    src2Desc = src1Desc;

    // allocate rpp roi and imagePatch buffers on host and hip for resize ops
    RpptROI *roiTensorPtrSrcRGB = (RpptROI *) calloc(srcDescPtrRGB->n, sizeof(RpptROI));
    RpptROI *d_roiTensorPtrSrcRGB;
    hipMalloc(&d_roiTensorPtrSrcRGB, srcDescPtrRGB->n * sizeof(RpptROI));
    RpptImagePatch *src1ImgSizes = (RpptImagePatch *) calloc(src1DescPtr->n, sizeof(RpptImagePatch));
    RpptImagePatch *d_src1ImgSizes;
    hipMalloc(&d_src1ImgSizes, src1DescPtr->n * sizeof(RpptImagePatch));

    // initialize rpp roi and imagePatch buffers on host and hip for resize ops (currently for single image - can be modified for batch)
    roiTensorPtrSrcRGB[0].xywhROI.xy.x = 0;
    roiTensorPtrSrcRGB[0].xywhROI.xy.y = 0;
    roiTensorPtrSrcRGB[0].xywhROI.roiWidth = srcDescPtrRGB->w;
    roiTensorPtrSrcRGB[0].xywhROI.roiHeight = srcDescPtrRGB->h;
    src1ImgSizes[0].width = src1DescPtr->w;
    src1ImgSizes[0].height = src1DescPtr->h;
    hipMemcpy(d_roiTensorPtrSrcRGB, roiTensorPtrSrcRGB, srcDescPtrRGB->n * sizeof(RpptROI), hipMemcpyHostToDevice);
    hipMemcpy(d_src1ImgSizes, src1ImgSizes, src1DescPtr->n * sizeof(RpptImagePatch), hipMemcpyHostToDevice);

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

    // read the first frame
    cv::Mat frame, previousFrame;
    capture >> frame;

    // copy frame into rpp strided tensor host and hip buffers
    rpp_tensor_create_strided(frame.data, srcRGB, srcDescPtrRGB);
    hipMemcpy(d_srcRGB, srcRGB, sizeInBytesSrcRGB, hipMemcpyHostToDevice);

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
    // -------------------- stage output dump check --------------------

    // resize frame
    rppt_resize_gpu(d_srcRGB, srcDescPtrRGB, d_srcResizedRGB, srcResizedDescPtrRGB, d_src1ImgSizes, interpolationType, d_roiTensorPtrSrcRGB, roiType, handle);
    hipDeviceSynchronize();
    // -------------------- stage output dump check --------------------
    // hipMemcpy(srcResizedRGB, d_srcResizedRGB, sizeInBytesSrcResizedRGB, hipMemcpyDeviceToHost);
    // rpp_tensor_write_to_file("srcResizedRGB", srcResizedRGB, srcResizedDescPtrRGB);
    // rpp_tensor_write_to_images("srcResizedRGB", srcResizedRGB, srcResizedDescPtrRGB, src1ImgSizes);
    // -------------------- stage output dump check --------------------

    // convert to gray
    rppt_color_to_greyscale_gpu(d_srcResizedRGB, srcResizedDescPtrRGB, d_src1, src1DescPtr, srcSubpixelLayout, handle);
    hipDeviceSynchronize();
    // -------------------- stage output dump check --------------------
    hipMemcpy(src1, d_src1, sizeInBytesSrc1, hipMemcpyDeviceToHost);
    rpp_tensor_write_to_file("src1", src1, src1DescPtr);
    rpp_tensor_write_to_images("src1", src1, src1DescPtr, src1ImgSizes);
    // -------------------- stage output dump check --------------------

    // declare cpu outputs for optical flow
    // cv::Mat hsv[3], angle, bgr;      // ------- revisit

    // declare gpu outputs for optical flow
    // cv::cuda::GpuMat gpuMagnitude, gpuNormalizedMagnitude, gpuAngle;      // ------- revisit
    // cv::cuda::GpuMat gpuHSV[3], gpuMergedHSV, gpuHSV_8u, gpuBGR;      // ------- revisit

    // set saturation to 1
    // hsv[1] = cv::Mat::ones(frame.size(), CV_32F);      // ------- revisit
    // gpuHSV[1].upload(hsv[1]);      // ------- revisit

    // while (true)
    // {
    //     // start full pipeline timer
    //     auto start_full_time = high_resolution_clock::now();

    //     // start reading timer
    //     auto start_read_time = high_resolution_clock::now();

    //     // capture frame-by-frame
    //     capture >> frame;

    //     if (frame.empty())
    //         break;

    //     // upload frame to GPU
    //     cv::cuda::GpuMat gpu_frame;
    //     gpu_frame.upload(frame);

    //     // end reading timer
    //     auto end_read_time = high_resolution_clock::now();

    //     // add elapsed iteration time
    //     timers["reading"].push_back(duration_cast<milliseconds>(end_read_time - start_read_time).count() / 1000.0);

    //     // start pre-process timer
    //     auto start_pre_time = high_resolution_clock::now();

    //     // resize frame
    //     cv::cuda::resize(gpu_frame, gpu_frame, Size(FARNEBACK_FRAME_WIDTH, FARNEBACK_FRAME_HEIGHT), 0, 0, INTER_LINEAR);

    //     // convert to gray
    //     cv::cuda::GpuMat gpu_current;
    //     cv::cuda::cvtColor(gpu_frame, gpu_current, COLOR_BGR2GRAY);

    //     // end pre-process timer
    //     auto end_pre_time = high_resolution_clock::now();

    //     // add elapsed iteration time
    //     timers["pre-process"].push_back(duration_cast<milliseconds>(end_pre_time - start_pre_time).count() / 1000.0);

    //     // start optical flow timer
    //     auto start_of_time = high_resolution_clock::now();

    //     // create optical flow instance
    //     Ptr<cuda::FarnebackOpticalFlow> ptr_calc = cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, 0);
    //     // calculate optical flow
    //     cv::cuda::GpuMat gpu_flow;
    //     ptr_calc->calc(gpuPrevious, gpu_current, gpu_flow);

    //     // end optical flow timer
    //     auto end_of_time = high_resolution_clock::now();

    //     // add elapsed iteration time
    //     timers["optical flow"].push_back(duration_cast<milliseconds>(end_of_time - start_of_time).count() / 1000.0);

    //     // start post-process timer
    //     auto start_post_time = high_resolution_clock::now();

    //     // split the output flow into 2 vectors
    //     cv::cuda::GpuMat gpu_flow_xy[2];
    //     cv::cuda::split(gpu_flow, gpu_flow_xy);

    //     // convert from cartesian to polar coordinates
    //     cv::cuda::cartToPolar(gpu_flow_xy[0], gpu_flow_xy[1], gpuMagnitude, gpuAngle, true);

    //     // normalize magnitude from 0 to 1
    //     cv::cuda::normalize(gpuMagnitude, gpuNormalizedMagnitude, 0.0, 1.0, NORM_MINMAX, -1);

    //     // get angle of optical flow
    //     gpuAngle.download(angle);
    //     angle *= ((1 / 360.0) * (180 / 255.0));

    //     // build hsv image
    //     gpuHSV[0].upload(angle);
    //     gpuHSV[2] = gpuNormalizedMagnitude;
    //     cv::cuda::merge(gpuHSV, 3, gpuMergedHSV);

    //     // multiply each pixel value to 255
    //     gpuMergedHSV.cv::cuda::GpuMat::convertTo(gpuHSV_8u, CV_8U, 255.0);

    //     // convert hsv to bgr
    //     cv::cuda::cvtColor(gpuHSV_8u, gpuBGR, COLOR_HSV2BGR);

    //     // send original frame from GPU back to CPU
    //     gpu_frame.download(frame);

    //     // send result from GPU back to CPU
    //     gpuBGR.download(bgr);

    //     // update previousFrame value
    //     gpuPrevious = gpu_current;

    //     // end post pipeline timer
    //     auto end_post_time = high_resolution_clock::now();

    //     // add elapsed iteration time
    //     timers["post-process"].push_back(duration_cast<milliseconds>(end_post_time - start_post_time).count() / 1000.0);

    //     // end full pipeline timer
    //     auto end_full_time = high_resolution_clock::now();

    //     // add elapsed iteration time
    //     timers["full pipeline"].push_back(duration_cast<milliseconds>(end_full_time - start_full_time).count() / 1000.0);

    //     // visualization
    //     imshow("original", frame);
    //     imshow("result", bgr);
    //     int keyboard = waitKey(1);
    //     if (keyboard == 27)
    //         break;
    // }

    // capture.release();
    // destroyAllWindows();

    // destroy rpp handle and deallocate all buffers
    rppDestroyGPU(handle);
    hipFree(&d_roiTensorPtrSrcRGB);
    hipFree(&d_src1ImgSizes);
    hipFree(&d_srcRGB);
    hipFree(&d_srcResizedRGB);
    hipFree(&d_src1);
    hipFree(&d_src2);
    free(roiTensorPtrSrcRGB);
    free(src1ImgSizes);
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
        cout << "\n- " << timer.first << " : " << std::accumulate(timer.second.begin(), timer.second.end(), 0.0) << " seconds";

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















// #include <stdio.h>
// #include <dirent.h>
// #include <string.h>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include "rpp.h"
// #include <sys/types.h>
// #include <sys/stat.h>
// #include <unistd.h>
// #include <time.h>
// #include <omp.h>
// #include <hip/hip_fp16.h>
// #include <fstream>

// using namespace cv;
// using namespace std;

// #define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))
// #define RPPMAX2(a,b) ((a > b) ? a : b)
// #define RPPMIN2(a,b) ((a < b) ? a : b)

// std::string get_interpolation_type(unsigned int val, RpptInterpolationType &interpolationType)
// {
//     switch(val)
//     {
//         case 0:
//         {
//             interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
//             return "NearestNeighbor";
//         }
//         case 2:
//         {
//             interpolationType = RpptInterpolationType::BICUBIC;
//             return "Bicubic";
//         }
//         case 3:
//         {
//             interpolationType = RpptInterpolationType::LANCZOS;
//             return "Lanczos";
//         }
//         case 4:
//         {
//             interpolationType = RpptInterpolationType::TRIANGULAR;
//             return "Triangular";
//         }
//         case 5:
//         {
//             interpolationType = RpptInterpolationType::GAUSSIAN;
//             return "Gaussian";
//         }
//         default:
//         {
//             interpolationType = RpptInterpolationType::BILINEAR;
//             return "Bilinear";
//         }
//     }
// }

// std::string get_noise_type(unsigned int val)
// {
//     switch(val)
//     {
//         case 0: return "SaltAndPepper";
//         case 1: return "Gaussian";
//         case 2: return "Shot";
//         default:return "SaltAndPepper";
//     }
// }

// int main(int argc, char **argv)
// {
//     // Handle inputs

//     const int MIN_ARG_COUNT = 8;

//     if (argc < MIN_ARG_COUNT)
//     {
//         printf("\nImproper Usage! Needs all arguments!\n");
//         printf("\nUsage: ./Tensor_hip_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <verbosity = 0/1>\n");
//         return -1;
//     }

//     char *src = argv[1];
//     char *src_second = argv[2];
//     char *dst = argv[3];
//     int ip_bitDepth = atoi(argv[4]);
//     unsigned int outputFormatToggle = atoi(argv[5]);
//     int test_case = atoi(argv[6]);

//     bool additionalParamCase = (test_case == 8 || test_case == 21 || test_case == 24 || test_case == 40 || test_case == 41 || test_case == 49);
//     bool kernelSizeCase = (test_case == 40 || test_case == 41 || test_case == 49);
//     bool interpolationTypeCase = (test_case == 21 || test_case == 24);
//     bool noiseTypeCase = (test_case == 8);
//     bool pln1OutTypeCase = (test_case == 86);

//     unsigned int verbosity = additionalParamCase ? atoi(argv[8]) : atoi(argv[7]);
//     unsigned int additionalParam = additionalParamCase ? atoi(argv[7]) : 1;

//     if (verbosity == 1)
//     {
//         printf("\nInputs for this test case are:");
//         printf("\nsrc1 = %s", argv[1]);
//         printf("\nsrc2 = %s", argv[2]);
//         printf("\ndst = %s", argv[3]);
//         printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[4]);
//         printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[5]);
//         printf("\ncase number (0:86) = %s", argv[6]);
//     }

//     int ip_channel = 3;

//     // Set case names

//     char funcType[1000] = {"Tensor_HIP_PKD3"};

//     char funcName[1000];
//     switch (test_case)
//     {
//     case 0:
//         strcpy(funcName, "brightness");
//         break;
//     case 1:
//         strcpy(funcName, "gamma_correction");
//         break;
//     case 2:
//         strcpy(funcName, "blend");
//         break;
//     case 4:
//         strcpy(funcName, "contrast");
//         break;
//     case 8:
//         strcpy(funcName, "noise");
//         break;
//     case 13:
//         strcpy(funcName, "exposure");
//         break;
//     case 20:
//         strcpy(funcName, "flip");
//         break;
//     case 21:
//         strcpy(funcName, "resize");
//         break;
//     case 24:
//         strcpy(funcName, "warp_affine");
//         break;
//     case 31:
//         strcpy(funcName, "color_cast");
//         break;
//     case 36:
//         strcpy(funcName, "color_twist");
//         break;
//     case 37:
//         strcpy(funcName, "crop");
//         break;
//     case 38:
//         strcpy(funcName, "crop_mirror_normalize");
//         break;
//     case 39:
//         strcpy(funcName, "resize_crop_mirror");
//         break;
//     case 40:
//         strcpy(funcName, "erode");
//         break;
//     case 41:
//         strcpy(funcName, "dilate");
//         break;
//     case 49:
//         strcpy(funcName, "box_filter");
//         break;
//     case 70:
//         strcpy(funcName, "copy");
//         break;
//     case 80:
//         strcpy(funcName, "resize_mirror_normalize");
//         break;
//     case 83:
//         strcpy(funcName, "gridmask");
//         break;
//     case 84:
//         strcpy(funcName, "spatter");
//         break;
//     case 85:
//         strcpy(funcName, "swap_channels");
//         break;
//     case 86:
//         strcpy(funcName, "color_to_greyscale");
//         break;
//     default:
//         strcpy(funcName, "test_case");
//         break;
//     }

//     // Initialize tensor descriptors

//     RpptDesc srcDesc, dstDesc;
//     RpptDescPtr srcDescPtr, dstDescPtr;
//     srcDescPtr = &srcDesc;
//     dstDescPtr = &dstDesc;

//     // Set src/dst layouts in tensor descriptors

//     srcDescPtr->layout = RpptLayout::NHWC;
//     if (pln1OutTypeCase)
//     {
//         strcat(funcType, "_toPLN1");
//         dstDescPtr->layout = RpptLayout::NCHW;
//     }
//     else
//     {
//         if (outputFormatToggle == 0)
//         {
//             strcat(funcType, "_toPKD3");
//             dstDescPtr->layout = RpptLayout::NHWC;
//         }
//         else if (outputFormatToggle == 1)
//         {
//             strcat(funcType, "_toPLN3");
//             dstDescPtr->layout = RpptLayout::NCHW;
//         }
//     }

//     // Set src/dst data types in tensor descriptors

//     if (ip_bitDepth == 0)
//     {
//         strcat(funcName, "_u8_");
//         srcDescPtr->dataType = RpptDataType::U8;
//         dstDescPtr->dataType = RpptDataType::U8;
//     }
//     else if (ip_bitDepth == 1)
//     {
//         strcat(funcName, "_f16_");
//         srcDescPtr->dataType = RpptDataType::F16;
//         dstDescPtr->dataType = RpptDataType::F16;
//     }
//     else if (ip_bitDepth == 2)
//     {
//         strcat(funcName, "_f32_");
//         srcDescPtr->dataType = RpptDataType::F32;
//         dstDescPtr->dataType = RpptDataType::F32;
//     }
//     else if (ip_bitDepth == 3)
//     {
//         strcat(funcName, "_u8_f16_");
//         srcDescPtr->dataType = RpptDataType::U8;
//         dstDescPtr->dataType = RpptDataType::F16;
//     }
//     else if (ip_bitDepth == 4)
//     {
//         strcat(funcName, "_u8_f32_");
//         srcDescPtr->dataType = RpptDataType::U8;
//         dstDescPtr->dataType = RpptDataType::F32;
//     }
//     else if (ip_bitDepth == 5)
//     {
//         strcat(funcName, "_i8_");
//         srcDescPtr->dataType = RpptDataType::I8;
//         dstDescPtr->dataType = RpptDataType::I8;
//     }
//     else if (ip_bitDepth == 6)
//     {
//         strcat(funcName, "_u8_i8_");
//         srcDescPtr->dataType = RpptDataType::U8;
//         dstDescPtr->dataType = RpptDataType::I8;
//     }

//     // Other initializations

//     int missingFuncFlag = 0;
//     int i = 0, j = 0;
//     int maxHeight = 0, maxWidth = 0;
//     int maxDstHeight = 0, maxDstWidth = 0;
//     unsigned long long count = 0;
//     unsigned long long ioBufferSize = 0;
//     unsigned long long oBufferSize = 0;
//     static int noOfImages = 0;
//     Mat image, image_second;

//     // String ops on function name

//     char src1[1000];
//     strcpy(src1, src);
//     strcat(src1, "/");
//     char src1_second[1000];
//     strcpy(src1_second, src_second);
//     strcat(src1_second, "/");

//     char func[1000];
//     strcpy(func, funcName);
//     strcat(func, funcType);
//     strcat(funcName, funcType);
//     strcat(dst, "/");
//     strcat(dst, funcName);

//     RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
//     if (kernelSizeCase)
//     {
//         char additionalParam_char[2];
//         std::sprintf(additionalParam_char, "%u", additionalParam);
//         strcat(func, "_kSize");
//         strcat(func, additionalParam_char);
//         strcat(dst, "_kSize");
//         strcat(dst, additionalParam_char);
//     }
//     else if (interpolationTypeCase)
//     {
//         std::string interpolationTypeName;
//         interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
//         strcat(func, "_interpolationType");
//         strcat(func, interpolationTypeName.c_str());
//         strcat(dst, "_interpolationType");
//         strcat(dst, interpolationTypeName.c_str());
//     }
//     else if (noiseTypeCase)
//     {
//         std::string noiseTypeName;
//         noiseTypeName = get_noise_type(additionalParam);
//         strcat(func, "_noiseType");
//         strcat(func, noiseTypeName.c_str());
//         strcat(dst, "_noiseType");
//         strcat(dst, noiseTypeName.c_str());
//     }

//     printf("\nRunning %s...", func);

//     // Get number of images

//     struct dirent *de;
//     DIR *dr = opendir(src);
//     while ((de = readdir(dr)) != NULL)
//     {
//         if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
//             continue;
//         noOfImages += 1;
//     }
//     closedir(dr);

//     // Initialize ROI tensors for src/dst

//     RpptROI *roiTensorPtrSrc = (RpptROI *) calloc(noOfImages, sizeof(RpptROI));
//     RpptROI *roiTensorPtrDst = (RpptROI *) calloc(noOfImages, sizeof(RpptROI));

//     RpptROI *d_roiTensorPtrSrc, *d_roiTensorPtrDst;
//     hipMalloc(&d_roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
//     hipMalloc(&d_roiTensorPtrDst, noOfImages * sizeof(RpptROI));

//     // Initialize the ImagePatch for source and destination

//     RpptImagePatch *srcImgSizes = (RpptImagePatch *) calloc(noOfImages, sizeof(RpptImagePatch));
//     RpptImagePatch *dstImgSizes = (RpptImagePatch *) calloc(noOfImages, sizeof(RpptImagePatch));

//     RpptImagePatch *d_srcImgSizes, *d_dstImgSizes;
//     hipMalloc(&d_srcImgSizes, noOfImages * sizeof(RpptImagePatch));
//     hipMalloc(&d_dstImgSizes, noOfImages * sizeof(RpptImagePatch));

//     // Set ROI tensors types for src/dst

//     RpptRoiType roiTypeSrc, roiTypeDst;
//     roiTypeSrc = RpptRoiType::XYWH;
//     roiTypeDst = RpptRoiType::XYWH;

//     // Set maxHeight, maxWidth and ROIs for src/dst

//     const int images = noOfImages;
//     char imageNames[images][1000];

//     DIR *dr1 = opendir(src);
//     while ((de = readdir(dr1)) != NULL)
//     {
//         if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
//             continue;
//         strcpy(imageNames[count], de->d_name);
//         char temp[1000];
//         strcpy(temp, src1);
//         strcat(temp, imageNames[count]);

//         image = imread(temp, 1);

//         roiTensorPtrSrc[count].xywhROI.xy.x = 0;
//         roiTensorPtrSrc[count].xywhROI.xy.y = 0;
//         roiTensorPtrSrc[count].xywhROI.roiWidth = image.cols;
//         roiTensorPtrSrc[count].xywhROI.roiHeight = image.rows;

//         roiTensorPtrDst[count].xywhROI.xy.x = 0;
//         roiTensorPtrDst[count].xywhROI.xy.y = 0;
//         roiTensorPtrDst[count].xywhROI.roiWidth = image.cols;
//         roiTensorPtrDst[count].xywhROI.roiHeight = image.rows;

//         srcImgSizes[count].width = roiTensorPtrSrc[count].xywhROI.roiWidth;
//         srcImgSizes[count].height = roiTensorPtrSrc[count].xywhROI.roiHeight;
//         dstImgSizes[count].width = roiTensorPtrDst[count].xywhROI.roiWidth;
//         dstImgSizes[count].height = roiTensorPtrDst[count].xywhROI.roiHeight;

//         maxHeight = RPPMAX2(maxHeight, roiTensorPtrSrc[count].xywhROI.roiHeight);
//         maxWidth = RPPMAX2(maxWidth, roiTensorPtrSrc[count].xywhROI.roiWidth);
//         maxDstHeight = RPPMAX2(maxDstHeight, roiTensorPtrDst[count].xywhROI.roiHeight);
//         maxDstWidth = RPPMAX2(maxDstWidth, roiTensorPtrDst[count].xywhROI.roiWidth);

//         count++;
//     }
//     closedir(dr1);

//     // Set numDims, offset, n/c/h/w values for src/dst

//     srcDescPtr->numDims = 4;
//     dstDescPtr->numDims = 4;

//     srcDescPtr->offsetInBytes = 64;
//     dstDescPtr->offsetInBytes = 0;

//     srcDescPtr->n = noOfImages;
//     srcDescPtr->h = maxHeight;
//     srcDescPtr->w = maxWidth;
//     srcDescPtr->c = ip_channel;

//     dstDescPtr->n = noOfImages;
//     dstDescPtr->h = maxDstHeight;
//     dstDescPtr->w = maxDstWidth;
//     dstDescPtr->c = (pln1OutTypeCase) ? 1 : ip_channel;

//     // Optionally set w stride as a multiple of 8 for src/dst

//     srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
//     dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

//     // Set n/c/h/w strides for src/dst

//     srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
//     srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
//     srcDescPtr->strides.wStride = srcDescPtr->c;
//     srcDescPtr->strides.cStride = 1;

//     if (dstDescPtr->layout == RpptLayout::NHWC)
//     {
//         dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
//         dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
//         dstDescPtr->strides.wStride = dstDescPtr->c;
//         dstDescPtr->strides.cStride = 1;
//     }
//     else if (dstDescPtr->layout == RpptLayout::NCHW)
//     {
//         dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
//         dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
//         dstDescPtr->strides.hStride = dstDescPtr->w;
//         dstDescPtr->strides.wStride = 1;
//     }

//     // Set buffer sizes in pixels for src/dst

//     ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
//     oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

//     // Set buffer sizes in bytes for src/dst (including offsets)

//     unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
//     unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
//     unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
//     unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;
//     unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
//     unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;
//     unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
//     unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

//     // Initialize 8u host buffers for src/dst

//     Rpp8u *input = (Rpp8u *)calloc(ioBufferSizeInBytes_u8, 1);
//     Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSizeInBytes_u8, 1);
//     Rpp8u *output = (Rpp8u *)calloc(oBufferSizeInBytes_u8, 1);
//     if (test_case == 40) memset(input, 0xFF, ioBufferSizeInBytes_u8);

//     // Set 8u host buffers for src/dst

//     DIR *dr2 = opendir(src);
//     DIR *dr2_second = opendir(src_second);
//     count = 0;
//     i = 0;

//     Rpp8u *offsetted_input, *offsetted_input_second;
//     offsetted_input = input + srcDescPtr->offsetInBytes;
//     offsetted_input_second = input_second + srcDescPtr->offsetInBytes;

//     while ((de = readdir(dr2)) != NULL)
//     {
//         Rpp8u *input_temp, *input_second_temp;
//         input_temp = offsetted_input + (i * srcDescPtr->strides.nStride);
//         input_second_temp = offsetted_input_second + (i * srcDescPtr->strides.nStride);

//         if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
//             continue;

//         char temp[1000];
//         strcpy(temp, src1);
//         strcat(temp, de->d_name);

//         char temp_second[1000];
//         strcpy(temp_second, src1_second);
//         strcat(temp_second, de->d_name);

//         image = imread(temp, 1);
//         image_second = imread(temp_second, 1);

//         Rpp8u *ip_image = image.data;
//         Rpp8u *ip_image_second = image_second.data;

//         Rpp32u elementsInRow = roiTensorPtrSrc[i].xywhROI.roiWidth * srcDescPtr->c;

//         for (j = 0; j < roiTensorPtrSrc[i].xywhROI.roiHeight; j++)
//         {
//             memcpy(input_temp, ip_image, elementsInRow * sizeof (Rpp8u));
//             memcpy(input_second_temp, ip_image_second, elementsInRow * sizeof (Rpp8u));
//             ip_image += elementsInRow;
//             ip_image_second += elementsInRow;
//             input_temp += srcDescPtr->strides.hStride;
//             input_second_temp += srcDescPtr->strides.hStride;
//         }
//         i++;
//         count += srcDescPtr->strides.nStride;
//     }
//     closedir(dr2);

//     // Convert inputs to test various other bit depths and copy to hip buffers

//     half *inputf16, *inputf16_second, *outputf16;
//     Rpp32f *inputf32, *inputf32_second, *outputf32;
//     Rpp8s *inputi8, *inputi8_second, *outputi8;
//     int *d_input, *d_input_second, *d_inputf16, *d_inputf16_second, *d_inputf32, *d_inputf32_second, *d_inputi8, *d_inputi8_second;
//     int *d_output, *d_outputf16, *d_outputf32, *d_outputi8;

//     if (ip_bitDepth == 0)
//     {
//         hipMalloc(&d_input, ioBufferSizeInBytes_u8);
//         hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
//         hipMalloc(&d_output, oBufferSizeInBytes_u8);
//         hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
//         hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
//         hipMemcpy(d_output, output, oBufferSizeInBytes_u8, hipMemcpyHostToDevice);
//     }
//     else if (ip_bitDepth == 1)
//     {
//         inputf16 = (half *)calloc(ioBufferSizeInBytes_f16, 1);
//         inputf16_second = (half *)calloc(ioBufferSizeInBytes_f16, 1);
//         outputf16 = (half *)calloc(oBufferSizeInBytes_f16, 1);

//         Rpp8u *inputTemp, *input_secondTemp;
//         half *inputf16Temp, *inputf16_secondTemp;

//         inputTemp = input + srcDescPtr->offsetInBytes;
//         input_secondTemp = input_second + srcDescPtr->offsetInBytes;

//         inputf16Temp = (half *)((Rpp8u *)inputf16 + srcDescPtr->offsetInBytes);
//         inputf16_secondTemp = (half *)((Rpp8u *)inputf16_second + srcDescPtr->offsetInBytes);

//         for (int i = 0; i < ioBufferSize; i++)
//         {
//             *inputf16Temp = (half)(((float)*inputTemp) / 255.0);
//             *inputf16_secondTemp = (half)(((float)*input_secondTemp) / 255.0);
//             inputTemp++;
//             inputf16Temp++;
//             input_secondTemp++;
//             inputf16_secondTemp++;
//         }

//         hipMalloc(&d_inputf16, ioBufferSizeInBytes_f16);
//         hipMalloc(&d_inputf16_second, ioBufferSizeInBytes_f16);
//         hipMalloc(&d_outputf16, oBufferSizeInBytes_f16);
//         hipMemcpy(d_inputf16, inputf16, ioBufferSizeInBytes_f16, hipMemcpyHostToDevice);
//         hipMemcpy(d_inputf16_second, inputf16_second, ioBufferSizeInBytes_f16, hipMemcpyHostToDevice);
//         hipMemcpy(d_outputf16, outputf16, oBufferSizeInBytes_f16, hipMemcpyHostToDevice);
//     }
//     else if (ip_bitDepth == 2)
//     {
//         inputf32 = (Rpp32f *)calloc(ioBufferSizeInBytes_f32, 1);
//         inputf32_second = (Rpp32f *)calloc(ioBufferSizeInBytes_f32, 1);
//         outputf32 = (Rpp32f *)calloc(oBufferSizeInBytes_f32, 1);

//         Rpp8u *inputTemp, *input_secondTemp;
//         Rpp32f *inputf32Temp, *inputf32_secondTemp;

//         inputTemp = input + srcDescPtr->offsetInBytes;
//         input_secondTemp = input_second + srcDescPtr->offsetInBytes;

//         inputf32Temp = (Rpp32f *)((Rpp8u *)inputf32 + srcDescPtr->offsetInBytes);
//         inputf32_secondTemp = (Rpp32f *)((Rpp8u *)inputf32_second + srcDescPtr->offsetInBytes);

//         for (int i = 0; i < ioBufferSize; i++)
//         {
//             *inputf32Temp = ((Rpp32f)*inputTemp) / 255.0;
//             *inputf32_secondTemp = ((Rpp32f)*input_secondTemp) / 255.0;
//             inputTemp++;
//             inputf32Temp++;
//             input_secondTemp++;
//             inputf32_secondTemp++;
//         }

//         hipMalloc(&d_inputf32, ioBufferSizeInBytes_f32);
//         hipMalloc(&d_inputf32_second, ioBufferSizeInBytes_f32);
//         hipMalloc(&d_outputf32, oBufferSizeInBytes_f32);
//         hipMemcpy(d_inputf32, inputf32, ioBufferSizeInBytes_f32, hipMemcpyHostToDevice);
//         hipMemcpy(d_inputf32_second, inputf32_second, ioBufferSizeInBytes_f32, hipMemcpyHostToDevice);
//         hipMemcpy(d_outputf32, outputf32, oBufferSizeInBytes_f32, hipMemcpyHostToDevice);
//     }
//     else if (ip_bitDepth == 3)
//     {
//         outputf16 = (half *)calloc(oBufferSizeInBytes_f16, 1);
//         hipMalloc(&d_input, ioBufferSizeInBytes_u8);
//         hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
//         hipMalloc(&d_outputf16, oBufferSizeInBytes_f16);
//         hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
//         hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
//         hipMemcpy(d_outputf16, outputf16, oBufferSizeInBytes_f16, hipMemcpyHostToDevice);
//     }
//     else if (ip_bitDepth == 4)
//     {
//         outputf32 = (Rpp32f *)calloc(oBufferSizeInBytes_f32, 1);
//         hipMalloc(&d_input, ioBufferSizeInBytes_u8);
//         hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
//         hipMalloc(&d_outputf32, oBufferSizeInBytes_f32);
//         hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
//         hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
//         hipMemcpy(d_outputf32, outputf32, oBufferSizeInBytes_f32, hipMemcpyHostToDevice);
//     }
//     else if (ip_bitDepth == 5)
//     {
//         inputi8 = (Rpp8s *)calloc(ioBufferSizeInBytes_i8, 1);
//         inputi8_second = (Rpp8s *)calloc(ioBufferSizeInBytes_i8, 1);
//         outputi8 = (Rpp8s *)calloc(oBufferSizeInBytes_i8, 1);

//         Rpp8u *inputTemp, *input_secondTemp;
//         Rpp8s *inputi8Temp, *inputi8_secondTemp;

//         inputTemp = input + srcDescPtr->offsetInBytes;
//         input_secondTemp = input_second + srcDescPtr->offsetInBytes;

//         inputi8Temp = inputi8 + srcDescPtr->offsetInBytes;
//         inputi8_secondTemp = inputi8_second + srcDescPtr->offsetInBytes;

//         for (int i = 0; i < ioBufferSize; i++)
//         {
//             *inputi8Temp = (Rpp8s) (((Rpp32s) *inputTemp) - 128);
//             *inputi8_secondTemp = (Rpp8s) (((Rpp32s) *input_secondTemp) - 128);
//             inputTemp++;
//             inputi8Temp++;
//             input_secondTemp++;
//             inputi8_secondTemp++;
//         }

//         hipMalloc(&d_inputi8, ioBufferSizeInBytes_i8);
//         hipMalloc(&d_inputi8_second, ioBufferSizeInBytes_i8);
//         hipMalloc(&d_outputi8, oBufferSizeInBytes_i8);
//         hipMemcpy(d_inputi8, inputi8, ioBufferSizeInBytes_i8, hipMemcpyHostToDevice);
//         hipMemcpy(d_inputi8_second, inputi8_second, ioBufferSizeInBytes_i8, hipMemcpyHostToDevice);
//         hipMemcpy(d_outputi8, outputi8, oBufferSizeInBytes_i8, hipMemcpyHostToDevice);
//     }
//     else if (ip_bitDepth == 6)
//     {
//         outputi8 = (Rpp8s *)calloc(oBufferSizeInBytes_i8, 1);
//         hipMalloc(&d_input, ioBufferSizeInBytes_u8);
//         hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
//         hipMalloc(&d_outputi8, oBufferSizeInBytes_i8);
//         hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
//         hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
//         hipMemcpy(d_outputi8, outputi8, oBufferSizeInBytes_i8, hipMemcpyHostToDevice);
//     }

//     // Run case-wise RPP API and measure time

//     rppHandle_t handle;
//     hipStream_t stream;
//     hipStreamCreate(&stream);
//     rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

//     clock_t start, end;
//     double gpu_time_used;

//     string test_case_name;

//     switch (test_case)
//     {
//     case 0:
//     {
//         test_case_name = "brightness";

//         Rpp32f alpha[images];
//         Rpp32f beta[images];
//         for (i = 0; i < images; i++)
//         {
//             alpha[i] = 1.75;
//             beta[i] = 50;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_brightness_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_brightness_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_brightness_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 1:
//     {
//         test_case_name = "gamma_correction";

//         Rpp32f gammaVal[images];
//         for (i = 0; i < images; i++)
//         {
//             gammaVal[i] = 1.9;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_gamma_correction_gpu(d_input, srcDescPtr, d_output, dstDescPtr, gammaVal, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_gamma_correction_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, gammaVal, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_gamma_correction_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, gammaVal, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_gamma_correction_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, gammaVal, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 2:
//     {
//         test_case_name = "blend";

//         Rpp32f alpha[images];
//         for (i = 0; i < images; i++)
//         {
//             alpha[i] = 0.4;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_blend_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, alpha, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_blend_gpu(d_inputf16, d_inputf16_second, srcDescPtr, d_outputf16, dstDescPtr, alpha, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_blend_gpu(d_inputf32, d_inputf32_second, srcDescPtr, d_outputf32, dstDescPtr, alpha, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_blend_gpu(d_inputi8, d_inputi8_second, srcDescPtr, d_outputi8, dstDescPtr, alpha, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 4:
//     {
//         test_case_name = "contrast";

//         Rpp32f contrastFactor[images];
//         Rpp32f contrastCenter[images];
//         for (i = 0; i < images; i++)
//         {
//             contrastFactor[i] = 2.96;
//             contrastCenter[i] = 128;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/


//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_contrast_gpu(d_input, srcDescPtr, d_output, dstDescPtr, contrastFactor, contrastCenter, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_contrast_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, contrastFactor, contrastCenter, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_contrast_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, contrastFactor, contrastCenter, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_contrast_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, contrastFactor, contrastCenter, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 8:
//     {
//         test_case_name = "noise";

//         switch(additionalParam)
//         {
//             case 0:
//             {
//                 Rpp32f noiseProbabilityTensor[images];
//                 Rpp32f saltProbabilityTensor[images];
//                 Rpp32f saltValueTensor[images];
//                 Rpp32f pepperValueTensor[images];
//                 Rpp32u seed = 1255459;
//                 for (i = 0; i < images; i++)
//                 {
//                     noiseProbabilityTensor[i] = 0.1f;
//                     saltProbabilityTensor[i] = 0.5f;
//                     saltValueTensor[i] = 1.0f;
//                     pepperValueTensor[i] = 0.0f;
//                 }

//                 // Uncomment to run test case with an xywhROI override
//                 /*for (i = 0; i < images; i++)
//                 {
//                     roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//                     roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//                     dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//                     dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//                 }*/

//                 // Uncomment to run test case with an ltrbROI override
//                 /*for (i = 0; i < images; i++)
//                 {
//                     roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//                     roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//                     roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//                     roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//                     dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//                     dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//                 }
//                 roiTypeSrc = RpptRoiType::LTRB;
//                 roiTypeDst = RpptRoiType::LTRB;*/

//                 hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//                 start = clock();

//                 if (ip_bitDepth == 0)
//                     rppt_salt_and_pepper_noise_gpu(d_input, srcDescPtr, d_output, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 1)
//                     rppt_salt_and_pepper_noise_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 2)
//                     rppt_salt_and_pepper_noise_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 3)
//                     missingFuncFlag = 1;
//                 else if (ip_bitDepth == 4)
//                     missingFuncFlag = 1;
//                 else if (ip_bitDepth == 5)
//                     rppt_salt_and_pepper_noise_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 6)
//                     missingFuncFlag = 1;
//                 else
//                     missingFuncFlag = 1;

//                 break;
//             }
//             case 1:
//             {
//                 Rpp32f meanTensor[images];
//                 Rpp32f stdDevTensor[images];
//                 Rpp32u seed = 1255459;
//                 for (i = 0; i < images; i++)
//                 {
//                     meanTensor[i] = 0.0f;
//                     stdDevTensor[i] = 0.2f;
//                 }

//                 // Uncomment to run test case with an xywhROI override
//                 /*for (i = 0; i < images; i++)
//                 {
//                     roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//                     roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//                     dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//                     dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//                 }*/

//                 // Uncomment to run test case with an ltrbROI override
//                 /*for (i = 0; i < images; i++)
//                 {
//                     roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//                     roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//                     roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//                     roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//                     dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//                     dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//                 }
//                 roiTypeSrc = RpptRoiType::LTRB;
//                 roiTypeDst = RpptRoiType::LTRB;*/

//                 hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//                 start = clock();

//                 if (ip_bitDepth == 0)
//                     rppt_gaussian_noise_gpu(d_input, srcDescPtr, d_output, dstDescPtr, meanTensor, stdDevTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 1)
//                     rppt_gaussian_noise_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, meanTensor, stdDevTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 2)
//                     rppt_gaussian_noise_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, meanTensor, stdDevTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 3)
//                     missingFuncFlag = 1;
//                 else if (ip_bitDepth == 4)
//                     missingFuncFlag = 1;
//                 else if (ip_bitDepth == 5)
//                     rppt_gaussian_noise_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, meanTensor, stdDevTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 6)
//                     missingFuncFlag = 1;
//                 else
//                     missingFuncFlag = 1;

//                 break;
//             }
//             case 2:
//             {
//                 Rpp32f shotNoiseFactorTensor[images];
//                 Rpp32u seed = 1255459;
//                 for (i = 0; i < images; i++)
//                 {
//                     shotNoiseFactorTensor[i] = 80.0f;
//                 }

//                 // Uncomment to run test case with an xywhROI override
//                 /*for (i = 0; i < images; i++)
//                 {
//                     roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//                     roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//                     dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//                     dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//                 }*/

//                 // Uncomment to run test case with an ltrbROI override
//                 /*for (i = 0; i < images; i++)
//                 {
//                     roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//                     roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//                     roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//                     roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//                     dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//                     dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//                 }
//                 roiTypeSrc = RpptRoiType::LTRB;
//                 roiTypeDst = RpptRoiType::LTRB;*/

//                 hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//                 start = clock();

//                 if (ip_bitDepth == 0)
//                     rppt_shot_noise_gpu(d_input, srcDescPtr, d_output, dstDescPtr, shotNoiseFactorTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 1)
//                     rppt_shot_noise_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, shotNoiseFactorTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 2)
//                     rppt_shot_noise_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, shotNoiseFactorTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 3)
//                     missingFuncFlag = 1;
//                 else if (ip_bitDepth == 4)
//                     missingFuncFlag = 1;
//                 else if (ip_bitDepth == 5)
//                     rppt_shot_noise_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, shotNoiseFactorTensor, seed, d_roiTensorPtrSrc, roiTypeSrc, handle);
//                 else if (ip_bitDepth == 6)
//                     missingFuncFlag = 1;
//                 else
//                     missingFuncFlag = 1;

//                 break;
//             }
//             default:
//             {
//                 missingFuncFlag = 1;
//                 break;
//             }
//         }

//         break;
//     }
//     case 13:
//     {
//         test_case_name = "exposure";

//         Rpp32f exposureFactor[images];
//         for (i = 0; i < images; i++)
//         {
//             exposureFactor[i] = 1.4;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_exposure_gpu(d_input, srcDescPtr, d_output, dstDescPtr, exposureFactor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_exposure_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, exposureFactor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_exposure_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, exposureFactor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_exposure_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, exposureFactor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 20:
//     {
//         test_case_name = "flip";

//         Rpp32u horizontalFlag[images];
//         Rpp32u verticalFlag[images];
//         for (i = 0; i < images; i++)
//         {
//             horizontalFlag[i] = 1;
//             verticalFlag[i] = 0;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_flip_gpu(d_input, srcDescPtr, d_output, dstDescPtr, horizontalFlag, verticalFlag, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_flip_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, horizontalFlag, verticalFlag, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_flip_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, horizontalFlag, verticalFlag, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_flip_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, horizontalFlag, verticalFlag, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 21:
//     {
//         test_case_name = "resize";

//         if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
//         {
//             missingFuncFlag = 1;
//             break;
//         }

//         for (i = 0; i < images; i++)
//         {
//             dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
//             dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);
//         hipMemcpy(d_dstImgSizes, dstImgSizes, images * sizeof(RpptImagePatch), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, d_dstImgSizes, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_resize_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, d_dstImgSizes, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_resize_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, d_dstImgSizes, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_resize_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, d_dstImgSizes, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 24:
//     {
//         test_case_name = "warp_affine";

//         if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
//         {
//             missingFuncFlag = 1;
//             break;
//         }

//         Rpp32f6 affineTensor_f6[images];
//         Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
//         for (i = 0; i < images; i++)
//         {
//             affineTensor_f6[i].data[0] = 1.23;
//             affineTensor_f6[i].data[1] = 0.5;
//             affineTensor_f6[i].data[2] = 0;
//             affineTensor_f6[i].data[3] = -0.8;
//             affineTensor_f6[i].data[4] = 0.83;
//             affineTensor_f6[i].data[5] = 0;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_warp_affine_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, affineTensor, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_warp_affine_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, affineTensor, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_warp_affine_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, affineTensor, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 31:
//     {
//         test_case_name = "color_cast";

//         RpptRGB rgbTensor[images];
//         Rpp32f alphaTensor[images];

//         for (i = 0; i < images; i++)
//         {
//             rgbTensor[i].R = 0;
//             rgbTensor[i].G = 0;
//             rgbTensor[i].B = 100;
//             alphaTensor[i] = 0.5;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_color_cast_gpu(d_input, srcDescPtr, d_output, dstDescPtr, rgbTensor, alphaTensor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_color_cast_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, rgbTensor, alphaTensor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_color_cast_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, rgbTensor, alphaTensor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_color_cast_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, rgbTensor, alphaTensor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 36:
//     {
//         test_case_name = "color_twist";

//         Rpp32f brightness[images];
//         Rpp32f contrast[images];
//         Rpp32f hue[images];
//         Rpp32f saturation[images];
//         for (i = 0; i < images; i++)
//         {
//             brightness[i] = 1.4;
//             contrast[i] = 0.0;
//             hue[i] = 60.0;
//             saturation[i] = 1.9;
//         }

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, brightness, contrast, hue, saturation, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_color_twist_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, brightness, contrast, hue, saturation, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_color_twist_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, brightness, contrast, hue, saturation, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_color_twist_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, brightness, contrast, hue, saturation, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 37:
//     {
//         test_case_name = "crop";

//         // Uncomment to run test case with an xywhROI override
//         for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_crop_gpu(d_input, srcDescPtr, d_output, dstDescPtr, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_crop_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_crop_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_crop_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 38:
//     {
//         test_case_name = "crop_mirror_normalize";
//         Rpp32f mean[images];
//         Rpp32f stdDev[images];
//         Rpp32u mirror[images];
//         for (i = 0; i < images; i++)
//         {
//             mean[i] = 0.0;
//             stdDev[i] = 1.0;
//             mirror[i] = 1;
//         }

//         // Uncomment to run test case with an xywhROI override
//         for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 50;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 50;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 100;
//         }

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_crop_mirror_normalize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_crop_mirror_normalize_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_crop_mirror_normalize_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_crop_mirror_normalize_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 39:
//     {
//         test_case_name = "resize_crop_mirror";

//         if (interpolationType != RpptInterpolationType::BILINEAR)
//         {
//             missingFuncFlag = 1;
//             break;
//         }

//         Rpp32u mirror[images];
//         for (i = 0; i < images; i++)
//         {
//             mirror[i] = 1;
//         }

//         for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
//             roiTensorPtrSrc[i].xywhROI.roiWidth = 50;
//             roiTensorPtrSrc[i].xywhROI.roiHeight = 50;
//         }

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);
//         hipMemcpy(d_dstImgSizes, dstImgSizes, images * sizeof(RpptImagePatch), hipMemcpyHostToDevice);

//         start = clock();
//         if (ip_bitDepth == 0)
//             rppt_resize_crop_mirror_gpu(d_input, srcDescPtr, d_output, dstDescPtr, d_dstImgSizes, interpolationType, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_resize_crop_mirror_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, d_dstImgSizes, interpolationType, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_resize_crop_mirror_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, d_dstImgSizes, interpolationType, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_resize_crop_mirror_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, d_dstImgSizes, interpolationType, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 40:
//     {
//         test_case_name = "erode";

//         Rpp32u kernelSize = additionalParam;

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_erode_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_erode_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_erode_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_erode_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 41:
//     {
//         test_case_name = "dilate";

//         Rpp32u kernelSize = additionalParam;

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_dilate_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_dilate_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_dilate_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_dilate_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 49:
//     {
//         test_case_name = "box_filter";

//         Rpp32u kernelSize = additionalParam;

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_box_filter_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_box_filter_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_box_filter_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 70:
//     {
//         test_case_name = "copy";

//         start = clock();
//         if (ip_bitDepth == 0)
//             rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
//         else if (ip_bitDepth == 1)
//             rppt_copy_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, handle);
//         else if (ip_bitDepth == 2)
//             rppt_copy_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_copy_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 80:
//     {
//         test_case_name = "resize_mirror_normalize";

//         if (interpolationType != RpptInterpolationType::BILINEAR)
//         {
//             missingFuncFlag = 1;
//             break;
//         }

//         for (i = 0; i < images; i++)
//         {
//             dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
//             dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
//         }

//         Rpp32f mean[images * 3];
//         Rpp32f stdDev[images * 3];
//         Rpp32u mirror[images];
//         for (i = 0, j = 0; i < images; i++, j += 3)
//         {
//             mean[j] = 60.0;
//             stdDev[j] = 1.0;

//             mean[j + 1] = 80.0;
//             stdDev[j + 1] = 1.0;

//             mean[j + 2] = 100.0;
//             stdDev[j + 2] = 1.0;
//             mirror[i] = 1;
//         }

//         // Uncomment to run test case with an xywhROI override
//         // for (i = 0; i < images; i++)
//         // {
//         //     roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//         //     roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//         //     dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//         //     dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         // }

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/

//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);
//         hipMemcpy(d_dstImgSizes, dstImgSizes, images * sizeof(RpptImagePatch), hipMemcpyHostToDevice);

//         start = clock();
//         if (ip_bitDepth == 0)
//             rppt_resize_mirror_normalize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, d_dstImgSizes, interpolationType, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_resize_mirror_normalize_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, d_dstImgSizes, interpolationType, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_resize_mirror_normalize_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, d_dstImgSizes, interpolationType, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             rppt_resize_mirror_normalize_gpu(d_input, srcDescPtr, d_outputf16, dstDescPtr, d_dstImgSizes, interpolationType, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 4)
//             rppt_resize_mirror_normalize_gpu(d_input, srcDescPtr, d_outputf32, dstDescPtr, d_dstImgSizes, interpolationType, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 5)
//             rppt_resize_mirror_normalize_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, d_dstImgSizes, interpolationType, mean, stdDev, mirror, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 83:
//     {
//         test_case_name = "gridmask";

//         Rpp32u tileWidth = 40;
//         Rpp32f gridRatio = 0.6;
//         Rpp32f gridAngle = 0.5;
//         RpptUintVector2D translateVector;
//         translateVector.x = 0.0;
//         translateVector.y = 0.0;

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/


//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_gridmask_gpu(d_input, srcDescPtr, d_output, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_gridmask_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_gridmask_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_gridmask_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 84:
//     {
//         test_case_name = "spatter";

//         RpptRGB spatterColor;

//         // Mud Spatter
//         spatterColor.R = 65;
//         spatterColor.G = 50;
//         spatterColor.B = 23;

//         // Blood Spatter
//         // spatterColor.R = 98;
//         // spatterColor.G = 3;
//         // spatterColor.B = 3;

//         // Ink Spatter
//         // spatterColor.R = 5;
//         // spatterColor.G = 20;
//         // spatterColor.B = 64;

//         // Uncomment to run test case with an xywhROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].xywhROI.xy.x = 0;
//             roiTensorPtrSrc[i].xywhROI.xy.y = 0;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
//         }*/

//         // Uncomment to run test case with an ltrbROI override
//         /*for (i = 0; i < images; i++)
//         {
//             roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
//             roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
//             roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
//             roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
//             dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
//             dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
//         }
//         roiTypeSrc = RpptRoiType::LTRB;
//         roiTypeDst = RpptRoiType::LTRB;*/


//         hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

//         start = clock();

//         if (ip_bitDepth == 0)
//             rppt_spatter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, spatterColor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 1)
//             rppt_spatter_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, spatterColor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 2)
//             rppt_spatter_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, spatterColor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_spatter_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, spatterColor, d_roiTensorPtrSrc, roiTypeSrc, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 85:
//     {
//         test_case_name = "swap_channels";

//         start = clock();
//         if (ip_bitDepth == 0)
//             rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
//         else if (ip_bitDepth == 1)
//             rppt_swap_channels_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, handle);
//         else if (ip_bitDepth == 2)
//             rppt_swap_channels_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_swap_channels_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     case 86:
//     {
//         test_case_name = "color_to_greyscale";

//         RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

//         start = clock();
//         if (ip_bitDepth == 0)
//             rppt_color_to_greyscale_gpu(d_input, srcDescPtr, d_output, dstDescPtr, srcSubpixelLayout, handle);
//         else if (ip_bitDepth == 1)
//             rppt_color_to_greyscale_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, srcSubpixelLayout, handle);
//         else if (ip_bitDepth == 2)
//             rppt_color_to_greyscale_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, srcSubpixelLayout, handle);
//         else if (ip_bitDepth == 3)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 4)
//             missingFuncFlag = 1;
//         else if (ip_bitDepth == 5)
//             rppt_color_to_greyscale_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, srcSubpixelLayout, handle);
//         else if (ip_bitDepth == 6)
//             missingFuncFlag = 1;
//         else
//             missingFuncFlag = 1;

//         break;
//     }
//     default:
//         missingFuncFlag = 1;
//         break;
//     }

//     hipDeviceSynchronize();
//     end = clock();

//     if (missingFuncFlag == 1)
//     {
//         printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
//         return -1;
//     }

//     // Display measured times

//     gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//     cout << "\nGPU Time - BatchPD : " << gpu_time_used << "s";
//     printf("\n");

//     // Reconvert other bit depths to 8u for output display purposes

//     string fileName = std::to_string(ip_bitDepth);
//     ofstream outputFile (fileName + ".csv");

//     if (ip_bitDepth == 0)
//     {
//         hipMemcpy(output, d_output, oBufferSizeInBytes_u8, hipMemcpyDeviceToHost);
//         Rpp8u *outputTemp;
//         outputTemp = output + dstDescPtr->offsetInBytes;

//         if (outputFile.is_open())
//         {
//             for (int i = 0; i < oBufferSize; i++)
//             {
//                 outputFile << (Rpp32u) *outputTemp << ",";
//                 outputTemp++;
//             }
//             outputFile.close();
//         }
//         else
//             cout << "Unable to open file!";
//     }
//     else if ((ip_bitDepth == 1) || (ip_bitDepth == 3))
//     {
//         hipMemcpy(outputf16, d_outputf16, oBufferSizeInBytes_f16, hipMemcpyDeviceToHost);
//         Rpp8u *outputTemp;
//         outputTemp = output + dstDescPtr->offsetInBytes;
//         half *outputf16Temp;
//         outputf16Temp = (half *)((Rpp8u *)outputf16 + dstDescPtr->offsetInBytes);

//         if (outputFile.is_open())
//         {
//             for (int i = 0; i < oBufferSize; i++)
//             {
//                 outputFile << (char) *outputf16Temp << ",";
//                 *outputTemp = (Rpp8u)RPPPIXELCHECK((float)*outputf16Temp * 255.0);
//                 outputf16Temp++;
//                 outputTemp++;
//             }
//             outputFile.close();
//         }
//         else
//             cout << "Unable to open file!";
//     }
//     else if ((ip_bitDepth == 2) || (ip_bitDepth == 4))
//     {
//         hipMemcpy(outputf32, d_outputf32, oBufferSizeInBytes_f32, hipMemcpyDeviceToHost);
//         Rpp8u *outputTemp;
//         outputTemp = output + dstDescPtr->offsetInBytes;
//         Rpp32f *outputf32Temp;
//         outputf32Temp = (Rpp32f *)((Rpp8u *)outputf32 + dstDescPtr->offsetInBytes);

//         if (outputFile.is_open())
//         {
//             for (int i = 0; i < oBufferSize; i++)
//             {
//                 outputFile << *outputf32Temp << ",";
//                 *outputTemp = (Rpp8u)RPPPIXELCHECK(*outputf32Temp * 255.0);
//                 outputf32Temp++;
//                 outputTemp++;
//             }
//             outputFile.close();
//         }
//         else
//             cout << "Unable to open file!";
//     }
//     else if ((ip_bitDepth == 5) || (ip_bitDepth == 6))
//     {
//         hipMemcpy(outputi8, d_outputi8, oBufferSizeInBytes_i8, hipMemcpyDeviceToHost);
//         Rpp8u *outputTemp;
//         outputTemp = output + dstDescPtr->offsetInBytes;
//         Rpp8s *outputi8Temp;
//         outputi8Temp = outputi8 + dstDescPtr->offsetInBytes;

//         if (outputFile.is_open())
//         {
//             for (int i = 0; i < oBufferSize; i++)
//             {
//                 outputFile << (Rpp32s) *outputi8Temp << ",";
//                 *outputTemp = (Rpp8u) RPPPIXELCHECK(((Rpp32s) *outputi8Temp) + 128);
//                 outputi8Temp++;
//                 outputTemp++;
//             }
//             outputFile.close();
//         }
//         else
//             cout << "Unable to open file!";
//     }

//     // Calculate exact dstROI in XYWH format for OpenCV dump

//     if (roiTypeSrc == RpptRoiType::LTRB)
//     {
//         for (int i = 0; i < dstDescPtr->n; i++)
//         {
//             int ltX = roiTensorPtrSrc[i].ltrbROI.lt.x;
//             int ltY = roiTensorPtrSrc[i].ltrbROI.lt.y;
//             int rbX = roiTensorPtrSrc[i].ltrbROI.rb.x;
//             int rbY = roiTensorPtrSrc[i].ltrbROI.rb.y;

//             roiTensorPtrSrc[i].xywhROI.xy.x = ltX;
//             roiTensorPtrSrc[i].xywhROI.xy.y = ltY;
//             roiTensorPtrSrc[i].xywhROI.roiWidth = rbX - ltX + 1;
//             roiTensorPtrSrc[i].xywhROI.roiHeight = rbY - ltY + 1;
//         }
//     }

//     RpptROI roiDefault;
//     RpptROIPtr roiPtrDefault;
//     roiPtrDefault = &roiDefault;
//     roiPtrDefault->xywhROI.xy.x = 0;
//     roiPtrDefault->xywhROI.xy.y = 0;
//     roiPtrDefault->xywhROI.roiWidth = dstDescPtr->w;
//     roiPtrDefault->xywhROI.roiHeight = dstDescPtr->h;

//     for (int i = 0; i < dstDescPtr->n; i++)
//     {
//         roiTensorPtrSrc[i].xywhROI.roiWidth = RPPMIN2(roiPtrDefault->xywhROI.roiWidth - roiTensorPtrSrc[i].xywhROI.xy.x, roiTensorPtrSrc[i].xywhROI.roiWidth);
//         roiTensorPtrSrc[i].xywhROI.roiHeight = RPPMIN2(roiPtrDefault->xywhROI.roiHeight - roiTensorPtrSrc[i].xywhROI.xy.y, roiTensorPtrSrc[i].xywhROI.roiHeight);
//         roiTensorPtrSrc[i].xywhROI.xy.x = RPPMAX2(roiPtrDefault->xywhROI.xy.x, roiTensorPtrSrc[i].xywhROI.xy.x);
//         roiTensorPtrSrc[i].xywhROI.xy.y = RPPMAX2(roiPtrDefault->xywhROI.xy.y, roiTensorPtrSrc[i].xywhROI.xy.y);
//     }

//     // Convert any PLN3 outputs to the corresponding PKD3 version for OpenCV dump

//     if ((dstDescPtr->c == 3) && (dstDescPtr->layout == RpptLayout::NCHW))
//     {
//         Rpp8u *outputCopy = (Rpp8u *)calloc(oBufferSizeInBytes_u8, 1);
//         memcpy(outputCopy, output, oBufferSizeInBytes_u8);

//         Rpp8u *outputTemp, *outputCopyTemp;
//         outputTemp = output + dstDescPtr->offsetInBytes;
//         outputCopyTemp = outputCopy + dstDescPtr->offsetInBytes;

//         for (int count = 0; count < dstDescPtr->n; count++)
//         {
//             Rpp8u *outputCopyTempR, *outputCopyTempG, *outputCopyTempB;
//             outputCopyTempR = outputCopyTemp;
//             outputCopyTempG = outputCopyTempR + dstDescPtr->strides.cStride;
//             outputCopyTempB = outputCopyTempG + dstDescPtr->strides.cStride;

//             for (int i = 0; i < dstDescPtr->h; i++)
//             {
//                 for (int j = 0; j < dstDescPtr->w; j++)
//                 {
//                     *outputTemp = *outputCopyTempR;
//                     outputTemp++;
//                     outputCopyTempR++;
//                     *outputTemp = *outputCopyTempG;
//                     outputTemp++;
//                     outputCopyTempG++;
//                     *outputTemp = *outputCopyTempB;
//                     outputTemp++;
//                     outputCopyTempB++;
//                 }
//             }

//             outputCopyTemp += dstDescPtr->strides.nStride;
//         }

//         free(outputCopy);
//     }

//     rppDestroyGPU(handle);

//     // OpenCV dump

//     mkdir(dst, 0700);
//     strcat(dst, "/");

//     count = 0;
//     Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;

//     Rpp8u *offsetted_output;
//     offsetted_output = output + dstDescPtr->offsetInBytes;

//     for (j = 0; j < dstDescPtr->n; j++)
//     {
//         int height = dstImgSizes[j].height;
//         int width = dstImgSizes[j].width;

//         int op_size = height * width * dstDescPtr->c;
//         Rpp8u *temp_output = (Rpp8u *)calloc(op_size, sizeof(Rpp8u));
//         Rpp8u *temp_output_row;
//         temp_output_row = temp_output;
//         Rpp32u elementsInRow = width * dstDescPtr->c;
//         Rpp8u *output_row = offsetted_output + count;

//         for (int k = 0; k < height; k++)
//         {
//             memcpy(temp_output_row, (output_row), elementsInRow * sizeof (Rpp8u));
//             temp_output_row += elementsInRow;
//             output_row += elementsInRowMax;
//         }
//         count += dstDescPtr->strides.nStride;

//         char temp[1000];
//         strcpy(temp, dst);
//         strcat(temp, imageNames[j]);

//         Mat mat_op_image;
//         mat_op_image = (pln1OutTypeCase) ? Mat(height, width, CV_8UC1, temp_output) : Mat(height, width, CV_8UC3, temp_output);
//         imwrite(temp, mat_op_image);

//         free(temp_output);
//     }

//     // Free memory

//     free(roiTensorPtrSrc);
//     free(roiTensorPtrDst);
//     hipFree(d_roiTensorPtrSrc);
//     hipFree(d_roiTensorPtrDst);
//     free(srcImgSizes);
//     free(dstImgSizes);
//     hipFree(d_srcImgSizes);
//     hipFree(d_dstImgSizes);
//     free(input);
//     free(input_second);
//     free(output);

//     if (ip_bitDepth == 0)
//     {
//         hipFree(d_input);
//         hipFree(d_input_second);
//         hipFree(d_output);
//     }
//     else if (ip_bitDepth == 1)
//     {
//         free(inputf16);
//         free(inputf16_second);
//         free(outputf16);
//         hipFree(d_inputf16);
//         hipFree(d_inputf16_second);
//         hipFree(d_outputf16);
//     }
//     else if (ip_bitDepth == 2)
//     {
//         free(inputf32);
//         free(inputf32_second);
//         free(outputf32);
//         hipFree(d_inputf32);
//         hipFree(d_inputf32_second);
//         hipFree(d_outputf32);
//     }
//     else if (ip_bitDepth == 3)
//     {
//         free(outputf16);
//         hipFree(d_input);
//         hipFree(d_input_second);
//         hipFree(d_outputf16);
//     }
//     else if (ip_bitDepth == 4)
//     {
//         free(outputf32);
//         hipFree(d_input);
//         hipFree(d_input_second);
//         hipFree(d_outputf32);
//     }
//     else if (ip_bitDepth == 5)
//     {
//         free(inputi8);
//         free(inputi8_second);
//         free(outputi8);
//         hipFree(d_inputi8);
//         hipFree(d_inputi8_second);
//         hipFree(d_outputi8);
//     }
//     else if (ip_bitDepth == 6)
//     {
//         free(outputi8);
//         hipFree(d_input);
//         hipFree(d_input_second);
//         hipFree(d_outputi8);
//     }

// }
//     return 0;