/*
* This code requires opencv2 and CUDA. It was configured with versions
*  4.9.0 and 12.3 respectivly, and run on Windows 10 in Visual Studio 17 (2022).
*
* Written by Raymond Adkins in March, 2024. contact: raymond.adkins@yale.edu.
*
*
*/

#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cufft.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cuComplex.h>

#include <nvrtc.h>

#include <chrono>

using namespace std;
using namespace cv;



vector<Mat> RL_3Ddeconv(const vector<cuda::GpuMat> H_real, const vector<cuda::GpuMat> H_imag, const vector<cuda::GpuMat> Ht_real, const vector<cuda::GpuMat> Ht_imag, const cuda::GpuMat img, const vector<cuda::GpuMat> rec_vol_0, const cufftHandle forward, const cufftHandle reverse, const int d, const int N, const int n_iter)
{

    // I'd guess it would be better to do all these definitions elsewhere
    vector<Mat> output_rec_vol(d);

    cuda::GpuMat temp_mat(N, N, CV_32FC1), fft_rec_vol_slice(N, N / 2 + 1, CV_32FC2), im_error(N, N, CV_32FC1);
    
    Rect topLeftRect(0, 0, N / 2, N / 2);
    Rect topRightRect(N / 2, 0, N / 2, N / 2);
    Rect bottomLeftRect(0, N / 2, N / 2, N / 2);
    Rect bottomRightRect(N / 2, N / 2, N / 2, N / 2);

    vector<cuda::GpuMat> rec_vol(d), tmp_rec_vol(d), rec_vol_err(d);

    for (int s = 0; s < d; s++) {
        rec_vol[s] = rec_vol_0[s];
        output_rec_vol[s] = Mat(N, N, CV_32FC1);
        tmp_rec_vol[s].create(N, N, CV_32FC1);
        rec_vol_err[s].create(N, N, CV_32FC1);
    }

    cuda::GpuMat proj_im(N, N, CV_32FC1),TopLeft(N, N, CV_32FC1), TopRight(N, N, CV_32FC1), BottomLeft(N, N, CV_32FC1), BottomRight(N, N, CV_32FC1);
    cuda::GpuMat m1n1(N, N / 2 + 1, CV_32FC2), m2n2(N, N / 2 + 1, CV_32FC2), m2n1(N, N / 2 + 1, CV_32FC2), m1n2(N, N / 2 + 1, CV_32FC2);
    cuda::GpuMat fft_multiply(N, N / 2 + 1, CV_32FC2), proj_im_comp(N, N, CV_32FC1), mask1(N, N, CV_32FC1), mask2(N, N, CV_32FC1);
    cuda::GpuMat fft_im_error(N, N / 2 + 1, CV_32FC2), fft_multiply2(N, N / 2 + 1, CV_32FC2);

    cuda::GpuMat split[2], Mul[2];

    

    for (int it = 0; it < n_iter; it++){    
        
        for (int s = 0; s < d; s++){
            // Fourier transform of rec_vol
            
            cufftExecR2C(forward, (cufftReal*)rec_vol[s].data, (cufftComplex*)fft_rec_vol_slice.data);            
            
            cv::cuda::split(fft_rec_vol_slice, split);

            cv::cuda::multiply(H_real[s], split[0], m1n1); // This portion takes the most time
            cv::cuda::multiply(H_imag[s], split[0], m2n1);
            cv::cuda::multiply(H_real[s], split[1], m1n2);
            cv::cuda::multiply(H_imag[s], split[1], m2n2);
            
            cv::cuda::subtract(m1n1, m2n2, Mul[0]);
            cv::cuda::add(m2n1, m1n2, Mul[1]);

            
            cv::cuda::merge(Mul, 2, fft_multiply);

            // Inverse Fourier transform to complete the convolution
            
            cufftExecC2R(reverse, (cufftComplex*)fft_multiply.data, (cufftReal*)proj_im_comp.data);

            cuda::add(proj_im,proj_im_comp,proj_im);

            fft_multiply.release();
            
        }
        
        cuda::divide(proj_im, N*N, proj_im);
        
        cuda::divide(img, proj_im, im_error);

        //Mask where proj_im==0
        cuda::compare(proj_im, cv::Scalar(0.0), mask1, cv::CMP_EQ);
        // Set values in im_error to zero where mask is zero.
        im_error.setTo(0, mask1);

        //Mask where im_error<0
        cuda::compare(im_error, cv::Scalar(0.0), mask2, cv::CMP_LT);
        // Set values in im_error to zero where mask is zero.
        im_error.setTo(0, mask2);

        // Fourier transform of im_error
        cufftExecR2C(forward, (cufftReal*)im_error.data, (cufftComplex*)fft_im_error.data);
        
        
        cv::cuda::split(fft_im_error, split);

        
        for (int s = 0; s < d; s++) {

            // Multiply Fourier transform

            
            cv::cuda::multiply(Ht_real[s], split[0], m1n1); // This portion takes the most time
            cv::cuda::multiply(Ht_imag[s], split[0], m2n1);
            cv::cuda::multiply(Ht_real[s], split[1], m1n2);
            cv::cuda::multiply(Ht_imag[s], split[1], m2n2);
            
            cv::cuda::subtract(m1n1, m2n2, Mul[0]);
            cv::cuda::add(m2n1, m1n2, Mul[1]);

            
            cv::cuda::merge(Mul, 2, fft_multiply2);
            // Inverse Fourier transform to complete the convolution
            
            cufftExecC2R(reverse, (cufftComplex*)fft_multiply2.data, (cufftReal*)rec_vol_err[s].data);

            cuda::divide(rec_vol_err[s], N * N, rec_vol_err[s]);

            // FFTshift on rec_vol_error
            
            //Cut image into quadrants
            TopLeft = rec_vol_err[s](topLeftRect);
            TopRight = rec_vol_err[s](topRightRect);
            BottomLeft = rec_vol_err[s](bottomLeftRect);
            BottomRight = rec_vol_err[s](bottomRightRect);

            
            // Reorder the four quadrants to make the correct image
            BottomRight.copyTo(temp_mat(topLeftRect));
            BottomLeft.copyTo(temp_mat(topRightRect));
            TopRight.copyTo(temp_mat(bottomLeftRect));
            TopLeft.copyTo(temp_mat(bottomRightRect));

            // Multiply images together 
            cuda::multiply(rec_vol[s], temp_mat, rec_vol[s]);
            cuda::abs(tmp_rec_vol[s], tmp_rec_vol[s]); //Not clear if we need this
            fft_multiply2.release();
        }


        
    }

    // Download results from GPU
    for (int s = 0; s < d; s++) {
        rec_vol[s].download(output_rec_vol[s]);
    }

    
    return output_rec_vol;
}



int main()
{

    // -----------------------------------------
    //
    // Section 1: Variables defined by user
    //
    // -----------------------------------------

    std::string filename_PSF = "../../Standard dataset/Exp_PSF.tif";


    //THESE CAN BOTH BE MEASURED FROM THE IMAGE
    int d = 41;
    int N = 2048;

    int N_iter = 10; // Number of iterations for RL deconvolution


    // -----------------------------------------
    //
    // Section 2: Initialize for computation
    //
    // -----------------------------------------

    // Initialize cuFFT plans
    cufftHandle forward, reverse;
    cufftPlan2d(&forward, N, N, CUFFT_R2C); // Forward transform
    cufftPlan2d(&reverse, N, N, CUFFT_C2R); // Reverse transform

    // Initialize CUDA
    cuda::setDevice(0); // Choose GPU device index (default is 0)

    // Initialize vol_0
    vector<cuda::GpuMat> vol_0(d);
    Mat reconst_slice(N, N, CV_32F, Scalar(1.0));
    for (int s = 0; s < d; s++) {
        vol_0[s].upload(reconst_slice);
    }

    // Make all objects
    Mat PSF_r_temp;
    cuda::GpuMat temp1, temp1_f, temp2, temp2_f;
    vector<Mat> PSF;
    vector<cuda::GpuMat> PSF_gpu_FT_real(d), PSF_r_gpu_FT_real(d), PSF_gpu_FT_imag(d), PSF_r_gpu_FT_imag(d);

    // Set up for saving image
    //ALL THIS CAN BE MEASURED AS THE CENTER LENS POSITION
    int av_radius = 230;
    Point cent(1000, 980);
    Point circCent(av_radius, av_radius);
    Rect cropRect(cent.x - av_radius, cent.y - av_radius, 2 * av_radius, 2 * av_radius);
    Mat1b mask(2 * av_radius, 2 * av_radius, double(0.0)); 
    circle(mask, circCent, av_radius, Scalar(1.0), FILLED);

    // -----------------------------------------
    //
    // Section 2: Load in the PSF, and prepare for analysis
    //
    // -----------------------------------------

    // Load in H
    bool successH = imreadmulti(filename_PSF, PSF, IMREAD_UNCHANGED);
    if (!successH) {
        std::cerr << "Error loading PSF" << std::endl;
        return -1;
    }
    else {
        std::cout << "Loaded PSF" << std::endl;
    }

    // Upload PSFs to GPU and convert to F32
    for (int s = 0; s < d; s++) {

        temp1.upload(PSF[s]);
        temp1.convertTo(temp1_f, CV_32FC1);
        cuda::divide(temp1_f, N * N, temp1_f);

        // Rotate PSF by 180
        cv::rotate(PSF[s], PSF_r_temp, 1); 
        temp2.upload(PSF_r_temp);
        temp2.convertTo(temp2_f, CV_32FC1);
        cuda::divide(temp2_f, N * N, temp2_f);

    
        cuda::GpuMat PSF_FT_temp(N, N / 2 + 1, CV_32FC2), PSF_split_temp[2], PSF_r_FT_temp(N, N / 2 + 1, CV_32FC2), PSF_r_split_temp[2];

        // FFT of H (The Optical Transfer Function)
        cufftExecR2C(forward, (cufftReal*)temp1_f.data, (cufftComplex*)PSF_FT_temp.data);
        cuda::split(PSF_FT_temp, PSF_split_temp);
        PSF_gpu_FT_real[s] = PSF_split_temp[0];
        PSF_gpu_FT_imag[s] = PSF_split_temp[1];

        // FFT of Ht
        cufftExecR2C(forward, (cufftReal*)temp2_f.data, (cufftComplex*)PSF_r_FT_temp.data);
        cuda::split(PSF_r_FT_temp, PSF_r_split_temp);
        PSF_r_gpu_FT_real[s] = PSF_r_split_temp[0];
        PSF_r_gpu_FT_imag[s] = PSF_r_split_temp[1];
    }

    // -----------------------------------------
    //
    // Section 3: Load in light field image
    //
    // -----------------------------------------

    // Open LF image
    std::string filename_LF = "../../Standard dataset/LFImage.tif"; // BUG: If image isn't normalized, information abotu intensity is lost on conversion during import
    Mat img = cv::imread(filename_LF);
    if (img.empty()) {
        std::cerr << "Error loading" << filename_LF << std::endl; 
        return -1;
    }

    

    // Upload the light field image to GPU
    cuda::GpuMat img_gpu;
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    img_gpu.upload(img);
    img_gpu.convertTo(img_gpu, CV_32FC1, 1.0/255);
    

    // -----------------------------------------
    //
    // Section 4: Call 3D deconvolution
    //
    // -----------------------------------------


    // Preform RL devonvolution step
    auto tstart_l2 = std::chrono::steady_clock::now();

    vector<Mat> reconst_volume = RL_3Ddeconv(PSF_gpu_FT_real, PSF_gpu_FT_imag, PSF_r_gpu_FT_real, PSF_r_gpu_FT_imag, img_gpu, vol_0, forward, reverse, d, N, N_iter);
    
    auto elapsed_L2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tstart_l2);
    std::cout << "Reconstruction  took " << elapsed_L2.count() << " millisecond (ms)." << endl;
    // -----------------------------------------
    //
    // Section 5: Crop and export 3D images
    //
    // -----------------------------------------
     
    // Save the results

    vector<Mat> cropped_reconst_volume;
    for (int s = 0; s < d; s++) {
        Mat aux_mat = reconst_volume[s](cropRect);
        Mat masked_aux_mat;
        aux_mat.copyTo(masked_aux_mat, mask);

        cropped_reconst_volume.push_back(masked_aux_mat);
    }

    cv::imwrite("../../Standard dataset/ReconstructedVolumes/recon_OurCode.tif", cropped_reconst_volume);

    // Destroy all cuFFT plans
    cufftDestroy(forward);
    cufftDestroy(reverse);

    return 0;
}
