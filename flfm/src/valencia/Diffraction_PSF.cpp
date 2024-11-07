/*LICENSE NOTE
* /** 
 * Copyright (C) 2024 Nicolo Incardona - All Rights Reserved
 *
 * This code is shared upon agreement, for its use for academic purpose only.
 * Commercial use in source and binary forms, with or without modifications, is strictly prohibited.
 * Unauthorized sharing in source and binary forms, with or without modifications, via any medium is strictly prohibited.
 * 
 * When using this code, with or without modifications, or any other implementations based on it, please reference the following citation:
 * Incardona, N., Tolosa, A., Saavedra, G., Martinez-Corral, M., & Sanchez-Ortiga, E. (2023). 
 * "Fast and robust wave optics-based reconstruction protocol for Fourier lightfield microscopy". Optics and Lasers in Engineering, 161, 107336.
 * 
 *
 * Proprietary and confidential
 * Written by {Nicolo Incardona} <{nicolo.incardona@uv.es}>, {04/02/2024}
 */



//// LICENSE OF CODE FOR MICROLENS PSF GENERATION
//// Source: https://www.optinav.info/download/Diffraction_PSF_3D.java
//// The part of the code corresponding to Microlens PSF generation and belonging to OptiNav, Inc. is indicated
//
/*	License:
	Copyright (c) 2005, OptiNav, Inc.
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions
	are met:

		Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.
		Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.
		Neither the name of OptiNav, Inc. nor the names of its contributors
	may be used to endorse or promote products derived from this software
	without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
	"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
	LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
	A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
	CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
	EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
	PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <functional>
#include <sstream>

using namespace cv;
using namespace std;

////////////////////////
//////////////////////// CODE FOR MICROLENS PSF GENERATION (start)
////////////////////////
static std::vector<double> t = {
	1,
		-2.2499997,
		1.2656208,
		-0.3163866,
		0.0444479,
		-0.0039444,
		0.0002100};
static std::vector<double> p = {
	-.78539816,
		-.04166397,
		-.00003954,
		0.00262573,
		-.00054125,
		-.00029333,
		.00013558};
static std::vector<double> f = {
	.79788456,
		-0.00000077,
		-.00552740,
		-.00009512,
		0.00137237,
		-0.00072805,
		0.00014476};


float interp(float* y, float x) {
	int i = (int)x;
	float fract = x - i;
	return (1 - fract) * y[i] + fract * y[i + 1];
}


//Bessel function J0(x).  Uses the polynomial approximations on p. 369-70 of Abramowitz & Stegun
//The error in J0 is supposed to be less than or equal to 5 x 10^-8.
double J0(double xIn) {
	double x = xIn;
	if (x < 0) x *= -1;
	double r;
	if (x <= 3) {
		double y = x * x / 9;
		r = t[0] + y * (t[1] + y * (t[2] + y * (t[3] + y * (t[4] + y * (t[5] + y * t[6])))));
	}
	else {
		double y = 3 / x;
		double theta0 = x + p[0] + y * (p[1] + y * (p[2] + y * (p[3] + y * (p[4] + y * (p[5] + y * p[6])))));
		double f0 = f[0] + y * (f[1] + y * (f[2] + y * (f[3] + y * (f[4] + y * (f[5] + y * f[6])))));
		r = sqrt(1 / x) * f0 * cos(theta0);
	}
	return r;
}
////////////////////////
//////////////////////// CODE FOR MICROLENS PSF GENERATION (end)
////////////////////////


int av_radius;								// average radius of the EIs
float av_pitch;								// average pitch between EIs
Point2i c_center;							// coordinates of central EI
vector<Point2i> centers;					// coordinates of EIs
vector<Point2f> norm_distances;				// normalized distances


// Sort EIs from top-left to bottom-right
void SortCircles()
{
	Point temp;
	for (int i = 0; i < centers.size(); i++)
	{
		for (int j = centers.size() - 2; j >= i; j--)
		{
			if (centers[j].y > centers[j + 1].y)
			{
				temp = centers[j];
				centers[j] = centers[j + 1];
				centers[j + 1] = temp;
			}
		}
	}

	for (int i = 0; i < centers.size(); i++)
	{
		for (int j = centers.size() - 2; j >= i; j--)
		{
			if (centers[j].x > centers[j + 1].x)
			{
				if (abs(centers[j].y - centers[j + 1].y) < av_radius / 2)
				{
					temp = centers[j];
					centers[j] = centers[j + 1];
					centers[j + 1] = temp;
				}
			}
		}
	}

}

// Get coordinates from calibration file
void GetCoordinates(std::string calib_file)
{
	int p; bool x = true;
	Point2i c;

	ifstream file;
	file.open(calib_file);
	if (!file)
	{
		cout << "Unable to open file";
		exit(1); // terminate with error
	}
	file >> av_radius;
	while (file >> p)
	{
		if (x)
			c.x = p;
		else
			c.y = p;
		x ^= 1;
		if (x)
			centers.push_back(c);
	}
	file.close();

	c_center.x = centers[0].x;
	c_center.y = centers[0].y;

	SortCircles();

	// Calculate average pitch
	av_pitch = 0;
	int N = 0;
	for (int i = 0; i < centers.size() - 1; i++)
	{
		if (abs(centers[i].y - centers[i + 1].y) < av_radius && abs(centers[i].x - centers[i + 1].x) < 4 * av_radius)
		{
			av_pitch += sqrt(pow((centers[i].x - centers[i + 1].x), 2) + pow((centers[i].y - centers[i + 1].y), 2));
			N++;
		}
	}
	av_pitch = av_pitch / N;

	// Calculate normalized distances from the central view
	for (int i = 0; i < centers.size(); i++)
	{
		Point2f d;
		d.x = float(centers[i].x - centers[centers.size() / 2].x) / float(av_pitch);
		d.y = float(centers[i].y - centers[centers.size() / 2].y) / float(av_pitch);
		norm_distances.push_back(d);
	}
}

//Forward-projection
Mat forwProj(vector<Mat> rec_vol, vector<Mat> LF_PSF_mat)
{
	Mat proj_im(rec_vol.at(0).size(), CV_32F, Scalar(0.0));

	Ptr<cuda::Convolution> convolver = cuda::createConvolution(proj_im.size());

	int vertical_pad = (proj_im.rows - 1) / 2;
	int horizontal_pad = (proj_im.cols - 1) / 2;

	for (int s = 0; s < LF_PSF_mat.size(); s++){
		Mat aux;
		cuda::GpuMat gpu_rec_slice, gpu_LF_PSF_slice, gpu_aux, gpu_padded_rec_slice;
		gpu_rec_slice.upload(rec_vol.at(s));
		cuda::copyMakeBorder(gpu_rec_slice, gpu_padded_rec_slice, vertical_pad, vertical_pad, horizontal_pad, horizontal_pad, BORDER_CONSTANT, 0.0);
		gpu_LF_PSF_slice.upload(LF_PSF_mat.at(s));
		//filter2D(rec_vol.at(s), aux, -1, LF_PSF_mat.at(s), Point(-1, -1), 0, BORDER_CONSTANT);
		convolver->convolve(gpu_padded_rec_slice, gpu_LF_PSF_slice, gpu_aux, true);
		gpu_aux.download(aux);
		proj_im += aux;
	}

	return proj_im;
}

// Back-projection
vector<Mat> backProj(Mat err, vector<Mat> LF_PSF_mat_inv)
{
	vector<Mat> rec_vol_err;

	Ptr<cuda::Convolution> convolver = cuda::createConvolution(err.size());

	int vertical_pad = (err.rows - 1) / 2;
	int horizontal_pad = (err.cols - 1) / 2;

	for (int s = 0; s < LF_PSF_mat_inv.size(); s++) {
		Mat aux;
		cuda::GpuMat gpu_err, gpu_LF_PSF_inv_slice, gpu_aux, gpu_padded_err;
		gpu_err.upload(err);
		cuda::copyMakeBorder(gpu_err, gpu_padded_err, vertical_pad, vertical_pad, horizontal_pad, horizontal_pad, BORDER_CONSTANT, 0.0);
		gpu_LF_PSF_inv_slice.upload(LF_PSF_mat_inv.at(s));
		//filter2D(err, aux, -1, LF_PSF_mat_inv.at(s), Point(-1, -1), 0, BORDER_CONSTANT);
		convolver->convolve(gpu_padded_err, gpu_LF_PSF_inv_slice, gpu_aux, true);
		gpu_aux.download(aux);
		rec_vol_err.push_back(aux);
		
		aux.release();
	}

	return rec_vol_err;
}

// Richardson-Lucy 3D-to-2D deconvolution
vector<Mat> RL_3Ddeconv(Mat img, vector<Mat> LF_PSF_mat, vector<Mat> rec_vol, int n_iter)
{
	int d = LF_PSF_mat.size();
	vector<Mat> LF_PSF_mat_inv;
	double vals_sum = 0.0;
	for (int s = 0; s < d; s++) {
		Mat aux;
		LF_PSF_mat.at(s).copyTo(aux);
		flip(aux, aux, -1);
		LF_PSF_mat_inv.push_back(aux);
		vals_sum += sum(aux)[0];
	}

	vector<Mat> LF_PSF_mat_inv_norm;
	for (int s = 0; s < d; s++) {
		Mat tmp;
		tmp = LF_PSF_mat_inv.at(s) / vals_sum;
		LF_PSF_mat_inv_norm.push_back(tmp);
	}

	for (int it = 0; it < n_iter; it++)
	{
		printf("Richardson-Lucy Deconvolution: iteration n. %i\n", it + 1);

		Mat proj_im = forwProj(rec_vol, LF_PSF_mat);

		Mat im_error;
		im_error = img / proj_im;
		im_error.setTo(0, (proj_im == 0));
		im_error.setTo(0, (im_error < 0));
		
		vector<Mat> rec_vol_err = backProj(im_error, LF_PSF_mat_inv_norm);

		vector<Mat> tmp_rec_vol;

		for (int s = 0; s < d; s++) {
			Mat aux;
			multiply(rec_vol.at(s), rec_vol_err.at(s), aux);
			tmp_rec_vol.push_back(aux);
		}

		rec_vol.clear();
		rec_vol = tmp_rec_vol;
	}

	return rec_vol;
}


int main(int argc, char** argv)
{
	////////////////////////
	//////////////////////// CODE FOR MICROLENS PSF GENERATION (start)
	////////////////////////
	//

	//Data files.
	// TODO: Make these args of main().
    std::string filename_PSF = "data/valencia/generated_psf.tif";
    std::string filename_LF = "data/valencia/light_field_image.tif";
    std::string filename_reconstructed_image = "data/valencia/reconstructed_image.tif";
    std::string filename_calibration = "data/valencia/calibration.txt";

    bool generate_psf = true;
    if (argc > 1) {
        std::istringstream(argv[1]) >> std::boolalpha >> generate_psf;
    }
    if (generate_psf) {
        std::cout << "Generating PSF." << std::endl;
    } else {
        std::cout << "Not generating PSF, reading from file instead." << std::endl;
    }

	// Parameters of capture
	double lambda = 555.0;						// illumination wavelength [nm]
	double indexRefr = 1.333;					// refractive index of medium
	double pixelSpacing = 1792;					// in object space [nm]
	double sliceSpacing = 5000;					// in object space [nm]
	double na = 0.1333;							// effective NA
	double sa = 0.0;
	int w = 455;								// width of one EI [pixels]
	int h = 455;								// height of one EI [pixels]
	int d = 41;									// number of reconstruction planes

	int stepsPerCycle = 8;
	int normalization = 2;
	bool dB = false;

	int ic = w / 2;
	int jc = h / 2;
	int kc = d / 2;

	auto tstart_l1 = std::chrono::steady_clock::now();

    // Open LF image to deconvolve and convert to greyscale
    cv::Mat img = cv::imread(filename_LF);

    int row = img.rows - 1;          // Row to delete.
    int col = img.cols - 1;          // Column to delete.
    cv::Rect rect(0, 0, col, row);
    img(rect).copyTo(img);
    cvtColor(img, img, COLOR_BGR2GRAY);


	// Microlens PSF creation
	vector<Mat> LF_psf_mat;

    // LF PSF creation
    GetCoordinates(filename_calibration);
    av_pitch = w;

	if (generate_psf) {
        float a = (float)(2 * 3.1415 * na / lambda);
        double dRing = 0.6 * lambda / (pixelSpacing * na);

        auto pixels = new float[d][455*455];

        int rMax = 2 + (int)sqrt(ic * ic + jc * jc);
        float* integral = new float[rMax];
        double upperLimit = tan(asin(na / indexRefr));
        double waveNumber = 2 * 3.1415 * indexRefr / lambda;
        for (int k = 0; k < d; k++) {
            double kz = waveNumber * ((double)k - (double)kc) * sliceSpacing;
            for (int r = 0; r < rMax; r++) {
                double kr = waveNumber * r * pixelSpacing;
                int numCyclesJ = 1 + (int)(kr * upperLimit / 3);
                int numCyclesCos = 1 + (int)(abs(kz) * 0.36 * upperLimit / 6);
                int numCycles = numCyclesJ;
                if (numCyclesCos > numCycles)numCycles = numCyclesCos;
                int nStep = 2 * stepsPerCycle * numCycles;
                int m = nStep / 2;
                double step = upperLimit / nStep;
                double sumR = 0;
                double sumI = 0;
                //Simpson's rule
                //Assume that the sperical aberration varies with the  (% aperture)^4
                //f(a) = f(0) = 0, so no contribution
                double u = 0;
                double bessel = 1;
                double root = 1;
                double angle = kz;
                //2j terms
                for (int j = 1; j < m; j++) {
                    u = 2 * (double)j * step;
                    kz = waveNumber * (((double)k - (double)kc) * sliceSpacing +
                        sa * (u / upperLimit) * (u / upperLimit) * (u / upperLimit) * (u / upperLimit));
                    root = sqrt(1 + u * u);
                    bessel = J0(kr * u / root);
                    angle = kz / root;
                    sumR += 2 * cos(angle) * u * bessel / 2;
                    sumI += 2 * sin(angle) * u * bessel / 2;
                }

                //2j - 1 terms
                for (int j = 1; j <= m; j++) {
                    u = (2 * (double)j - 1) * step;
                    kz = waveNumber * (((double)k - (double)kc) * sliceSpacing +
                        sa * (u / upperLimit) * (u / upperLimit) * (u / upperLimit) * (u / upperLimit));
                    root = sqrt(1 + u * u);
                    bessel = J0(kr * u / root);
                    angle = kz / root;
                    sumR += 4 * cos(angle) * u * bessel / 2;
                    sumI += 4 * sin(angle) * u * bessel / 2;
                }

                //f(b)
                u = upperLimit;
                kz = waveNumber * (((double)k - (double)kc) * sliceSpacing + sa);
                root = sqrt(1 + u * u);
                bessel = J0(kr * u / root);
                angle = kz / root;
                sumR += cos(angle) * u * bessel / 2;
                sumI += sin(angle) * u * bessel / 2;

                integral[r] = (float)(step * step * (sumR * sumR + sumI * sumI) / 9);
            }
            double uSlices = ((double)k - (double)kc);
            for (int j = 0; j < h; j++) {
                for (int i = 0; i < w; i++) {
                    double rPixels = sqrt((i - ic) * (i - ic) + (j - jc) * (j - jc));
                    pixels[k][i + w * j] = interp(integral, (float)rPixels);
                    //pixels[kSym][i + w*j] = interp(integral,(float)rPixels);
                }
            }
        }
        int n = w * h;

        if (normalization == 2) {
            float peak = pixels[kc][ic + w * jc];
            for (int k = 0; k < d; k++) {
                for (int ind = 0; ind < n; ind++) {
                    if (pixels[k][ind] > peak)
                        peak = pixels[k][ind];
                }
            }
            float f = 255 / peak;
            for (int k = 0; k < d; k++) {
                for (int ind = 0; ind < n; ind++) {
                    pixels[k][ind] *= f;
                    if (pixels[k][ind] > 255)pixels[k][ind] = 255;
                }
            }
        }
        //
        ////////////////////////
        //////////////////////// CODE FOR MICROLENS PSF GENERATION (end)
        ////////////////////////


        std::vector<cv::Mat> PSF_mat;
        cv::Mat psf_slice(455, 455, CV_8U);
        for (int s = 0; s < d; s++) {
            uchar* temp = new uchar[455 * 455];
            for (int i = 0; i < 455 * 455; i++) {
                temp[i] = (uchar)pixels[s][i];
            }
            psf_slice.data = temp;
            PSF_mat.push_back(psf_slice);
        }

        int ov_samp = 5;

        double vals_sum;
        //d = d - 6;
        for (int s = 0; s < d; s++) {
            Mat interp_PSF_slice;
            resize(PSF_mat.at(s), interp_PSF_slice, PSF_mat.at(s).size() * ov_samp, 0, 0, INTER_NEAREST);
            Mat LF_psf_slice(img.size() * ov_samp, CV_8U, Scalar(0));
            for (int cc = 0; cc < centers.size(); cc++)
            {
                interp_PSF_slice.copyTo(LF_psf_slice.colRange(centers[cc].x * ov_samp - int(floor(av_pitch / 2.0 * (float)ov_samp)) + int(round((s - kc) * norm_distances[cc].x * ov_samp)), centers[cc].x * ov_samp + int(floor(av_pitch / 2.0 * (float)ov_samp + 1)) + int(round((s - kc) * norm_distances[cc].x * ov_samp))).rowRange(centers[cc].y * ov_samp - int(floor(av_pitch / 2.0 * (float)ov_samp)) + int(round((s - kc) * norm_distances[cc].y * ov_samp)), centers[cc].y * ov_samp + int(floor(av_pitch / 2.0 * (float)ov_samp)) + int(round((s - kc) * norm_distances[cc].y * ov_samp)) + 1));
            }
            resize(LF_psf_slice, LF_psf_slice, img.size(), 0, 0, INTER_CUBIC);

            LF_psf_slice.convertTo(LF_psf_slice, CV_32F);
            LF_psf_slice *= 1.0 / 255;
            flip(LF_psf_slice, LF_psf_slice, -1);
            vals_sum = sum(LF_psf_slice)[0];
            LF_psf_slice = LF_psf_slice / vals_sum;
            LF_psf_mat.push_back(LF_psf_slice);
        }

        // Write out generated PSF.
        imwrite(filename_PSF, LF_psf_mat);
    } else {
       bool successH = imreadmulti(filename_PSF, LF_psf_mat, IMREAD_UNCHANGED);
        if (!successH) {
            std::cerr << "Error loading PSF" << std::endl;
            return -1;
        }
        else {
            std::cout << "Loaded PSF" << std::endl;
        }
    }


	img.convertTo(img, CV_32F);
	img *= 1.0 / 255;
	vector<Mat> vol_0;
	Mat reconst_slice(img.size(), CV_32F, Scalar(1.0));
	for (int s = 0; s < d; s++) {
		vol_0.push_back(reconst_slice);
	}

	// Call 3D deconvolution
	vector<Mat> reconst_volume;

	auto tstart_l2 = std::chrono::steady_clock::now();

	reconst_volume = RL_3Ddeconv(img, LF_psf_mat, vol_0, 10);

	auto elapsed_L2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tstart_l2);
	std::cout << "Reconstruction  took " << elapsed_L2.count() << " millisecond (ms)." << endl;
	

	auto elapsed_L1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tstart_l1);
	std::cout << "Reconstruction plus PSF generation took " << elapsed_L1.count() << " millisecond (ms)." << endl;

	// Save the results
	vector<Mat> cropped_reconst_vol;
	double max_val = 0;
	for (int s = 0; s < d; s++) {
		Mat aux_mat;
		reconst_volume.at(s).colRange(img.size().width / 2 - int(floor(av_pitch / 2.0)), img.size().width / 2 + int(floor(av_pitch / 2.0) + 1)).rowRange(img.size().height / 2 - int(floor(av_pitch / 2.0)), img.size().height / 2 + int(floor(av_pitch / 2.0) + 1)).copyTo(aux_mat);
		Mat masked_aux_mat;
		Mat1b mask(aux_mat.size(), double(0.0));
		Point cent(aux_mat.size().width / 2, aux_mat.size().height / 2);
		circle(mask, cent, av_radius, Scalar(1.0), FILLED);
		aux_mat.copyTo(masked_aux_mat, mask);
		cropped_reconst_vol.push_back(masked_aux_mat);
		double min, max;
		cv::minMaxLoc(masked_aux_mat, &min, &max);
		if (max > max_val)
			max_val = max;
		//normalize(aux_mat, aux_mat, 0, 1, NORM_MINMAX, -1, Mat());
		//reconst_volume.at(s).copyTo(aux_mat);
		/*aux_mat *= 255;
		aux_mat.convertTo(aux_mat, CV_8U);
		stringstream str;
		str << "Rec_plane_" << s + 1 << ".png";
		imwrite(str.str(), aux_mat);*/
	}
	//for (int s = 0; s < d; s++) {
	//	Mat aux_mat;
	//	aux_mat = cropped_reconst_vol.at(s) / max_val;
	//	aux_mat *= 255;
	//	aux_mat.convertTo(aux_mat, CV_8U);
	//	stringstream str;
	//	str << "Rec_plane_" << s + 1 << ".png";
	//	imwrite(str.str(), aux_mat);
	//}

	imwrite(filename_reconstructed_image, cropped_reconst_vol);
}
