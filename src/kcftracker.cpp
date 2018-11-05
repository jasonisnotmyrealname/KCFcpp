/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#include "opencv2/core.hpp"
#include "colorname.h"
#endif

// Constructor
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab,bool cn)
{

    // Parameters equal in all cases
    lambda = 0.0001;  
    padding = 2.5; 
	output_sigma_factor = 0.125;
	d_valid = true;
	compression_learning_rate = 0.15;
	num_compressed_dim = 2;   //把colorname的通道由10压缩到2

    if (hog) {    // HOG
        // VOT
        interp_factor = 0.075;   //原来是0.012，但是在快速变换view angle时会失效
        sigma = 0.6; 
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5; 
        cell_size = 4;
        _hogfeatures = true;
		peak_value_valid_threshold = 0.4;
		peak_value_invalid_threshold = 0.3;
		re_scale = 3;
		bb_shift = 0.5;

        if (lab) {
            interp_factor = 0.005;
            sigma = 0.4; 
            //output_sigma_factor = 0.025;
            output_sigma_factor = 0.1;

            _labfeatures = true;
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);
            cell_sizeQ = cell_size*cell_size;
        }
        else{
            _labfeatures = false;
        }
    }
	else if (cn)
	{
		interp_factor = 0.075;
		sigma = 0.2;
		cell_size = 1;
		_cnfeatures = true;
	}
    else {   // RAW
        interp_factor = 0.075;
        sigma = 0.2; 
        cell_size = 1;
        _hogfeatures = false;

        if (lab) {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }


    if (multiscale) { // multiscale
        template_size = 96;
//		template_size = 1;
        //template_size = 100;
        scale_step = 1.05;
        scale_weight = 0.95;
        if (!fixed_window) {
            printf("Multiscale does not support non-fixed window.\n");
//            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
    }
    else {
        template_size = 1;
        scale_step = 1;
    }

	angle_step = 5;  //单位:degree
}

// Initialize tracker 
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;
	_angle = 0;
    assert(roi.width >= 0 && roi.height >= 0);

	//计算roi与预定模板尺寸之间的scale
	int padded_w = roi.width * padding;
	int padded_h = roi.height * padding;
	//将roi拉伸到预定模板的尺寸(长宽最大达到template_size)
	if (template_size > 1) {  // Fit largest dimension to the given template size
		if (padded_w >= padded_h)  //fit to width  选长宽中最大值求与template_size之间的scale
			_scale = padded_w / (float)template_size;
		else
			_scale = padded_h / (float)template_size;

		_tmpl_sz.width = padded_w / _scale;   //roi中长宽的最大值调整到template_size，另外一个按scale拉伸。即将padding后的roi拉伸到template的尺度
		_tmpl_sz.height = padded_h / _scale;
	}
	else {  //No template size given, use ROI size
		_tmpl_sz.width = padded_w;
		_tmpl_sz.height = padded_h;
		_scale = 1;
	}
	//如果使用hog特征，则需要调整模板尺寸为cell大小的偶数倍
	if (_hogfeatures) {
		// Round to cell size（用int强制转换，相当于求模） and also make it even（对2 * cell_size求模）
		_tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
		_tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
	}
	else {  //Make number of pixels even (helps with some logic involving half-dimensions)
		_tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
		_tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
	}

	_tmpl = getFeatures(image, _roi, _scale, 1,1.0f);   //1是指使能"inithann"

	//这句是为了获得y（x的标签）
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);  //对于raw特征:size_patch[0]指z的rows，size_patch[1]指的是z的cols
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0,true); // train with initial frame   //生成project_matrix,_alphaf
 }
 
// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat image)
{
	bool tvalid_temp=false;
	cv::Rect_<float> roi_temp;
	cv::Point2f res_temp;
	float scale_temp;
	float angle_temp;
	cv::Rect_<float> R;
    float max_response = 0.0f;  //用于latch所有ROI中的最大响应
	float cx;
	float cy;
	float peak_value =0;
    cv::Point2f res;
	cv::Mat a =image.clone();  //debug zjx
	unsigned int m = 0;
	max_response = 0;
	cv::Point2f new_res;
	cv::Mat image_r;
	_angle = 0;

	//越界处理
	if (_roi.x + _roi.width <= 0){
		_roi.x = -_roi.width + 1;
	}
	if (_roi.y + _roi.height <= 0){
		_roi.y = -_roi.height + 1;
	}
	if (_roi.x >= image.cols - 1){
		_roi.x = image.cols - 2;
	}
	if (_roi.y >= image.rows - 1){
		_roi.y = image.rows - 2;
	}

	//后面会更新_roi，这里暂时存储它的中心点坐标
	cx = _roi.x + _roi.width / 2.0f;
	cy = _roi.y + _roi.height / 2.0f;

	//如果刚发生drift，则在原roi的周围产生candidate detect region: candidate_ROI
	if (!d_valid)
	{
		//清空candidate_ROI
        std::vector < cv::Rect_<float> >().swap(candidate_ROI);
        rectangle(a, _roi, cv::Scalar(255, 255, 255), 10, 1);  //debug zjx
		int i_num = re_scale / bb_shift;
		int j_num = re_scale / bb_shift;
		//初始化candidate_ROI的成员R
		R.x = _roi.x - (_roi.width*re_scale / 2) + (_roi.width / 2);	  //将roi的x,y移到扩大后的左上角
		R.y = _roi.y - (_roi.height*re_scale / 2) + (_roi.height / 2);
//		cv::rectangle(image, cv::Rect(R.x, R.y, (_roi.width*re_scale), (_roi.height*re_scale)), (255, 0, 0), 8, 1);  //debug zjx
		R.width = _roi.width;
		R.height = _roi.height;
		//产生不同的R(相当于在roi的周围产生一组bb_shift的roi组，做重检测用)
		for (int i = 0; i < i_num; i++)
		{
			for (int j = 0; j < j_num; j++)
			{
				if ((R.x>0) && (R.y>0) && ((R.x + R.width) <image.size().width) && ((R.y + R.height) <image.size().height))
				{
					candidate_ROI.push_back(R);
                    rectangle(a, R, cv::Scalar(255, 255, 0),10, 1);  //debug zjx
//					rectangle(a, R, cv::Scalar(255, 255, 0), 2, 1);  //debug used. zjx
				}
				R.y += bb_shift*_roi.height;
			}
			R.x += bb_shift*_roi.width;
			R.y = _roi.y - (_roi.height*re_scale / 2);
		}
	}
	//如果没有发生Drift，则将_roi压入向量中
	else  
	{
        std::vector < cv::Rect_<float> >().swap(candidate_ROI);  //清空candidate_ROI
		candidate_ROI.push_back(_roi);
	}
	
#ifdef REDETECT
	d_valid = false;  //需要预先假设detect是失败的
#else
	d_valid = true;
#endif

	do{
		//读取candidate_ROI(可能是_roi，也可能是第一个R）
		R = candidate_ROI[m];
        rectangle(a, R, (255, 0, 0), 10, 1);  //debug zjx
//		cv::imshow("debug", a);   //debug zjx

		//下面是自适应Scale变化的代码，每次缩小0.95x，放大1.05x。假设目标尺度不会突变，这样的假设是合理的。
		//三个尺度检测:1x roi,0.95x roi,1.05x roi
		res_temp = detect(_tmpl, getFeatures(image, R, _scale, 0, 1.0f), peak_value);    //detect后返回的坐标(res坐标）是内插过的浮点数字，且是以center为坐标原点的（为了下面的缩放）		
		scale_temp = _scale;
		angle_temp = _angle;
		roi_temp.width = R.width;
		roi_temp.height = R.height;
		float new_peak_value;
		if (scale_step != 1) {
			// Test at a smaller _scale
			new_res = detect(_tmpl, getFeatures(image, R, _scale, 0, 1.0f / scale_step), new_peak_value);  //将extractor_roi拉伸1.0f / scale_step（实际是缩放0.95x），再detect一遍

			if (new_peak_value > peak_value) {   //如果缩放的效果更好，则更新scale
				res_temp = new_res;
				peak_value = new_peak_value;
				scale_temp /= scale_step;
				roi_temp.width /= scale_step;
				roi_temp.height /= scale_step;
			}

			// Test at a bigger _scale
//			new_res = detect(_tmpl, getFeatures(image, cv::Rect_<float>(R.x, R.y, roi_temp.width, roi_temp.height), _scale, 0, scale_step), new_peak_value);
			new_res = detect(_tmpl, getFeatures(image, R, _scale, 0, scale_step), new_peak_value);

			if (new_peak_value > peak_value) {
				res_temp = new_res;
				peak_value = new_peak_value;
				scale_temp *= scale_step;
				roi_temp.width *= scale_step;
				roi_temp.height *= scale_step;
			}
		}

		//两个旋转角度检测
		/*
		if (angle_step != 0)
		{
			cv::Mat H;
			//clockwise matrix
			H = cv::getRotationMatrix2D(cv::Point2f(R.x + roi_temp.width / 2, R.y + roi_temp.height/2),angle_step,1);
			cv::warpAffine(image, image_r, H, image.size());
			H = getFeatures(image_r, cv::Rect_<float>(R.x, R.y, roi_temp.width, roi_temp.height), scale_temp, 0, 1.0f);  //zjx debug
			new_res = detect(_tmpl,H , new_peak_value);

			if (new_peak_value > peak_value) { 
				res_temp = new_res;
				peak_value = new_peak_value;
				angle_temp = angle_step;
				std::cout << "angle: " << angle_temp << std::endl;
			}

			H = cv::getRotationMatrix2D(cv::Point2f(R.x + roi_temp.width / 2, R.y + roi_temp.height / 2), -angle_step, 1);
			cv::warpAffine(image, image_r, H, image.size());
			H = getFeatures(image_r, cv::Rect_<float>(R.x, R.y, roi_temp.width, roi_temp.height), scale_temp, 0, 1.0f);   //zjx debug
			new_res = detect(_tmpl,H , new_peak_value);

			if (new_peak_value > peak_value) { 
				res_temp = new_res;
				peak_value = new_peak_value;
				angle_temp = -angle_step;
				std::cout << "angle: " << angle_temp << std::endl;
			}

		}*/

		// 增加阈值判断
		if ((peak_value >peak_value_valid_threshold) || (d_valid))
		{
			//在所有R中求出response最大的R
			if (peak_value> max_response)
			{
				max_response = peak_value;
				res = res_temp;
				//计算R的中心坐标，此时R作为候选的roi区域，由于尺度搜索，导致R的width和不同尺度的width并不相同，因此这里先确定R的中心坐标(cx,cy)，而width由经过尺度搜索的roi_temp确定
				cx = R.x + R.width / 2.0f;
				cy = R.y + R.height / 2.0f;
				_roi.width = roi_temp.width;
				_roi.height = roi_temp.height;
				_scale = scale_temp;
				_angle = angle_temp;
				d_valid = true;
			}
		}
		m++;
	} while (m < candidate_ROI.size());
	
    // Adjust by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);
    rectangle(a, _roi, (255, 0, 0), 10, 1);  //debug zjx

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

    assert(_roi.width >= 0 && _roi.height >= 0);

	//更新旋转角度
	/*
	cv::Point2f *pts = new cv::Point2f;
	cv::RotatedRect roi_rotated = cv::RotatedRect(cv::Point2f(_roi.x + _roi.width / 2, _roi.y + _roi.height / 2), _roi.size(), _angle);
	roi_rotated.points(pts);


	if (_angle != 0)
	{
		float px_min = image.size().width;
		float px_max = 0;
		float py_min = image.size().height;
		float py_max = 0;
//		_roi = roi_rotated.boundingRect();   //这个地方有问题，即使angle==0，这个函数返回的rect也比原来的要大
		for (int i = 0; i < 4; i++)
		{
			if (pts[i].x <px_min)
			{
				px_min = pts[i].x;
			}
			if (pts[i].x >px_max)
			{
				px_max = pts[i].x;
			}
			if (pts[i].y <py_min)
			{
				py_min = pts[i].y;
			}
			if (pts[i].y >py_max)
			{
				py_max = pts[i].y;
			}
		}
		_roi.width = px_max - px_min;
		_roi.height = py_max - py_min;
		_roi.x = px_min;
		_roi.y = py_min;
	}*/
	if (d_valid)  //更新模板x和alphaf
	{
		getFeatures(image, R, _scale, 0, 1.0f)
		cv::Mat x = getFeatures(image, _roi, _scale, 0, 1);  //0表示更新时
		if (_angle != 0)
		{
			train(x, 0.4,false);
		}
		else
		{
			train(x, interp_factor, false);
		}
	}
    return _roi;
}


// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
    using namespace FFTTools;
	cv::Mat x_pro;
	cv::Mat z_pro;
	if (_hogfeatures)
	{
		x_pro = x.clone();
		z_pro = z.clone();
		x_pro = x_pro.mul(hann);
		z_pro = z_pro.mul(hann);
	}
	else if (_cnfeatures)
	{
		x_pro = projection_matrix.t()*x;
		z_pro = projection_matrix.t()*z;
		//mul.cos
//		x_pro = x_pro.reshape(num_compressed_dim, size_patch[0] * size_patch[1]);
//		z_pro = z_pro.reshape(num_compressed_dim, size_patch[0] * size_patch[1]);
		x_pro = x_pro.mul(hann);
		z_pro = z_pro.mul(hann);
	}
	else
	{
		x_pro = x.clone();
		z_pro = z.clone();
		x_pro = x_pro.mul(hann);
		z_pro = z_pro.mul(hann);
	}	

	cv::Mat k = gaussianCorrelation(x_pro, z_pro);
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float) pv;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols-1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1), peak_value, res.at<float>(pi.y, pi.x+1));
    }

    if (pi.y > 0 && pi.y < res.rows-1) {
        p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x), peak_value, res.at<float>(pi.y+1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p;
}

// train tracker with a single image
// 20161215增加：增加SVD求解projection_matrix
void KCFTracker::train(cv::Mat x, float train_interp_factor, const bool inithann)
{
    using namespace FFTTools;
	cv::Mat x_pro;
	//更新z(_tmpl)
	if (inithann)
	{
		_tmpl = x;
	}
	else
	{
		_tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)* x;
	}
	
	// cn特征:根据z更新project_matrix
	if (_cnfeatures)
	{
		if (inithann) {
			// compute the mean appearance
			data_matrix = cv::Mat::zeros(_tmpl.rows, _tmpl.cols, CV_32FC1);
		}
		// compute the mean appearance
		reduce(_tmpl, data_mean, 0, CV_REDUCE_AVG);
		// substract the mean from the appearance to get the data matrix
		float*data = ((float*)data_matrix.data);
		for (int i = 0; i < _tmpl.rows; i++)
		{
			memcpy(data + i * _tmpl.cols, ((cv::Mat)(_tmpl.row(i) - data_mean)).data, _tmpl.cols * sizeof(float));
		}
		// calculate the covariance matrix
		cov_matrix = (1.0 / (_tmpl_sz.width * _tmpl_sz.height - 1))* (data_matrix * data_matrix.t());
		//---------------------------对z_pca的协方差矩阵进行PCA降维---------------------------//
		// calculate the principal components (pca_basis) and corresponding variances
		if (inithann)
		{
			cv::Mat vt;
			cv::SVD::compute(cov_matrix, pca_variances, pca_basis, vt);
		}
		else
		{
			cv::Mat vt;
			cv::SVD::compute((1 - compression_learning_rate) * old_cov_matrix + compression_learning_rate * cov_matrix,
				pca_variances, pca_basis, vt);
		}

		// calculate the projection matrix as the first principal
		// components and extract their corresponding variances
		projection_matrix = pca_basis(cv::Rect(0, 0, num_compressed_dim, pca_basis.rows)).clone();    //降维：提取pca_basis矩阵的前num_compressed_dim个奇异值作为投影矩阵projection_matrix
		cv::Mat projection_variances = cv::Mat::zeros(num_compressed_dim, num_compressed_dim, CV_32FC1);
		for (int i = 0; i < num_compressed_dim; i++)
		{
			((float*)projection_variances.data)[i + i*num_compressed_dim] = ((float*)pca_variances.data)[i];   //抽取pca_variances的前num_compressed_dim个对角元素
		}

		//-------------------------更新old_cov_matrix------------------------------------------//
		if (inithann)
		{
			// initialize the old covariance matrix using the computed
			// projection matrix and variances
			old_cov_matrix = projection_matrix * projection_variances * projection_matrix.t();
		}
		else
		{
			// update the old covariance matrix using the computed
			// projection matrix and variances
			old_cov_matrix =
				(1 - compression_learning_rate) * old_cov_matrix +
				compression_learning_rate * (projection_matrix * projection_variances * projection_matrix.t());
		}
	}

	if (_cnfeatures) //cn特征:根据projection_matrix,对x进行降维
	{
		x_pro = projection_matrix.t()*x;
	}
	else
	{
		x_pro = x.clone();
	}

	//更新alphaf
	x_pro = x_pro.mul(hann);
	cv::Mat k = gaussianCorrelation(x_pro, x_pro);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));  
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;



    /*cv::Mat kf = fftd(gaussianCorrelation(x, x));
    cv::Mat num = complexMultiplication(kf, _prob);
    cv::Mat den = complexMultiplication(kf, kf + lambda);
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
    _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

    _alphaf = complexDivision(_num, _den);*/

}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat( cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0) );
    // HOG features(分channel计算，再求和）
    if (_hogfeatures) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++) {   //size_patch[2]指channel
            x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);  //reshape(channels,rows)
            x2aux = x2.row(i).reshape(1, size_patch[0]);
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true); 
            caux = fftd(caux, true);
            rearrange(caux);
            caux.convertTo(caux,CV_32F);
            c = c + real(caux);
        }
    }
	else if(_cnfeatures)
	{
		cv::Mat caux;
		cv::Mat x1aux;
		cv::Mat x2aux;
		for (int i = 0; i < num_compressed_dim; i++) {   //size_patch[2]指channel
			x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
			x1aux = x1aux.reshape(1, size_patch[0]);  //reshape(channels,rows)
			x2aux = x2.row(i).reshape(1, size_patch[0]);
			cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
			caux = fftd(caux, true);
			rearrange(caux);
			caux.convertTo(caux, CV_32F);
			c = c + real(caux);
		}
	}
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);  //表示IFFT
        rearrange(c);
        c = real(c);
    }
    cv::Mat d; 
    cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0])- 2. * c) / (size_patch[0]*size_patch[1]*size_patch[2]) , 0, d);

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
    return k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);  //??????

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

/* Convert BGR to ColorNames
*/
void extractCN(cv::Mat patch_data, cv::Mat & cnFeatures) {
	cv::Vec3b & pixel = patch_data.at<cv::Vec3b>(0, 0);
	unsigned index;

	//cnFeatures的大小和patch_data一样
	if (cnFeatures.type() != CV_32FC(10))
		cnFeatures = cv::Mat::zeros(patch_data.rows, patch_data.cols, CV_32FC(10));

	for (int i = 0; i<patch_data.rows; i++){
		for (int j = 0; j<patch_data.cols; j++){
			pixel = patch_data.at<cv::Vec3b>(i, j);
			index = (unsigned)(floor((float)pixel[2] / 8) + 32 * floor((float)pixel[1] / 8) + 32 * 32 * floor((float)pixel[0] / 8));

			//copy the values
			for (int _k = 0; _k<10; _k++){
				cnFeatures.at<cv::Vec<float, 10> >(i, j)[_k] = ColorNames[index][_k];
			}
		}
	}

}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat KCFTracker::getFeatures(const cv::Mat & image, const cv::Rect_<float> roi, const float scale, const bool inithann, const float scale_adjust)
{
    cv::Rect_<float> extracted_roi;
	//roi的坐标转到center
	float cx = roi.x + roi.width / 2;
	float cy = roi.y + roi.height / 2;

	extracted_roi.width = scale_adjust * scale * _tmpl_sz.width;   //之前用padded_w/_scale得到_tmpl_sz.width，再经过对cell_size的偶数倍处理，这里再乘_scale，恢复到原来的roiscale。但是乘_scale后，又不是cell_size的倍数了。为什么不直接对padded_w对2*cell_size求模?
    extracted_roi.height = scale_adjust * scale * _tmpl_sz.height;

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    cv::Mat FeaturesMap;  
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);   //按照extracted_roi来从image中抽取padding后的roi做为z
//	cv::Mat z1 = RectTools::subwindow(image, _roi, cv::BORDER_REPLICATE);  //作用和下一句相同
//	cv::Mat z2 = cv::Mat(image, roi); //zjx debug
    
    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {   //_scale不为1的情况下，这句肯定是满足的
        cv::resize(z, z, _tmpl_sz);  //将z重新resize到template的size（是2*cell_size的倍数)
    }   //这段代码隐含的思想是:模板tmpl的尺寸在初始化时就固定了，在跟踪的过程中不变，这样导致视角发生变化时，模板不能随着视角变化而变化

    // HOG features
    if (_hogfeatures) {
        IplImage z_ipl = z;
        CvLSVMFeatureMapCaskade *map;   //以cell为单位的特征矢量结构体，维度是bin
        getFeatureMaps(&z_ipl, cell_size, &map);
        normalizeAndTruncate(map,0.2f);
        PCAFeatureMaps(map);
        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;  //共31个features

        FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug  channel变成1，宽度就是维度，高度就是roi.witdh*roi.height
        FeaturesMap = FeaturesMap.t();   //求mat转置
        freeFeatureMapObject(&map);

        // Lab features
        if (_labfeatures) {
            cv::Mat imgLab;
            cvtColor(z, imgLab, CV_BGR2Lab);
            unsigned char *input = (unsigned char*)(imgLab.data);

            // Sparse output vector
            cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0]*size_patch[1], CV_32F, float(0));

            int cntCell = 0;
            // Iterate through each cell
            for (int cY = cell_size; cY < z.rows-cell_size; cY+=cell_size){
                for (int cX = cell_size; cX < z.cols-cell_size; cX+=cell_size){
                    // Iterate through each pixel of cell (cX,cY)
                    for(int y = cY; y < cY+cell_size; ++y){
                        for(int x = cX; x < cX+cell_size; ++x){
                            // Lab components for each pixel
                            float l = (float)input[(z.cols * y + x) * 3];
                            float a = (float)input[(z.cols * y + x) * 3 + 1];
                            float b = (float)input[(z.cols * y + x) * 3 + 2];

                            // Iterate trough each centroid
                            float minDist = FLT_MAX;
                            int minIdx = 0;
                            float *inputCentroid = (float*)(_labCentroids.data);
                            for(int k = 0; k < _labCentroids.rows; ++k){
                                float dist = ( (l - inputCentroid[3*k]) * (l - inputCentroid[3*k]) )
                                           + ( (a - inputCentroid[3*k+1]) * (a - inputCentroid[3*k+1]) ) 
                                           + ( (b - inputCentroid[3*k+2]) * (b - inputCentroid[3*k+2]) );
                                if(dist < minDist){
                                    minDist = dist;
                                    minIdx = k;
                                }
                            }
                            // Store result at output
                            outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ; 
                            //((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ; 
                        }
                    }
                    cntCell++;
                }
            }
            // Update size_patch[2] and add features to FeaturesMap
            size_patch[2] += _labCentroids.rows;
            FeaturesMap.push_back(outputLab);
        }
    }
	else if (_cnfeatures)
	{
		CV_Assert(z.channels() == 3);
		extractCN(z, FeaturesMap);
		size_patch[0] = FeaturesMap.rows;   //和z一样
		size_patch[1] = FeaturesMap.cols;   //和z一样
		size_patch[2] = FeaturesMap.channels();   //ColorName有10通道
		FeaturesMap = FeaturesMap.reshape(1, size_patch[0] * size_patch[1]); // Procedure do deal with cv::Mat multichannel bug  channel变成1，宽度就是维度，高度就是roi.witdh*roi.height
		FeaturesMap = FeaturesMap.t();   //求mat转置
//		FeaturesMap = (FeaturesMap * projection_matrix);
	}
    else {   //Gray特征,通道数量为1
        FeaturesMap = RectTools::getGrayImage(z);
        FeaturesMap -= (float) 0.5; // In Paper;  范围变为(-0.5,0.5)
        size_patch[0] = z.rows;
        size_patch[1] = z.cols;
        size_patch[2] = 1;  
    }
    
    if (inithann) {
        createHanningMats();  //Hanning Window和size_patch有关，而size_patch与特征类型、模板大小、padding大小有关
    }  //输出的hann是一个2维矩阵（不管HOG\CN特征是多少维）
//    FeaturesMap = hann.mul(FeaturesMap);
    return FeaturesMap;
}
    
// Initialize Hanning window. Function called only in the first frame.
void KCFTracker::createHanningMats()
{   
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[0]), CV_32F, cv::Scalar(0)); 

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
	if (_hogfeatures) {
        cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug
        
        hann = cv::Mat(cv::Size(size_patch[0]*size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++) {
            for (int j = 0; j<size_patch[0]*size_patch[1]; j++) {
                hann.at<float>(i,j) = hann1d.at<float>(0,j);
            }
        }
    }
	else if (_cnfeatures)  //PCA
	{
		cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

		hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], num_compressed_dim), CV_32F, cv::Scalar(0));
		for (int i = 0; i < num_compressed_dim; i++) {
			for (int j = 0; j<size_patch[0] * size_patch[1]; j++) {
				hann.at<float>(i, j) = hann1d.at<float>(0, j);
			}
		}
	}
    // Gray features 
    else {
        hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{   
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;
    
    return 0.5 * (right - left) / divisor;
}
