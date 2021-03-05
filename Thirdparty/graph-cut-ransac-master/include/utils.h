#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Eigen>

/*
	Function declaration
*/
void drawMatches(
	const cv::Mat &points_,
	const std::vector<int> &inliers_,
	const cv::Mat &image1_,
	const cv::Mat &image2_,
	cv::Mat &out_image_,
	int circle_radius_ = 5);

bool savePointsToFile(
	const cv::Mat &points_,
	const char* file_,
	const std::vector<int> *inliers_ = NULL);

bool loadPointsFromFile(
	cv::Mat &points_,
	const char* file_);

void detectFeatures(
	std::string name_,
	cv::Mat image1_,
	cv::Mat image2_,
	cv::Mat &points_);

void showImage(
	const cv::Mat &image_,
	std::string window_name_,
	int max_width_,
	int max_height_,
	bool wait_);

template<typename T, int N, int M>
bool loadMatrix(const std::string &path_,
	Eigen::Matrix<T, N, M> &matrix_);

void normalizeCorrespondences(const cv::Mat &points_,
	const Eigen::Matrix3d &intrinsics_src_,
	const Eigen::Matrix3d &intrinsics_dst_,
	cv::Mat &normalized_points_);

/*
	Function definition
*/

void drawMatches(
	const cv::Mat &points_,
	const std::vector<int> &inliers_,
	const cv::Mat &image_src_,
	const cv::Mat &image_dst_,
	cv::Mat &out_image_,
	int circle_radius_)
{	
	// Final image
	out_image_.create(image_src_.rows, // Height
		2 * image_src_.cols, // Width
		image_src_.type()); // Type

	cv::Mat roi_img_result_left = 
		out_image_(cv::Rect(0, 0, image_src_.cols, image_src_.rows)); // Img1 will be on the left part
	cv::Mat roi_img_result_right =
		out_image_(cv::Rect(image_src_.cols, 0, image_dst_.cols, image_dst_.rows)); // Img2 will be on the right part, we shift the roi of img1.cols on the right

	cv::Mat roi_image_src = image_src_(cv::Rect(0, 0, image_src_.cols, image_src_.rows));
	cv::Mat roi_image_dst = image_dst_(cv::Rect(0, 0, image_dst_.cols, image_dst_.rows));

	roi_image_src.copyTo(roi_img_result_left); //Img1 will be on the left of imgResult
	roi_image_dst.copyTo(roi_img_result_right); //Img2 will be on the right of imgResult

	for (const auto &idx : inliers_)
	{
		cv::Point2d pt1(points_.at<double>(idx, 0), 
			points_.at<double>(idx, 1));
		cv::Point2d pt2(image_dst_.cols + points_.at<double>(idx, 2),
			points_.at<double>(idx, 3));

		cv::Scalar color(255 * static_cast<double>(rand()) / RAND_MAX, 
			255 * static_cast<double>(rand()) / RAND_MAX, 
			255 * static_cast<double>(rand()) / RAND_MAX);

		cv::circle(out_image_, pt1, circle_radius_, color, static_cast<int>(circle_radius_ * 0.4));
		cv::circle(out_image_, pt2, circle_radius_, color, static_cast<int>(circle_radius_ * 0.4));
		cv::line(out_image_, pt1, pt2, color, 2);
	}
}

void detectFeatures(std::string scene_name_,
	cv::Mat image1_,
	cv::Mat image2_,
	cv::Mat &points_)
{
  
}

/*
void detectFeatures(std::string scene_name_,
	cv::Mat image1_,
	cv::Mat image2_,
	cv::Mat &points_)
{
  
	if (loadPointsFromFile(points_,
		scene_name_.c_str()))
	{
		printf("Match number: %d\n", points_.rows);
		return;
	}

	printf("Detect AKAZE features\n");
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;

	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
	detector->detect(image1_, keypoints1);
	detector->compute(image1_, keypoints1, descriptors1);
	printf("Features found in the first image: %d\n", static_cast<int>(keypoints1.size()));

	detector->detect(image2_, keypoints2);
	detector->compute(image2_, keypoints2, descriptors2);
	printf("Features found in the second image: %d\n", static_cast<int>(keypoints2.size()));

	cv::BFMatcher matcher(cv::NORM_HAMMING);
	std::vector< std::vector< cv::DMatch >> matches_vector;
	matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);

	std::vector<std::tuple<double, cv::Point2d, cv::Point2d>> correspondences;
	for (auto match : matches_vector)
	{
		if (match.size() == 2 && match[0].distance < match[1].distance * 0.8)
		{
			auto& kp1 = keypoints1[match[0].queryIdx];
			auto& kp2 = keypoints2[match[0].trainIdx];
			correspondences.emplace_back(std::make_tuple<double, cv::Point2d, cv::Point2d>(match[0].distance / match[1].distance, (cv::Point2d)kp1.pt, (cv::Point2d)kp2.pt));
		}
	}

	// Sort the points for PROSAC
	std::sort(correspondences.begin(), correspondences.end(), [](const std::tuple<double, cv::Point2d, cv::Point2d>& correspondence_1_,
		const std::tuple<double, cv::Point2d, cv::Point2d>& correspondence_2_) -> bool
	{
		return std::get<0>(correspondence_1_) < std::get<0>(correspondence_2_);
	});

	points_ = cv::Mat(static_cast<int>(correspondences.size()), 4, CV_64F);
	double *points_ptr = reinterpret_cast<double*>(points_.data);

	for (auto[distance_ratio, point_1, point_2] : correspondences)
	{
		*(points_ptr++) = point_1.x;
		*(points_ptr++) = point_1.y;
		*(points_ptr++) = point_2.x;
		*(points_ptr++) = point_2.y;
	}
	
	savePointsToFile(points_, scene_name_.c_str());
	printf("Match number: %d\n", static_cast<int>(points_.rows));
}
*/

bool loadPointsFromFile(cv::Mat &points,
	const char* file)
{
	std::ifstream infile(file);

	if (!infile.is_open())
		return false;

	int N;
	std::string line;
	int line_idx = 0;
	double *points_ptr = NULL;

	while (getline(infile, line))
	{
		if (line_idx++ == 0)
		{
			N = atoi(line.c_str());
			points = cv::Mat(N, 4, CV_64F);
			points_ptr = reinterpret_cast<double*>(points.data);
			continue;
		}

		std::istringstream split(line);
		split >> *(points_ptr++);
		split >> *(points_ptr++);
		split >> *(points_ptr++);
		split >> *(points_ptr++);
	}

	infile.close();
	return true;
}

bool savePointsToFile(const cv::Mat &points, 
	const char* file, 
	const std::vector<int> *inliers)
{
	std::ofstream outfile(file, std::ios::out);

	double *points_ptr = reinterpret_cast<double*>(points.data);
	const int M = points.cols;

	if (inliers == NULL)
	{
		outfile << points.rows << std::endl;
		for (auto i = 0; i < points.rows; ++i)
		{
			for (auto j = 0; j < M; ++j)
				outfile << *(points_ptr++) << " ";
			outfile << std::endl;
		}
	}
	else
	{
		outfile << inliers->size() << std::endl;
		for (size_t i = 0; i < inliers->size(); ++i)
		{
			const int offset = inliers->at(i) * M;
			for (auto j = 0; j < M; ++j)
				outfile << *(points_ptr + offset + j) << " ";
			outfile << std::endl;
		}
	}

	outfile.close();

	return true;
}

template<typename T, int N, int M>
bool loadMatrix(const std::string &path_,
	Eigen::Matrix<T, N, M> &matrix_)
{
	std::ifstream infile(path_);

	if (!infile.is_open())
		return false;

	size_t row = 0,
		column = 0;
	double element;

	while (infile >> element)
	{
		matrix_(row, column) = element;
		++column;
		if (column >= M)
		{
			column = 0;
			++row;
		}
	}

	infile.close();

	return row == N &&
		column == 0;
}

void showImage(const cv::Mat &image_,
	std::string window_name_,
	int max_width_,
	int max_height_,
	bool wait_)
{
	// Resizing the window to fit into the screen if needed
	int window_width = image_.cols,
		window_height = image_.rows;
	if (static_cast<double>(image_.cols) / max_width_ > 1.0 &&
		static_cast<double>(image_.cols) / max_width_ >
		static_cast<double>(image_.rows) / max_height_)
	{
		window_width = max_width_;
		window_height = static_cast<int>(window_width * static_cast<double>(image_.rows) / static_cast<double>(image_.cols));
	}
	else if (static_cast<double>(image_.rows) / max_height_ > 1.0 &&
		static_cast<double>(image_.cols) / max_width_ <
		static_cast<double>(image_.rows) / max_height_)
	{
		window_height = max_height_;
		window_width = static_cast<int>(window_height * static_cast<double>(image_.cols) / static_cast<double>(image_.rows));
	}

	cv::namedWindow(window_name_, CV_WINDOW_NORMAL);
	cv::resizeWindow(window_name_, window_width, window_height);
	cv::imshow(window_name_, image_);
	if (wait_)  cv::waitKey(1);
}

void normalizeCorrespondences(const cv::Mat &points_,
	const Eigen::Matrix3d &intrinsics_src_,
	const Eigen::Matrix3d &intrinsics_dst_,
	cv::Mat &normalized_points_)
{
	const double *points_ptr = reinterpret_cast<double *>(points_.data);
	double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points_.data);
	const Eigen::Matrix3d inverse_intrinsics_src = intrinsics_src_.inverse(),
		inverse_intrinsics_dst = intrinsics_dst_.inverse();

	// Most likely, this is not the fastest solution, but it does
	// not affect the speed of Graph-cut RANSAC, so not a crucial part of
	// this example.
	double x0, y0, x1, y1;
	for (auto r = 0; r < points_.rows; ++r)
	{
		Eigen::Vector3d point_src,
			point_dst,
			normalized_point_src,
			normalized_point_dst;

		x0 = *(points_ptr++);
		y0 = *(points_ptr++);
		x1 = *(points_ptr++);
		y1 = *(points_ptr++);

		point_src << x0, y0, 1.0; // Homogeneous point in the first image
		point_dst << x1, y1, 1.0; // Homogeneous point in the second image
		
		// Normalized homogeneous point in the first image
		normalized_point_src = 
			inverse_intrinsics_src * point_src;
		// Normalized homogeneous point in the second image
		normalized_point_dst = 
			inverse_intrinsics_dst * point_dst;

		// The second four columns contain the normalized coordinates.
		*(normalized_points_ptr++) = normalized_point_src(0);
		*(normalized_points_ptr++) = normalized_point_src(1);
		*(normalized_points_ptr++) = normalized_point_dst(0);
		*(normalized_points_ptr++) = normalized_point_dst(1);
	}
}
