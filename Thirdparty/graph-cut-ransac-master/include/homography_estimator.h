#include "estimator.h"
#include "model.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Eigen>

class Homography : public Model
{
public:
	Homography() : 
		Model(Eigen::MatrixXd(3,3))
	{}

	Homography(const Homography& other)
	{
		descriptor = other.descriptor;
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class RobustHomographyEstimator : public theia::Estimator < cv::Mat, Model >
{
protected:

public:
	RobustHomographyEstimator() {}
	~RobustHomographyEstimator() {}

	size_t sampleSize() const {
		return 4;
	}

	size_t inlierLimit() const {
		return 7 * sampleSize();
	}

	bool estimateModel(
		const cv::Mat& data,
		const int *sample,
		std::vector<Model>* models) const
	{
		constexpr size_t M = 4;
		solverFourPoint(data,
			sample,
			M,
			models);
		return true;
	}

	bool estimateModelNonminimal(const cv::Mat& data,
		const int *sample,
		size_t sample_number,
		std::vector<Model>* models) const
	{
		if (sample_number < sampleSize())
			return false;
		
		cv::Mat normalized_points(sample_number, data.cols, data.type()); // The normalized point coordinates
		Eigen::Matrix3d T1, T2; // The normalizing transformations in the 1st and 2nd images

		// Normalize the point coordinates to achieve numerical stability when
		// applying the least-squares model fitting.
		if (!normalizePoints(data, // The data points
			sample, // The points to which the model will be fit
			sample_number, // The number of points
			normalized_points, // The normalized point coordinates
			T1, // The normalizing transformation in the first image
			T2)) // The normalizing transformation in the second image
			return false;

		// The four point fundamental matrix fitting algorithm
		solverFourPoint(normalized_points,
			nullptr,
			sample_number,
			models);

		// Denormalizing the estimated fundamental matrices
		const Eigen::Matrix3d T2_inverse = T2.inverse();
		for (auto &model : *models)
			model.descriptor = T2_inverse * model.descriptor * T1;
		return true;
	}

	double squaredResidual(const cv::Mat& point,
		const Model& model) const
	{
		return squaredResidual(point, model.descriptor);
	}

	inline double squaredResidual(const cv::Mat& point,
		const Eigen::MatrixXd& descriptor) const
	{
		const double* s = reinterpret_cast<double *>(point.data);

		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 2);
		const double y2 = *(s + 3);

		const double t1 = descriptor(0, 0) * x1 + descriptor(0, 1) * y1 + descriptor(0, 2);
		const double t2 = descriptor(1, 0) * x1 + descriptor(1, 1) * y1 + descriptor(1, 2);
		const double t3 = descriptor(2, 0) * x1 + descriptor(2, 1) * y1 + descriptor(2, 2);

		const double d1 = x2 - (t1 / t3);
		const double d2 = y2 - (t2 / t3);

		return d1 * d1 + d2 * d2;
	}

	double residual(const cv::Mat& point, 
		const Model& model) const
	{
		return residual(point, model.descriptor);
	}

	inline double residual(const cv::Mat& point,
		const Eigen::MatrixXd& descriptor) const
	{
		return sqrt(squaredResidual(point, descriptor));
	}

	inline bool normalizePoints(
		const cv::Mat& data, // The data points
		const int *sample, // The points to which the model will be fit
		size_t sample_number,// The number of points
		cv::Mat &normalized_points, // The normalized point coordinates
		Eigen::Matrix3d &T1, // The normalizing transformation in the first image
		Eigen::Matrix3d &T2) const // The normalizing transformation in the second image
	{
		const size_t cols = data.cols;
		double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points.data);
		const double *points_ptr = reinterpret_cast<double *>(data.data);

		double mass_point_src[2], // Mass point in the first image
			mass_point_dst[2]; // Mass point in the second image

		// Initializing the mass point coordinates
		mass_point_src[0] =
			mass_point_src[1] =
			mass_point_dst[0] =
			mass_point_dst[1] =
			0.0;

		// Calculating the mass points in both images
		for (size_t i = 0; i < sample_number; ++i)
		{
			// Get pointer of the current point
			const double *d_idx = points_ptr + cols * sample[i];

			// Add the coordinates to that of the mass points
			mass_point_src[0] += *(d_idx);
			mass_point_src[1] += *(d_idx + 1);
			mass_point_dst[0] += *(d_idx + 2);
			mass_point_dst[1] += *(d_idx + 3);
		}

		// Get the average
		mass_point_src[0] /= sample_number;
		mass_point_src[1] /= sample_number;
		mass_point_dst[0] /= sample_number;
		mass_point_dst[1] /= sample_number;

		// Get the mean distance from the mass points
		double average_distance_src = 0.0,
			average_distance_dst = 0.0;
		for (size_t i = 0; i < sample_number; ++i)
		{
			const double *d_idx = points_ptr + cols * sample[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 2);
			const double y2 = *(d_idx + 3);

			const double dx1 = mass_point_src[0] - x1;
			const double dy1 = mass_point_src[1] - y1;
			const double dx2 = mass_point_dst[0] - x2;
			const double dy2 = mass_point_dst[1] - y2;

			average_distance_src += sqrt(dx1 * dx1 + dy1 * dy1);
			average_distance_dst += sqrt(dx2 * dx2 + dy2 * dy2);
		}

		average_distance_src /= sample_number;
		average_distance_dst /= sample_number;

		// Calculate the sqrt(2) / MeanDistance ratios
		static const double sqrt_2 = sqrt(2);
		const double ratio_src = sqrt_2 / average_distance_src;
		const double ratio_dst = sqrt_2 / average_distance_dst;

		// Compute the normalized coordinates
		for (size_t i = 0; i < sample_number; ++i)
		{
			const double *d_idx = points_ptr + cols * sample[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 2);
			const double y2 = *(d_idx + 3);

			*normalized_points_ptr++ = (x1 - mass_point_src[0]) * ratio_src;
			*normalized_points_ptr++ = (y1 - mass_point_src[1]) * ratio_src;
			*normalized_points_ptr++ = (x2 - mass_point_dst[0]) * ratio_dst;
			*normalized_points_ptr++ = (y2 - mass_point_dst[1]) * ratio_dst;
		}

		// Creating the normalizing transformations
		T1 << ratio_src, 0, -ratio_src * mass_point_src[0],
			0, ratio_src, -ratio_src * mass_point_src[1],
			0, 0, 1;

		T2 << ratio_dst, 0, -ratio_dst * mass_point_dst[0],
			0, ratio_dst, -ratio_dst * mass_point_dst[1],
			0, 0, 1;
		return true;
	}

	inline bool solverFourPoint(
		const cv::Mat& data_,
		const int *sample_,
		const size_t sample_number_,
		std::vector<Model>* models_) const
	{
		constexpr size_t equation_number = 2;
		const size_t row_number = equation_number * sample_number_;
		Eigen::MatrixXd coefficients(row_number, 8);
		Eigen::MatrixXd inhomogeneous(row_number, 1);

		constexpr size_t columns = 4;
		const double *data_ptr = reinterpret_cast<double *>(data_.data);
		size_t row_idx = 0;
		
		for (size_t i = 0; i < sample_number_; ++i)
		{
			const double *point_ptr = sample_ == nullptr ?
				data_ptr + i * columns :
				data_ptr + sample_[i] * columns;

			const double x1 = point_ptr[0],
				y1 = point_ptr[1],
				x2 = point_ptr[2],
				y2 = point_ptr[3];
			
			coefficients(row_idx, 0) = -x1;
			coefficients(row_idx, 1) = -y1;
			coefficients(row_idx, 2) = -1;
			coefficients(row_idx, 3) = 0;
			coefficients(row_idx, 4) = 0;
			coefficients(row_idx, 5) = 0;
			coefficients(row_idx, 6) = x2 * x1;
			coefficients(row_idx, 7) = x2 * y1;
			inhomogeneous(row_idx) = -x2;
			++row_idx;

			coefficients(row_idx, 0) = 0;
			coefficients(row_idx, 1) = 0;
			coefficients(row_idx, 2) = 0;
			coefficients(row_idx, 3) = -x1;
			coefficients(row_idx, 4) = -y1;
			coefficients(row_idx, 5) = -1;
			coefficients(row_idx, 6) = y2 * x1;
			coefficients(row_idx, 7) = y2 * y1;
			inhomogeneous(row_idx) = -y2;
			++row_idx;
		}

		Eigen::Matrix<double, 8, 1> h = 
			coefficients.colPivHouseholderQr().solve(inhomogeneous);
		
		Homography model;
		model.descriptor << h(0), h(1), h(2), 
			h(3), h(4), h(5),
			h(6), h(7), 1.0;
		models_->emplace_back(model);
		return true;
	}

};