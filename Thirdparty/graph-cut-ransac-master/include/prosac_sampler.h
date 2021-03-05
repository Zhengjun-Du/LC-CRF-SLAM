// Copyright (C) 2013 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_SOLVERS_PROSAC_SAMPLER_H_
#define THEIA_SOLVERS_PROSAC_SAMPLER_H_

#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

#include "sampler.h"

namespace theia
{
	namespace
	{
		std::default_random_engine util_generator;
	}  // namespace

	// Prosac sampler used for PROSAC implemented according to "cv::Matching with PROSAC
	// - Progressive Sampling Consensus" by Chum and cv::Matas.
	template <class Datum> class ProsacSampler : public Sampler < Datum >
	{
	public:
		// Get a random double between lower and upper (inclusive).
		double RandDouble(double lower, double upper)
		{
			std::uniform_real_distribution<double> distribution(lower, upper);
			return distribution(util_generator);
		}

		// Get a random int between lower and upper (inclusive).
		int RandInt(int lower, int upper)
		{
			std::uniform_int_distribution<int> distribution(lower, upper);
			return distribution(util_generator);
		}

		// Gaussian Distribution with the corresponding mean and std dev.
		double RandGaussian(double mean, double std_dev)
		{
			std::normal_distribution<double> distribution(mean, std_dev);
			return distribution(util_generator);
		}

		explicit ProsacSampler(const int min_num_samples)
			: Sampler<Datum>(min_num_samples) {}
		~ProsacSampler() {}

		bool initialize()
		{
			ransac_convergence_iterations_ = 100000;
			kth_sample_number_ = 1;
			
			unsigned seed = static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count());
			util_generator.seed(seed);
			return true;
		}

		// Set the sample such that you are sampling the kth prosac sample (Eq. 6).
		void setSampleNumber(int k)
		{
			kth_sample_number_ = k;
		}

		// Samples the input variable data and fills the std::vector subset with the prosac
		// samples.
		// NOTE: This assumes that data is in sorted order by quality where data[i] is
		// of higher quality than data[j] for all i < j.
		bool sample(const std::vector<Datum>& data, std::vector<Datum>* subset)
		{
			// Set t_n according to the PROSAC paper's recommendation.
			double t_n = ransac_convergence_iterations_;
			const int point_number = static_cast<int>(data.size());
			int n = this->min_num_samples_;
			// From Equations leading up to Eq 3 in Chum et al.
			for (auto i = 0; i < this->min_num_samples_; i++)
			{
				t_n *= static_cast<double>(n - i) / (point_number - i);
			}

			double t_n_prime = 1.0;
			// Choose min n such that T_n_prime >= t (Eq. 5).
			for (auto t = 1; t <= kth_sample_number_; t++)
			{
				if (t > t_n_prime && n < point_number)
				{
					double t_n_plus1 =
						(t_n * (n + 1.0)) / (n + 1.0 - this->min_num_samples_);
					t_n_prime += ceil(t_n_plus1 - t_n);
					t_n = t_n_plus1;
					n++;
				}
			}
			subset->reserve(this->min_num_samples_);
			if (t_n_prime < kth_sample_number_)
			{
				// Randomly sample m data points from the top n data points.
				std::vector<int> random_numbers;
				for (auto i = 0; i < this->min_num_samples_; i++)
				{
					// Generate a random number that has not already been used.
					int rand_number;
					while (std::find(random_numbers.begin(), random_numbers.end(),
						(rand_number = RandInt(0, n - 1))) !=
						random_numbers.end())
					{
					}

					random_numbers.emplace_back(rand_number);

					// Push the *unique* random index back.
					subset->emplace_back(data[rand_number]);
				}
			}
			else
			{
				std::vector<int> random_numbers;
				// Randomly sample m-1 data points from the top n-1 data points.
				for (auto i = 0; i < this->min_num_samples_ - 1; i++)
				{
					// Generate a random number that has not already been used.
					int rand_number;
					while (std::find(random_numbers.begin(), random_numbers.end(),
						(rand_number = RandInt(0, n - 2))) !=
						random_numbers.end())
					{
					}
					random_numbers.emplace_back(rand_number);

					// Push the *unique* random index back.
					subset->emplace_back(data[rand_number]);
				}
				// Make the last point from the nth position.
				subset->emplace_back(data[n]);
			}
			if (subset->size() != this->min_num_samples_)
				std::cout << "Prosac subset is of incorrect " << "size!" << " @" << __LINE__ << std::endl;
			kth_sample_number_++;
			return true;
		}

		bool sample(const cv::Mat& data, std::vector<int>& subset)
		{
			// Set t_n according to the PROSAC paper's recommendation.
			double t_n = ransac_convergence_iterations_;
			int n = this->min_num_samples_;
			// From Equations leading up to Eq 3 in Chum et al.
			for (auto i = 0; i < this->min_num_samples_; i++)
			{
				t_n *= static_cast<double>(n - i) / (data.rows - i);
			}

			double t_n_prime = 1.0;
			// Choose min n such that T_n_prime >= t (Eq. 5).
			for (auto t = 1; t <= kth_sample_number_; t++)
			{
				if (t > t_n_prime && n < data.rows)
				{
					double t_n_plus1 =
						(t_n * (n + 1.0)) / (n + 1.0 - this->min_num_samples_);
					t_n_prime += ceil(t_n_plus1 - t_n);
					t_n = t_n_plus1;
					n++;
				}
			}
			subset.reserve(this->min_num_samples_);
			if (t_n_prime < kth_sample_number_)
			{
				// Randomly sample m data points from the top n data points.
				std::vector<int> random_numbers;
				for (auto i = 0; i < this->min_num_samples_; i++)
				{
					// Generate a random number that has not already been used.
					int rand_number;
					while (std::find(random_numbers.begin(), random_numbers.end(),
						(rand_number = RandInt(0, n - 1))) !=
						random_numbers.end())
					{
					}

					random_numbers.emplace_back(rand_number);

					// Push the *unique* random index back.
					subset.emplace_back(rand_number);
				}
			}
			else
			{
				std::vector<int> random_numbers;
				// Randomly sample m-1 data points from the top n-1 data points.
				for (auto i = 0; i < this->min_num_samples_ - 1; i++)
				{
					// Generate a random number that has not already been used.
					int rand_number;
					while (std::find(random_numbers.begin(), random_numbers.end(),
						(rand_number = RandInt(0, n - 2))) !=
						random_numbers.end())
					{
					}
					random_numbers.emplace_back(rand_number);

					// Push the *unique* random index back.
					subset.emplace_back(rand_number);
				}
				// Make the last point from the nth position.
				subset.emplace_back(n);
			}
			if (subset.size() != this->min_num_samples_)
				std::cout << "Prosac subset is of incorrect " << "size!" << " @" << __LINE__ << std::endl;
			kth_sample_number_++;
			return true;
		}
		
		bool sample(const cv::Mat& data, int * const subset)
		{
			// Set t_n according to the PROSAC paper's recommendation.
			double t_n = ransac_convergence_iterations_;
			int n = this->min_num_samples_;
			// From Equations leading up to Eq 3 in Chum et al.
			for (auto i = 0; i < this->min_num_samples_; i++)
			{
				t_n *= static_cast<double>(n - i) / (data.rows - i);
			}

			double t_n_prime = 1.0;
			// Choose min n such that T_n_prime >= t (Eq. 5).
			for (auto t = 1; t <= kth_sample_number_; t++)
			{
				if (t > t_n_prime && n < data.rows)
				{
					double t_n_plus1 =
						(t_n * (n + 1.0)) / (n + 1.0 - this->min_num_samples_);
					t_n_prime += ceil(t_n_plus1 - t_n);
					t_n = t_n_plus1;
					n++;
				}
			}

			int current_sample_number = 0;
			if (t_n_prime < kth_sample_number_)
			{
				// Randomly sample m data points from the top n data points.
				std::vector<int> random_numbers;
				for (auto i = 0; i < this->min_num_samples_; i++)
				{
					// Generate a random number that has not already been used.
					int rand_number;
					while (std::find(random_numbers.begin(), random_numbers.end(),
						(rand_number = RandInt(0, n - 1))) !=
						random_numbers.end())
					{
					}

					random_numbers.emplace_back(rand_number);

					// Push the *unique* random index back.
					subset[current_sample_number++] = rand_number;
				}
			}
			else
			{
				std::vector<int> random_numbers;
				// Randomly sample m-1 data points from the top n-1 data points.
				for (auto i = 0; i < this->min_num_samples_ - 1; i++)
				{
					// Generate a random number that has not already been used.
					int rand_number;
					while (std::find(random_numbers.begin(), random_numbers.end(),
						(rand_number = RandInt(0, n - 2))) !=
						random_numbers.end())
					{
					}
					random_numbers.emplace_back(rand_number);

					// Push the *unique* random index back.
					subset[current_sample_number++] = rand_number;
				}
				// Make the last point from the nth position.
				subset[current_sample_number++] = n;
			}
			if (current_sample_number != this->min_num_samples_)
				std::cout << "Prosac subset is of incorrect " << "size!" << " @" << __LINE__ << std::endl;
			kth_sample_number_++;
			return true;
		}

	private:
		// Number of iterations of PROSAC before it just acts like ransac.
		int ransac_convergence_iterations_;

		// The kth sample of prosac sampling.
		int kth_sample_number_;
	};

}  // namespace theia

#endif  // THEIA_SOLVERS_PROSAC_SAMPLER_H_
