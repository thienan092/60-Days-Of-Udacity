#pragma once
#include <string>
#include "definitions.h"
#include "npy.hpp"

using namespace std;

class MiniFlow
{
public:
	static void loadMat(float* dest, int height, int width, string& file_name)
	{
		npy::LoadArrayFromNumpy(file_name.c_str(), height * width, dest);
	}

	static void loadVec(float* dest, int d_len, string& file_name)
	{
		npy::LoadArrayFromNumpy(file_name.c_str(), d_len, dest);
	}

	static void loadBN(float* beta_dest, float*gamma_dest, float* mean_dest, float* var_dest, int d_len, string& prefix)
	{
		{
			string file_name = prefix + string(BETA_STR);
			npy::LoadArrayFromNumpy(file_name.c_str(), d_len, beta_dest);
		}
		{
			string file_name = prefix + string(GAMMA_STR);
			npy::LoadArrayFromNumpy(file_name.c_str(), d_len, gamma_dest);
		}
		{
			string file_name = prefix + string(MOVING_MEAN_STR);
			npy::LoadArrayFromNumpy(file_name.c_str(), d_len, mean_dest);
		}
		{
			string file_name = prefix + string(MOVING_VARIANCE_STR);
			npy::LoadArrayFromNumpy(file_name.c_str(), d_len, var_dest);
		}
	}

	static void BNorm(float* beta, float* gamma, float* mean, float* var, int len, float* result)
	{
		for (int i = 0; i < len; i++)
		{
			result[i] = gamma[i] * (result[i] - mean[i]) / sqrt(var[i] + 0.001f) + beta[i];
		}
	}

	static void Bias(float* bias, int len, float* result)
	{
		for (int i = 0; i < len; i++)
		{
			result[i] = result[i] + bias[i];
		}
	}

	static void ReLu(float* result, int len)
	{
		for (int i = 0; i < len; i++)
		{
			result[i] = max(ALPHA * result[i], result[i]);
		}
	}

	//void Sigmoid(float* result, int len)
	//{
	//	for (int i = 0; i < len; i++)
	//	{
	//		result[i] = 1.0f / (1.0f + exp(-result[i]));
	//	}
	//}

	static void Mul(float* Vec, int V_len, float* MatA, int A_width, float* result)
	{
		for (int j = 0; j < A_width; j++)
		{
			result[j] = 0;
			for (int i = 0; i < V_len; i++)
			{
				result[j] += Vec[i] * (*(((float*)MatA) + i * A_width + j));
			}
		}
	}


	MiniFlow();
	~MiniFlow();
};

