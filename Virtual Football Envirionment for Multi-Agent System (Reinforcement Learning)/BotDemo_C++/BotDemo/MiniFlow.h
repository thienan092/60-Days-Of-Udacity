#pragma once
#include <string>
#include "definitions.h"
#include "npy.hpp"
#include <fstream>

using namespace std;
#if LAYER_OUTPUT_TEST
extern int p_id;
#endif

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
#if LAYER_OUTPUT_TEST
		ofstream* out_file;
		if (p_id != -1)
			out_file = new ofstream(string("unit_test_outputs/test_layer_p") + to_string(p_id) + string(".txt"));
#endif

		for (int i = 0; i < len; i++)
		{
			result[i] = gamma[i] * (result[i] - mean[i]) / sqrt(var[i] + 0.001f) + beta[i];
#if LAYER_OUTPUT_TEST
			if (p_id != -1)
				*out_file << result[i] << " ";
#endif
		}

#if LAYER_OUTPUT_TEST
		if (p_id != -1)
			delete out_file; p_id = -1;
#endif
	}

	static void Bias(float* bias, int len, float* result)
	{
#if LAYER_OUTPUT_TEST
		ofstream* out_file;
		if (p_id != -1)
			out_file = new ofstream(string("unit_test_outputs/test_layer_p") + to_string(p_id) + string(".txt"));
#endif

		for (int i = 0; i < len; i++)
		{
			result[i] = result[i] + bias[i];
#if LAYER_OUTPUT_TEST
			if (p_id != -1)
				*out_file << result[i] << " ";
#endif
		}
#if LAYER_OUTPUT_TEST
		if (p_id != -1)
			delete out_file; p_id = -1;
#endif
	}

	static void ReLu(float* result, int len)
	{
#if LAYER_OUTPUT_TEST
		ofstream* out_file;
		if (p_id != -1)
			out_file = new ofstream(string("unit_test_outputs/test_layer_p") + to_string(p_id) + string(".txt"));
#endif
		for (int i = 0; i < len; i++)
		{
			result[i] = max(ALPHA * result[i], result[i]);
#if LAYER_OUTPUT_TEST
			if (p_id != -1)
				*out_file << result[i] << " ";
#endif
		}
#if LAYER_OUTPUT_TEST
		if (p_id != -1)
			delete out_file; p_id = -1;
#endif
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
#if LAYER_OUTPUT_TEST
		ofstream* out_file;
		if (p_id != -1)
			out_file = new ofstream(string("unit_test_outputs/test_layer_p") + to_string(p_id) + string(".txt"));
#endif

		for (int j = 0; j < A_width; j++)
		{
			result[j] = 0;
			for (int i = 0; i < V_len; i++)
			{
				result[j] += Vec[i] * (*(((float*)MatA) + i * A_width + j));
			}
#if LAYER_OUTPUT_TEST
			if (p_id != -1)
				*out_file << result[j] << " ";
#endif
		}
#if LAYER_OUTPUT_TEST
		if (p_id != -1)
			delete out_file; p_id = -1;
#endif
	}


	MiniFlow();
	~MiniFlow();
};

