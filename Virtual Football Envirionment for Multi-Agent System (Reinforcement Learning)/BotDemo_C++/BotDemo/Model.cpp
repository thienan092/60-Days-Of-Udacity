#include "Model.h"
#include "MiniFlow.h"

Model::Model()
{
}

int Model::FindIndexMaxQs(int first, int last)
{
	int max = first;
	for (int i = first; i <= last; i++)
	{
		if (Qs[max] < Qs[i])
		{
			max = i;
		}
	}
	return max;
}

int Model::GetMoveAction()
{
	return FindIndexMaxQs(0, MOVE_ACTION_SIZE - 1);
}

int Model::GetShootAction()
{
	return FindIndexMaxQs(MOVE_ACTION_SIZE, MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE - 1) - MOVE_ACTION_SIZE;
}

void Model::LoadTeamMate(int team_mate_id)
{
	string team_mate_id_str = to_string(team_mate_id) + "_";
	string sub_name = string(TEAM_MATE_STR) + team_mate_id_str;
	// Load 1st layer
	{
		string file_name = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_DENSE_STR) + string(WEIGHTS_STR);
		MiniFlow::loadMat(((float*)team_mate_matrices0) + team_mate_id * STATE_SIZE * HIDDEN_LAYER1, STATE_SIZE, HIDDEN_LAYER1, file_name);
	}
	// Load 1st BN
	{
		string bn_prefix = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_BATCH_NORM_STR);
		MiniFlow::loadBN(((float*)team_mate_beta0) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_gamma0) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_moving_mean0) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_moving_variance0) + team_mate_id * HIDDEN_LAYER1, HIDDEN_LAYER1, bn_prefix);
	}

	// Load 2nd layer
	{
		string file_name = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_DENSE_STR) + string("1_") + string(WEIGHTS_STR);
		MiniFlow::loadMat(((float*)team_mate_matrices1) + team_mate_id * HIDDEN_LAYER1 * HIDDEN_LAYER1, HIDDEN_LAYER1, HIDDEN_LAYER1, file_name);
	}

	// Load 2nd BN
	{
		string bn_prefix = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_BATCH_NORM_STR) + string("1_");
		MiniFlow::loadBN(((float*)team_mate_beta1) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_gamma1) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_moving_mean1) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_moving_variance1) + team_mate_id * HIDDEN_LAYER1, HIDDEN_LAYER1, bn_prefix);
	}

	// Load 3rd layer
	{
		string file_name = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_DENSE_STR) + string("2_") + string(WEIGHTS_STR);
		MiniFlow::loadMat(((float*)team_mate_matrices2) + team_mate_id * HIDDEN_LAYER1 * HIDDEN_LAYER2, HIDDEN_LAYER1, HIDDEN_LAYER2, file_name);
	}

	// Load 3rd BN
	{
		string bn_prefix = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_BATCH_NORM_STR) + string("2_");
		MiniFlow::loadBN(((float*)team_mate_beta2) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_gamma2) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_moving_mean2) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_moving_variance2) + team_mate_id * HIDDEN_LAYER2, HIDDEN_LAYER2, bn_prefix);
	}

	// Load 4th layer
	{
		string file_name = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_DENSE_STR) + string("3_") + string(WEIGHTS_STR);
		MiniFlow::loadMat(((float*)team_mate_matrices3) + team_mate_id * HIDDEN_LAYER2 * HIDDEN_LAYER2, HIDDEN_LAYER2, HIDDEN_LAYER2, file_name);
	}

	// Load 4th BN
	{
		string bn_prefix = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_BATCH_NORM_STR) + string("3_");
		MiniFlow::loadBN(((float*)team_mate_beta3) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_gamma3) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_moving_mean3) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_moving_variance3) + team_mate_id * HIDDEN_LAYER2, HIDDEN_LAYER2, bn_prefix);
	}

	// Load Out layer
	{
		string file_name = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_DENSE_STR) + string("4_") + string(WEIGHTS_STR);
		MiniFlow::loadMat(((float*)team_mate_matrices_output) + team_mate_id * HIDDEN_LAYER2 * (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE), HIDDEN_LAYER2, (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE), file_name);
	}
	// Load bias
	{
		string file_name = string(MATRICES_FOLDER) + sub_name + string(TEAM_MATE_DENSE_STR) + string("4_") + string(BIAS_STR);
		MiniFlow::loadVec(((float*)team_mate_bias_output) + team_mate_id * (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE), (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE), file_name);
	}
}

void Model::LoadMatrices()
{
	for (int i = 0; i < N_PLAYER; i++)
	{
		LoadTeamMate(i);
	}
}

void Model::CalculateQs(float* state)
{
	for (int team_mate_id = 0; team_mate_id < N_PLAYER; team_mate_id++)
	{
#if LAYER_OUTPUT_TEST
		{
			ofstream out_file("unit_test_outputs/Cstates.txt");
			for (int i = 0; i < STATE_SIZE; i++)
			{
				out_file << state[i] << " ";
			}
		}

TESTED_LAYER_SPECIFYING_POINT
#endif

		MiniFlow::Mul(state, STATE_SIZE, ((float*)team_mate_matrices0) + team_mate_id * STATE_SIZE * HIDDEN_LAYER1, HIDDEN_LAYER1, result_HIDDEN_LAYER1);

		MiniFlow::BNorm(((float*)team_mate_beta0) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_gamma0) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_moving_mean0) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_moving_variance0) + team_mate_id * HIDDEN_LAYER1, HIDDEN_LAYER1, result_HIDDEN_LAYER1);
		MiniFlow::ReLu(result_HIDDEN_LAYER1, HIDDEN_LAYER1);



		MiniFlow::Mul(result_HIDDEN_LAYER1, HIDDEN_LAYER1, ((float*)team_mate_matrices1) + team_mate_id * HIDDEN_LAYER1 * HIDDEN_LAYER1, HIDDEN_LAYER1, result_HIDDEN_LAYER1_SWP);
		MiniFlow::BNorm(((float*)team_mate_beta1) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_gamma1) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_moving_mean1) + team_mate_id * HIDDEN_LAYER1, \
			((float*)team_mate_moving_variance1) + team_mate_id * HIDDEN_LAYER1, HIDDEN_LAYER1, result_HIDDEN_LAYER1_SWP);
		MiniFlow::ReLu(result_HIDDEN_LAYER1_SWP, HIDDEN_LAYER1);

		MiniFlow::Mul(result_HIDDEN_LAYER1_SWP, HIDDEN_LAYER1, ((float*)team_mate_matrices2) + team_mate_id * HIDDEN_LAYER1 * HIDDEN_LAYER2, HIDDEN_LAYER2, result_HIDDEN_LAYER2);
		MiniFlow::BNorm(((float*)team_mate_beta2) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_gamma2) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_moving_mean2) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_moving_variance2) + team_mate_id * HIDDEN_LAYER2, HIDDEN_LAYER2, result_HIDDEN_LAYER2);
		MiniFlow::ReLu(result_HIDDEN_LAYER2, HIDDEN_LAYER2);

		MiniFlow::Mul(result_HIDDEN_LAYER2, HIDDEN_LAYER2, ((float*)team_mate_matrices3) + team_mate_id * HIDDEN_LAYER2 * HIDDEN_LAYER2, HIDDEN_LAYER2, result_HIDDEN_LAYER2_SWP);
		MiniFlow::BNorm(((float*)team_mate_beta3) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_gamma3) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_moving_mean3) + team_mate_id * HIDDEN_LAYER2, \
			((float*)team_mate_moving_variance3) + team_mate_id * HIDDEN_LAYER2, HIDDEN_LAYER2, result_HIDDEN_LAYER2_SWP);
		MiniFlow::ReLu(result_HIDDEN_LAYER2_SWP, HIDDEN_LAYER2);

		MiniFlow::Mul(result_HIDDEN_LAYER2_SWP, HIDDEN_LAYER2, ((float*)team_mate_matrices_output) + team_mate_id * HIDDEN_LAYER2 * (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE), MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE, ((float*)Qs) + team_mate_id * (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE));

		MiniFlow::Bias(((float*)team_mate_bias_output) + team_mate_id * (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE), (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE), ((float*)Qs) + team_mate_id * (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE));
	}
}

Model::~Model()
{
}
