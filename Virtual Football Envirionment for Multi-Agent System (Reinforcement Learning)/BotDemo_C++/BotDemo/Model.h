#pragma once
#include "MiniFlow.h"
#include <fstream>

class Model
{
	float team_mate_matrices0[N_PLAYER][STATE_SIZE][HIDDEN_LAYER1];
	float team_mate_beta0[N_PLAYER][HIDDEN_LAYER1];
	float team_mate_gamma0[N_PLAYER][HIDDEN_LAYER1];
	float team_mate_moving_mean0[N_PLAYER][HIDDEN_LAYER1];
	float team_mate_moving_variance0[N_PLAYER][HIDDEN_LAYER1];

	float team_mate_matrices1[N_PLAYER][HIDDEN_LAYER1][HIDDEN_LAYER1];
	float team_mate_beta1[N_PLAYER][HIDDEN_LAYER1];
	float team_mate_gamma1[N_PLAYER][HIDDEN_LAYER1];
	float team_mate_moving_mean1[N_PLAYER][HIDDEN_LAYER1];
	float team_mate_moving_variance1[N_PLAYER][HIDDEN_LAYER1];

	float team_mate_matrices2[N_PLAYER][HIDDEN_LAYER1][HIDDEN_LAYER2];
	float team_mate_beta2[N_PLAYER][HIDDEN_LAYER2];
	float team_mate_gamma2[N_PLAYER][HIDDEN_LAYER2];
	float team_mate_moving_mean2[N_PLAYER][HIDDEN_LAYER2];
	float team_mate_moving_variance2[N_PLAYER][HIDDEN_LAYER2];

	float team_mate_matrices3[N_PLAYER][HIDDEN_LAYER2][HIDDEN_LAYER2];
	float team_mate_beta3[N_PLAYER][HIDDEN_LAYER2];
	float team_mate_gamma3[N_PLAYER][HIDDEN_LAYER2];
	float team_mate_moving_mean3[N_PLAYER][HIDDEN_LAYER2];
	float team_mate_moving_variance3[N_PLAYER][HIDDEN_LAYER2];

	float team_mate_matrices_output[N_PLAYER][HIDDEN_LAYER2][MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE];
	float team_mate_bias_output[N_PLAYER][MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE];

	float Qs[N_PLAYER][MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE];
	float result_HIDDEN_LAYER1[HIDDEN_LAYER1];
	float result_HIDDEN_LAYER1_SWP[HIDDEN_LAYER1];
	float result_HIDDEN_LAYER2[HIDDEN_LAYER2];
	float result_HIDDEN_LAYER2_SWP[HIDDEN_LAYER2];

	void LoadTeamMate(int team_mate_id);
	int FindIndexMaxQs(int first, int last);

public:
#if LAYER_OUTPUT_TEST
	float test_layer_results[N_PLAYER][MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE];

	void write_test_layer()
	{
		for (int p_id = 0; p_id < N_PLAYER; p_id++)
		{
			ofstream out_file(string("unit_test_outputs/test_layer_p")+ to_string(p_id)+string(".txt"));
			for (int i = 0; i < (MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE); i++)
			{
				out_file << Qs[p_id][i] <<" ";
			}
		}
	}
#endif // LAYER_OUTPUT_TEST


	Model();
	void LoadMatrices();
	void CalculateQs(float* state);
	int GetMoveAction();
	int GetShootAction();

	~Model();
};

