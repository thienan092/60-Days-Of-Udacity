// BotDemo.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>
#include <time.h>
#include <math.h>

#include "definitions.h"
#include "MiniFlow.h"



using namespace std;


//
//float coach_matrix0[STATE_SIZE][HIDDEN_LAYER1];
//float coach_beta0[HIDDEN_LAYER1];
//float coach_gamma0[HIDDEN_LAYER1];
//float coach_moving_mean0[HIDDEN_LAYER1];
//float coach_moving_variance0[HIDDEN_LAYER1];
//
//float coach_matrix1[HIDDEN_LAYER1][HIDDEN_LAYER2];
//float coach_beta1[HIDDEN_LAYER2];
//float coach_gamma1[HIDDEN_LAYER2];
//float coach_moving_mean1[HIDDEN_LAYER2];
//float coach_moving_variance1[HIDDEN_LAYER2];

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

int mapWidth = 14400;
int mapHeight = 9600;

struct Model
{
	float Qs[N_PLAYER][MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE];
	float result_HIDDEN_LAYER1[HIDDEN_LAYER1];
	float result_HIDDEN_LAYER1_SWP[HIDDEN_LAYER1];
	float result_HIDDEN_LAYER2[HIDDEN_LAYER2];
	float result_HIDDEN_LAYER2_SWP[HIDDEN_LAYER2];

	void LoadTeamMate(int team_mate_id)
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

	void LoadMatrices()
	{
		for (int i = 0; i < N_PLAYER; i++)
		{
			LoadTeamMate(i);
		}
	}



	void CalculateQs(float* state)
	{
		//printf("%f %f %f %f", state[1], state[2], state[3], state[4]);
		//printf("%f %f %f %f", state[5], state[10], state[15], state[20]);
		//std::cin >> state[0];

		/*Mul(state, STATE_SIZE, (float*)coach_matrix0, HIDDEN_LAYER1, result_HIDDEN_LAYER1);
		BNorm(coach_beta0, coach_gamma0, coach_moving_mean0, coach_moving_variance0, HIDDEN_LAYER1, result_HIDDEN_LAYER1);
		ReLu(result_HIDDEN_LAYER1, HIDDEN_LAYER1);



		Mul(result_HIDDEN_LAYER1, HIDDEN_LAYER1, (float*)coach_matrix1, HIDDEN_LAYER2, Sts);
		BNorm(coach_beta1, coach_gamma1, coach_moving_mean1, coach_moving_variance1, HIDDEN_LAYER2, Sts);
		ReLu(Sts, HIDDEN_LAYER2);*/


		for (int team_mate_id = 0; team_mate_id < N_PLAYER; team_mate_id++)
		{
			MiniFlow::Mul(state, STATE_SIZE, ((float*)team_mate_matrices0) + team_mate_id * STATE_SIZE * HIDDEN_LAYER1, HIDDEN_LAYER1, result_HIDDEN_LAYER1);

			if (true) // testing purpose
			{
				int test = 1;
				printf("result_HIDDEN_LAYER\n");
				cout << result_HIDDEN_LAYER1[0] << endl << result_HIDDEN_LAYER1[1] << endl;
				cout << result_HIDDEN_LAYER1[512 - 1] << endl << result_HIDDEN_LAYER1[512 - 2] << endl;
				printf("something...");
				cin >> test;
			}

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
};

struct Position
{
	int x;
	int y;
	Position() : x(0), y(0) {}
	Position(int _x, int _y) : x(_x), y(_y) {}
	Position operator =(Position pos) { x = pos.x; y = pos.y; return *this; }
};

enum Action
{
	WAIT = 0,
	RUN,
	SHOOT
};

struct Object
{
	int ID;
	Position m_pos;
	Position m_targetPos; //direction of moving/target pos
	Position m_moveSpeed;

	int m_action;
	int m_force;
};


bool CanShoot(Position& mPos, Position& ballPos);
float DistancePos(Position& PosA, Position& PosB);
int FindMax(float* a, int first, int last);
void input_to_norm_states(int gameTurn, Object& oBall, Object** Player_A, Object** Player_B, float* state, int* Bmap);
void to_pos_x_y(Object& player, int q_id, int& x, int& y);
void to_pos_x_y_f(Object& player, int q_id, int& x, int& y, int& f);

struct Features
{
	int AtoB[N_PLAYER][N_PLAYER];
	int Bmap[N_PLAYER];
	Features(Object** Player_A, Object** Player_B, Object& oBall)
	{
		calc_AtoB(Player_A, Player_B);
		calc_Bmap();
	}

	void calc_AtoB(Object** Player_A, Object** Player_B)
	{
		for (int i = 0; i < N_PLAYER; i++)
		{
			for (int j = 0; j < N_PLAYER; j++)
			{
				AtoB[i][j] = DistancePos(Player_A[i]->m_pos, Player_B[j]->m_pos);
			}
		}
	}

	void calc_Bmap()
	{
		bool mark[N_PLAYER];
		for (int i = 0; i < N_PLAYER; i++)
		{
			mark[i] = false;
		}
		int old_id1 = -1;
		int old_id2 = -1;
		int old_id3 = -1;
		int old_id4 = -1;
		int old_id5 = -1;
		float min = -1;
		for (int id1 = 0; id1 < N_PLAYER; id1++)
		{

			// ID1 check validate and then, swap marks
			if (mark[id1] == false)
			{
				if (old_id1 != -1)
				{
					mark[old_id1] = false;
				}
				mark[id1] = true;
				old_id1 = id1;
			}
			else
			{
				continue;
			}

			for (int id2 = 0; id2 < N_PLAYER; id2++)
			{

				// ID2 check validate and then, swap marks
				if (mark[id2] == false)
				{
					if (old_id2 != -1)
					{
						mark[old_id2] = false;
					}
					mark[id2] = true;
					old_id2 = id2;
				}
				else
				{
					continue;
				}
				
				for (int id3 = 0; id3 < N_PLAYER; id3++)
				{
					// ID3 check validate and then, swap marks
					if (mark[id3] == false)
					{
						if (old_id3 != -1)
						{
							mark[old_id3] = false;
						}
						mark[id3] = true;
						old_id3 = id3;
					}
					else
					{
						continue;
					}
					
					for (int id4 = 0; id4 < N_PLAYER; id4++)
					{
						// ID4 check validate and then, swap marks
						if (mark[id4] == false)
						{
							if (old_id4 != -1)
							{
								mark[old_id4] = false;
							}
							mark[id4] = true;
							old_id4 = id4;
						}
						else
						{
							continue;
						}
						int id5 = (0 + 1 + 2 + 3 + 4) - (id1 + id2 + id3 + id4);
						mark[id5] = false;

						// Update min, map #
						float dist_map_value = 0.0;
						int tempt_map[N_PLAYER] = { id1, id2, id3, id4, id5 };
						for (int id_map = 0; id_map < N_PLAYER; id_map++)
						{
							dist_map_value = dist_map_value + this->AtoB[id_map][tempt_map[id_map]];
						}
								
						if ((min == -1) || (min > dist_map_value))
						{
							min = dist_map_value;
							for (int id_map = 0; id_map < N_PLAYER; id_map++)
							{
								this->Bmap[id_map] = tempt_map[id_map];
							}
						}
									
						// Update min, map #
					}

					if (old_id4 != -1)
					{
						mark[old_id4] = false;
					}
					old_id4 = -1;
				}

				if (old_id3 != -1)
				{
					mark[old_id3] = false;
				}
				old_id3 = -1;
			}

			if (old_id2 != -1)
			{
				mark[old_id2] = false;
			}
			old_id2 = -1;
		}
	}

};

void ChangePos(Object& A, int myTeamID, int match)
{
	if (((myTeamID == 0) && (match == HALF_1)) || ((myTeamID == 1) && (match != HALF_1)))
		return;
	A.m_pos.x = mapWidth - A.m_pos.x;
	A.m_pos.y = mapHeight - A.m_pos.y;
	A.m_moveSpeed.x = -A.m_moveSpeed.x;
	A.m_moveSpeed.y = -A.m_moveSpeed.y;
}

void ChangePos(int& x, int&y, int myTeamID, int match)
{
	if (((myTeamID == 0) && (match == HALF_1)) || ((myTeamID == 1) && (match != HALF_1)))
		return;
	x = mapWidth - x;
	y = mapHeight - y;
}

int main()
{
	float state[STATE_SIZE + 1];
	Model mn_flow;
	mn_flow.LoadMatrices();

	{
		printf("2400 2400 4800 2400 2400 4800 2400 7200 4800 7200");
	}

	int gameTurn = 0;
	int scoreTeamA = 0;
	int scoreTeamB = 0;
	int stateMath = 0;
	int myTeamID = 1;
	int maxTurn = 0;

	Object** Player_A = new Object* [5];
	for (int i = 0; i < N_PLAYER; i++)
	{
		Player_A[i] = new Object;
	}
	Object** Player_B = new Object* [5];
	for (int i = 0; i < N_PLAYER; i++)
	{
		Player_B[i] = new Object;
	}
	Object oBall;

	//init timer rand
	srand(time(NULL));

	Position oppGoal;

	//{
	//	float vec[2] = { 2.0, 1.0 };
	//	float mat[2][3] = { { 1.0, 3.0, 2.0 } ,{ 4.0, 5.0, 6.0 } };
	//	float bias[3] = { 0.01, 0.02, -4.03 };
	//	float res[3];
	//	mn_flow.Mul(vec, 2, (float*)mat, 3, bias, res);
	//	printf("%f %f %f %d", res[0], res[1], res[2], FindMax(res, 1, 0));
	//}

	//Formation your team, set your players position on field
	//If you don't send this, server will auto set postion for all your players.
	//format: [Player1_posX][Player1_posY][Player2_posX][Player2_posY]...[Player5_posX][Player5_posY]
	//cout <<2660 2400 5320 2400 2660 240 2660 7200 5320 7200;
	//cout << "133 120 133 125 133 240 133 360 133 365";

	//Init game: Server send team ID, Mapsize WH and maxTurn
	std::cin >> myTeamID >> mapWidth >> mapHeight >> maxTurn;
	myTeamID = is_ready;
	//{
	//	ofstream config_f("config_f");
	//	config_f << mapWidth << endl;
	//	config_f << mapHeight << endl;
	//}



	while (gameTurn++ < maxTurn)
	{

		//Server send to Bot each turn :
		//[Turn] [m_scoreTeamA] [m_scoreTeamB] [stateMath] [ballPosX] [ballPosY] [ballSpeedX] [ballSpeedY] [Player1_Team1] [Player1_Team1_posX] [Player1_Team1_posY] ...[Player1_Team2][Player1_Team2_posX][Player1_Team2_posY]...[Player5_Team2][Player5_Team2_posX][Player5_Team2_posY]

		//input common infos
		std::cin >> gameTurn >> scoreTeamA >> scoreTeamB >> stateMath;
		//gameTurn2 = 1; scoreTeamA = 1; scoreTeamB = 1; stateMath = 1;
		std::cin >> oBall.m_pos.x >> oBall.m_pos.y >> oBall.m_moveSpeed.x >> oBall.m_moveSpeed.y;
		ChangePos(oBall ,myTeamID ,stateMath);
		//oBall.m_pos.x = 1; oBall.m_pos.y = 1; oBall.m_moveSpeed.x = 1; oBall.m_moveSpeed.y = 1;
		//input team players A
		if (myTeamID == 0)
		{
			for (int i = 0; i < N_PLAYER; i++) {
				std::cin >> Player_A[i]->ID >> Player_A[i]->m_pos.x >> Player_A[i]->m_pos.y;
				ChangePos(*Player_A[i], myTeamID, stateMath);
				//Player_A[i]->ID = 1; Player_A[i]->m_pos.x = i; Player_A[i]->m_pos.y = i;
			}

			//Input team players B
			for (int i = 0; i < N_PLAYER; i++) {
				std::cin >> Player_B[i]->ID >> Player_B[i]->m_pos.x >> Player_B[i]->m_pos.y;
				ChangePos(*Player_B[i], myTeamID, stateMath);
				//Player_B[i]->ID = 1; Player_B[i]->m_pos.x = i-1; Player_B[i]->m_pos.y = i-1;
			}
		}
		else
		{
			//Input team players B
			for (int i = 0; i < N_PLAYER; i++) {
				std::cin >> Player_B[i]->ID >> Player_B[i]->m_pos.x >> Player_B[i]->m_pos.y;
				ChangePos(*Player_B[i], myTeamID, stateMath);
				//Player_B[i]->ID = 1; Player_B[i]->m_pos.x = i-1; Player_B[i]->m_pos.y = i-1;
			}

			for (int i = 0; i < N_PLAYER; i++) {
				std::cin >> Player_A[i]->ID >> Player_A[i]->m_pos.x >> Player_A[i]->m_pos.y;
				ChangePos(*Player_A[i], myTeamID, stateMath);
				//Player_A[i]->ID = 1; Player_A[i]->m_pos.x = i; Player_A[i]->m_pos.y = i;
			}
		}


		//====================	
		//Update players	  |
		//====================
		//Send action to server each turn with format:
		//[Player1_action] [X1] [Y1] [F1] [Player_2_action] [X2] [Y2] [F2]... [Player_5_action] [X5] [Y5] [F5]
		//Action: WAIT - SHOOT - RUN ;
		//X,Y: Target of the action;
		//F: force of shooting (If action == SHOOT)
		//Example output format: std::cout << "1 2 4 5 1 58 38 10 2 11 5 20 1 229 188 0 2 259 395 99";

		//Demo send players' action to server		
		Features features(Player_A, Player_B, oBall);
		input_to_norm_states(gameTurn, oBall, Player_A, Player_B, (float*)state, (int*)features.Bmap);
		
		mn_flow.CalculateQs(((float*)state) + 1);

		for (int i = 0; i < N_PLAYER; i++)
		{
			int Q_id = 0;
			int action = 2, x = 0, y = 0, f = 0;
			if (CanShoot(Player_A[i]->m_pos, oBall.m_pos))
			{
				action = 2;
				Q_id = FindMax(mn_flow.Qs[i], MOVE_ACTION_SIZE, MOVE_ACTION_SIZE + SHOOT_ACTION_SIZE - 1) - MOVE_ACTION_SIZE;
				to_pos_x_y_f(*(Player_A[i]), Q_id, x, y, f);
			}
			else
			{
				action = 1;
				Q_id = FindMax(mn_flow.Qs[i], 0, MOVE_ACTION_SIZE - 1);
				to_pos_x_y(*(Player_A[i]), Q_id, x, y);
			}

			if ((is_ready + stateMath) == HALF_2)
			{
				double r = ((double)rand() / (RAND_MAX));
				if (r < (noise_prob + 0.2f))
				{
					if (CanShoot(Player_A[i]->m_pos, oBall.m_pos))
					{
						action = 2;
						Q_id = (rand() % SHOOT_ACTION_SIZE);
						to_pos_x_y_f(*(Player_A[i]), Q_id, x, y, f);
					}
					else
					{
						action = 1;

						Q_id = rand() % MOVE_ACTION_SIZE;
						to_pos_x_y(*(Player_A[i]), Q_id, x, y);
					}
				}

			}
            else
            {
                double r = ((double)rand() / (RAND_MAX));
				if (r < (noise_prob))
				{
					if (CanShoot(Player_A[i]->m_pos, oBall.m_pos))
					{
						action = 2;
						Q_id = (rand() % SHOOT_ACTION_SIZE);
						to_pos_x_y_f(*(Player_A[i]), Q_id, x, y, f);
					}
					else
					{
						action = 1;

						Q_id = rand() % MOVE_ACTION_SIZE;
						to_pos_x_y(*(Player_A[i]), Q_id, x, y);
					}
				}
            }
			

			ChangePos(x, y, myTeamID, stateMath);
			printf("%d %d %d %d ", action, x, y, f);
		}

	}
	return 0;
}



bool CanShoot(Position& mPos, Position& ballPos)
{
	if (RAG_TO_SHOOT >= DistancePos(mPos, ballPos))
		return true;
	return false;
}

float DistancePos(Position& PosA, Position& PosB){
	float dis = sqrt((PosA.x - PosB.x)*(PosA.x - PosB.x) + (PosA.y - PosB.y)*(PosA.y - PosB.y));
	return dis;
}

int FindMax(float* a, int first, int last)
{
	int max = first;
	for (int i = first; i <= last; i++)
	{
		if (a[max] < a[i])
		{
			max = i;
		}
	}
	return max;
}

void input_to_norm_states(int gameTurn, Object& oBall, Object** Player_A, Object** Player_B, float* state, int* Bmap)
{
	state[0] = (float) gameTurn;
	state[1] = (((float) oBall.m_pos.x) - (((float) mapWidth) / 2)) / mapWidth;
	state[2] = (((float) oBall.m_pos.y) - (((float) mapHeight) / 2)) / mapHeight;
	state[3] = ((float) oBall.m_moveSpeed.x) / 1000;
	state[4] = ((float) oBall.m_moveSpeed.y) / 1000;

	for (int i = 0; i < N_PLAYER; i++)
	{
		state[5 + i] = (((float)Player_A[i]->m_pos.x) - (((float)mapWidth) / 2)) / mapWidth;
		state[10 + i] = (((float)Player_A[i]->m_pos.y) - (((float)mapHeight) / 2)) / mapHeight;
		state[15 + i] = (((float)Player_B[Bmap[i]]->m_pos.x) - (((float)mapWidth) / 2)) / mapWidth;;
		state[20 + i] = (((float)Player_B[Bmap[i]]->m_pos.y) - (((float)mapHeight) / 2)) / mapHeight;
	}

	for (int i = 0; i < 24; i++)
	{
		for (int j = 0; j < 24; j++)
		{
			state[i * 24 + j + 25] = state[i + 1] * state[j + 1];
		}
			
	}
}

void to_pos_x_y(Object& player, int q_id, int& x, int& y)
{
	x = q_id / H_BINS;
	y = q_id % H_BINS;
	x = (int)(x * (((float)mapWidth) / (W_BINS - 1)));
	y = (int)(y * (((float)mapHeight) / (H_BINS - 1)));

	x += player.m_pos.x;
	y += player.m_pos.y;
}

void to_pos_x_y_f(Object& player, int q_id, int& x, int& y, int& f)
{
	x = q_id / (H_BINS * FORCE_FD);
	int r = q_id % (H_BINS * FORCE_FD);
	y = r / FORCE_FD;
	f = r % FORCE_FD;
	x = (int)(x * (((float)mapWidth) / (W_BINS - 1)));
	y = (int)(y * (((float)mapHeight) / (H_BINS - 1)));
	f = (int)(f * (((float)(FORCE_MAX - FORCE_MIN)) / (FORCE_FD - 1))) + FORCE_MIN;

	x += player.m_pos.x;
	y += player.m_pos.y;
}

