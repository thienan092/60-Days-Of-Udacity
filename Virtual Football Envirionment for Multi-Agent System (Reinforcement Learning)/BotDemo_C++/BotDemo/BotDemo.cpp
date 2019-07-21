// BotDemo.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>
#include <time.h>

#include "definitions.h"
#include "Model.h"
#include "Features.h"

int mapWidth = 14400;
int mapHeight = 9600;

using namespace std;

#if !LAYER_OUTPUT_TEST
int main()
{
	float state[STATE_SIZE + 1];
	Model model;
	model.LoadMatrices();

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

	std::cin >> myTeamID >> mapWidth >> mapHeight >> maxTurn;
	myTeamID = is_ready;



	while (gameTurn++ < maxTurn)
	{

		//Server send to Bot each turn :
		//[Turn] [m_scoreTeamA] [m_scoreTeamB] [stateMath] [ballPosX] [ballPosY] [ballSpeedX] [ballSpeedY] [Player1_Team1] [Player1_Team1_posX] [Player1_Team1_posY] ...[Player1_Team2][Player1_Team2_posX][Player1_Team2_posY]...[Player5_Team2][Player5_Team2_posX][Player5_Team2_posY]

		//input common infos
		std::cin >> gameTurn >> scoreTeamA >> scoreTeamB >> stateMath;
		std::cin >> oBall.m_pos.x >> oBall.m_pos.y >> oBall.m_moveSpeed.x >> oBall.m_moveSpeed.y;
		ChangePos(oBall ,myTeamID ,stateMath);

		if (myTeamID == 0)
		{
			//Input team players A
			for (int i = 0; i < N_PLAYER; i++) {
				std::cin >> Player_A[i]->ID >> Player_A[i]->m_pos.x >> Player_A[i]->m_pos.y;
				ChangePos(*Player_A[i], myTeamID, stateMath);
			}

			//Input team players B
			for (int i = 0; i < N_PLAYER; i++) {
				std::cin >> Player_B[i]->ID >> Player_B[i]->m_pos.x >> Player_B[i]->m_pos.y;
				ChangePos(*Player_B[i], myTeamID, stateMath);
			}
		}
		else
		{
			//Input team players B
			for (int i = 0; i < N_PLAYER; i++) {
				std::cin >> Player_B[i]->ID >> Player_B[i]->m_pos.x >> Player_B[i]->m_pos.y;
				ChangePos(*Player_B[i], myTeamID, stateMath);
			}

			//Input team players A
			for (int i = 0; i < N_PLAYER; i++) {
				std::cin >> Player_A[i]->ID >> Player_A[i]->m_pos.x >> Player_A[i]->m_pos.y;
				ChangePos(*Player_A[i], myTeamID, stateMath);
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
		Features::states_to_complex_states(oBall, Player_A, Player_B, (float*)state, (int*)features.Bmap);
		
		model.CalculateQs(((float*)state));

		for (int i = 0; i < N_PLAYER; i++)
		{
			int Q_id = 0;
			int action = 2, x = 0, y = 0, f = 0;
			if (CanShoot(Player_A[i]->m_pos, oBall.m_pos))
			{
				action = 2;
				Q_id = model.GetShootAction();
				to_pos_x_y_f(*(Player_A[i]), Q_id, x, y, f);
			}
			else
			{
				action = 1;
				Q_id = model.GetMoveAction();
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
			x = (x < 0)? (-x) : ((x > mapWidth) ? (2 * mapWidth - x) : x);
			y = (y < 0) ? (-y) : ((y > mapHeight) ? (2 * mapHeight - y) : y);
			printf("%d %d %d %d ", action, x, y, f);
		}

	}
	return 0;
}
#else
int p_id = -1;

int main(int argc, char *argv[])
{
	if (argc > 1) {
		float state[STATE_SIZE];
		Model model;
		model.LoadMatrices();

		int gameTurn = 0;
		int scoreTeamA = 0;
		int scoreTeamB = 0;
		int stateMath = 0;
		int myTeamID = 1;
		int maxTurn = 0;

		Object** Player_A = new Object*[5];
		for (int i = 0; i < N_PLAYER; i++)
		{
			Player_A[i] = new Object;
		}
		Object** Player_B = new Object*[5];
		for (int i = 0; i < N_PLAYER; i++)
		{
			Player_B[i] = new Object;
		}
		Object oBall;

		istringstream ss(argv[1]);



		ss >> gameTurn >> scoreTeamA >> scoreTeamB >> stateMath;
		ss >> oBall.m_pos.x >> oBall.m_pos.y >> oBall.m_moveSpeed.x >> oBall.m_moveSpeed.y;

		//Input team players A
		for (int i = 0; i < N_PLAYER; i++) {
			ss >> Player_A[i]->ID >> Player_A[i]->m_pos.x >> Player_A[i]->m_pos.y;
		}

		//Input team players B
		for (int i = 0; i < N_PLAYER; i++) {
			ss >> Player_B[i]->ID >> Player_B[i]->m_pos.x >> Player_B[i]->m_pos.y;
		}

		Features features(Player_A, Player_B, oBall);
		Features::states_to_complex_states(oBall, Player_A, Player_B, (float*)state, (int*)features.Bmap);

		model.CalculateQs(((float*)state));

		return 0;
	}

	return -1;
}
#endif