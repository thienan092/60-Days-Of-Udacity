#pragma once

#define LAYER_OUTPUT_TEST 1


#if !LAYER_OUTPUT_TEST
#define BOT_ID 1

#else

#define BOT_TEST_ID 0
#define BOT_ID BOT_TEST_ID
#define TESTED_LAYER_SPECIFYING_POINT bool specifying_flag = true; p_id = team_mate_id;

#endif // !LAYER_OUTPUT_TEST


#define STR_X(x) #x
#define STR(x) STR_X(x)

#define MATRICES_FOLDER "matrices" STR(BOT_ID) "/"
#define COACH_STR "coach_dense_"
#define COACH_BATCH_NORM_STR "coach_batch_normalization_"
#define TEAM_MATE_STR "team_member_move_shoot_"
#define TEAM_MATE_DENSE_STR "dense_"
#define TEAM_MATE_BATCH_NORM_STR "batch_normalization_"
#define WEIGHTS_STR "kernel.npy"
#define BIAS_STR "bias.npy"
#define BETA_STR "beta.npy"
#define GAMMA_STR "gamma.npy"
#define MOVING_MEAN_STR "moving_mean.npy"
#define MOVING_VARIANCE_STR "moving_variance.npy"
#define HIDDEN_LAYER1 512
#define HIDDEN_LAYER2 256

const int N_PLAYER = 5;
const int TEAM_ID_A = 0;
const int TEAM_ID_B = 1;
const int RAG_TO_SHOOT = 200;
const int GOALWIDTH = 3000;
//const int RAG_TO_GOAL = 6000;
const int HALF_1 = 1;
const int HALF_2 = 2;
const int MATCH_EXTRA = 3;

const int W_BINS = 64;
const int H_BINS = 48;
const int FORCE_MAX = 100;
const int FORCE_MIN = 20;
const int FORCE_FD = 4;
const int ROOT_STATE_SIZE = 1 - 1 + (2 + 2) + (2 * N_PLAYER) + (2 * N_PLAYER);
const int STATE_SIZE = ROOT_STATE_SIZE * (ROOT_STATE_SIZE + 1);
const int MOVE_ACTION_SIZE = W_BINS * H_BINS;
const int SHOOT_ACTION_SIZE = MOVE_ACTION_SIZE * FORCE_FD;

const int is_ready = BOT_ID;
const double noise_prob = 0.4;
const float ALPHA = 0.01;

extern int mapWidth;
extern int mapHeight;
// const int mapWidth = 14400;
// const int mapHeight = 9600;

// Strutures and Functions

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

void ChangePos(Object& A, int myTeamID, int match);
void ChangePos(int& x, int&y, int myTeamID, int match);
bool CanShoot(Position& mPos, Position& ballPos);
float DistancePos(Position& PosA, Position& PosB);
void to_pos_x_y(Object& player, int q_id, int& x, int& y);
void to_pos_x_y_f(Object& player, int q_id, int& x, int& y, int& f);