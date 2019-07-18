#include "Features.h"
#include "definitions.h"

Features::Features(Object** Player_A, Object** Player_B, Object& oBall)
{
	calc_AtoB(Player_A, Player_B);
	calc_Bmap();
}

void Features::calc_AtoB(Object** Player_A, Object** Player_B)
{
	for (int i = 0; i < N_PLAYER; i++)
	{
		for (int j = 0; j < N_PLAYER; j++)
		{
			AtoB[i][j] = DistancePos(Player_A[i]->m_pos, Player_B[j]->m_pos);
		}
	}
}

void Features::calc_Bmap()
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


Features::~Features()
{

}
