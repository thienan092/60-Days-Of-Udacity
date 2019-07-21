import subprocess
import tensorflow as tf
import numpy as np
import globals, model, trainBot
from features import states_to_complex_states

if __name__ == '__main__':
    if not os.path.exists("unit_test_outputs/"):
        os.makedirs("unit_test_outputs/")

    if 'session' in locals() and session is not None:
        session.close()
    with tf.Session() as sess:
        team_group = model.Group()
        sess.run(tf.global_variables_initializer())
        print("Loading Model...")
        t_vars = tf.trainable_variables()
        vars_list = [var for var in t_vars]
        saver = tf.train.Saver(var_list=vars_list)
        saver.restore(sess, globals.MODEL_SAVE_PATH)
        model.save_running_model_matrices(sess)
        print("Done!")
    
        trainer = trainBot.Trainer(team_group, 0, 0, sess)
        
        with open("layer_output_test_input.txt", "r") as in_file:
            server_content = in_file.read()
        
        print(server_content)
        
        si = trainBot.ServerInfo()
        si.parse_server_content(server_content)
        complex_state = states_to_complex_states(globals.input_to_states(si.gameTurn, si.oBall, si.Player_A, si.Player_B, si.features.Bmap))
        # print(complex_state)

        
        outputs = trainer.perform_layer_output_test(np.expand_dims(complex_state, axis=0), globals.N_PLAYER)
        
        thd = subprocess.Popen(['cmd', '/c', r'BotDemo.exe', server_content])
        thd.wait()
        
        C_outputs = [None for _ in range(globals.N_PLAYER)]
        
        
        for i in range(globals.N_PLAYER):
            print(f"Player {i}-th")
            file_name = f"unit_test_outputs/test_layer_p{i}.txt"
            C_outputs[i] = np.loadtxt(file_name)
            C_matrix = np.array(C_outputs[i]).flatten()
            P_matrix = np.array(outputs[i]).flatten()
            print(globals.bcolors.OKBLUE+"Python:"+globals.bcolors.ENDC)
            print(outputs[i][:10])
            print(globals.bcolors.OKGREEN+"C++:"+globals.bcolors.ENDC)
            print(C_outputs[i][:10])
            print(globals.bcolors.WARNING+"Unmatched entries: "+globals.bcolors.ENDC)
            print(globals.bcolors.FAIL+"Python:"+globals.bcolors.ENDC)
            print((P_matrix[:10])[np.abs(C_matrix-P_matrix)[:10] > 1e-7])
            print(globals.bcolors.FAIL+"C++:"+globals.bcolors.ENDC)
            print((C_matrix[:10])[np.abs(C_matrix-P_matrix)[:10] > 1e-7])
            print("------------------------")
        