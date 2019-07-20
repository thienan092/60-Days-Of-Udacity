import os, subprocess, shutil, multiprocessing, time
import tensorflow as tf
import numpy as np
import globals, model, trainBot

BATCH_SIZE = 32
EPOCHS = 1000

if __name__ == '__main__':
    if not os.path.exists("matches_data/"):
        os.makedirs("matches_data/")
    if not os.path.exists(globals.REPLAY_SAVE_PATH):
        os.makedirs(globals.REPLAY_SAVE_PATH)
        
    if 'session' in locals() and session is not None:
        session.close()
    with tf.Session() as sess:
        team_group = model.Group()
        sess.run(tf.global_variables_initializer())
        if os.path.exists(globals.MODEL_SAVE_PATH.split('/')[0]):
            print("Loading Model...")
            t_vars = tf.trainable_variables()
            vars_list = [var for var in t_vars]
            saver = tf.train.Saver(var_list=vars_list)
            saver.restore(sess, globals.MODEL_SAVE_PATH)
            print("Done!")
        else:
            print("No checkpoint was found!!")
            print("Saving The First Model...")
            t_vars = tf.trainable_variables()
            vars_list = [var for var in t_vars]
            saver = tf.train.Saver(var_list=vars_list)
            saver.save(sess, globals.MODEL_SAVE_PATH)
            print("Done!")
            
        model.save_running_model_matrices(sess)
            
        trainer = trainBot.Trainer(team_group, BATCH_SIZE, EPOCHS, sess)
        
        epochs_count = 4
        thead_num = 2
        
        print("Training...")
        print(f"No. Epoch: {EPOCHS}. ")
        print(f"BATCH_SIZE: {BATCH_SIZE}. ")
        print(f"Loading ~{BATCH_SIZE*1000} samples into Replay buffer")
        for epoch_id in range(epochs_count): 
            print("Epoch {}: ".format(epoch_id))
            
            
            ths = [None for _ in range(thead_num)]
            ids = [None for _ in range(thead_num)]
            thd_id = 0
            for i in range(BATCH_SIZE): 
                time.sleep(10)
                ths[thd_id] = subprocess.Popen(['cmd', '/c', r'StartMatch.bat > log{}.txt'.format(i)])
                ids[thd_id] = i
                if ((thd_id == (thead_num - 1)) or (i == (BATCH_SIZE - 1))):
                    for j in range(thead_num):
                        if (ths[j] is not None):
                            ths[j].wait()
                            shutil.move("log{}.txt".format(ids[j]), globals.REPLAY_SAVE_PATH + str(ids[j]))
                            trainer.update_buffer([globals.REPLAY_SAVE_PATH + str(ids[j])])
                        else:
                            break
                        
                    ths = [None for _ in range(thead_num)]
                    ids = [None for _ in range(thead_num)]
                    
                thd_id = (thd_id + 1) % thead_num
                
            trainer.train(globals.N_PLAYER)
            model.save_running_model_matrices(sess)
            
            
            ths = [None for _ in range(thead_num)]
            ids = [None for _ in range(thead_num)]
            thd_id = 0
            
            TEST_SIZE = int(0.2 * BATCH_SIZE)
            for i in range(TEST_SIZE): 
                time.sleep(10)
                ths[thd_id] = subprocess.Popen(['cmd', '/c', r'StartMatch.bat > log{}.txt'.format(i)])
                ids[thd_id] = i
                
                if ((thd_id == (thead_num - 1)) or (i == (TEST_SIZE - 1))):
                    for j in range(thead_num):
                        if (ths[j] is not None):
                            ths[j].wait()
                            shutil.move("log{}.txt".format(ids[j]), globals.REPLAY_SAVE_PATH + str(ids[j]))
                            trainer.update_buffer([globals.REPLAY_SAVE_PATH + str(ids[j])])
                        else:
                            break
                            
                    ths = [None for _ in range(thead_num)]
                    ids = [None for _ in range(thead_num)]
                        
                thd_id = (thd_id + 1) % thead_num
                
            trainer.test(globals.N_PLAYER)
            