from cnn import *

#Main function
if __name__ == "__main__":

    saver = new_training_graph(sys.argv[1])
    sess = tf.Session()


    #Ensure variables initialised
    sess.run(tf.global_variables_initializer())


    save_model(sys.argv[1], sess, saver)
    sess.close()
