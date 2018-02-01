from cnn import *

#Main function
if __name__ == "__main__":

    saver = new_training_graph("test-graph")
    sess = tf.Session()


    #Ensure variables initialised
    sess.run(tf.global_variables_initializer())


    save_model("test-graph", sess, saver)
    sess.close()
