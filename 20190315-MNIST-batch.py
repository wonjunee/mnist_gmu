"""

This is the same codes as 20190315-MNIST.ipynb

Not using the full size

"""


from model_20190302 import *

from helper.generate_data_mnist import *

import sys, getopt
import random
import pickle


def main(argv):

    random.seed(702)

    """--------------------------------

    # Parameters

    --------------------------------"""

    try:
        SIZE, EPOCHS, filename = argv[1:]
    except:
        print("\nYou need to specify the following arguments")
        print("MNIST.py <SIZE> <EPOCHS> <filename>")
        sys.exit()
        
    SIZE = int(SIZE)
    EPOCHS = int(EPOCHS)

    reduced = True
    padding = True

    # learning rate
    rate = 0.001

    BATCH_SIZE = 128

    # Generate the data
    X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot = mnist_generate_data(reduced=reduced, train_size = SIZE, padding=padding)

    def evaluate(X_data, y_data):
        
        num_examples = len(X_data)
        sess = tf.get_default_session()
        accuracy = sess.run([accuracy_operation,correct_prediction,logits,fc4,fc6], feed_dict={x: X_data, y: y_data})

        return accuracy

    # Matrix sizes for svm training
    mat_type_list = ["2", "4", "6"]

    tf_only  = []
    tf_svm2  = []
    tf_svm4  = []
    tf_svm6  = []
        
    # initialize x axis for plot
    epochs_ranges = []

    accuracy_svc_linear = {}

    for mat in mat_type_list:
        accuracy_svc_linear[mat] = []

    # with tf.device('/device:gpu:1'):
    x = tf.placeholder(tf.float32, (None, X_train.shape[1],X_train.shape[2],X_train.shape[3]), "x")
    y = tf.placeholder(tf.int32, (None, 10), "y")

    logits, fc4, fc6 = pipeline(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate = rate)

    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")

        for i in range(EPOCHS):
            for offset in range(0, len(y_train), BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train_one_hot[offset:end]
                output_batch = sess.run([training_operation, logits, fc4, fc6], feed_dict={x: batch_x, y: batch_y})
                
                if offset == 0:
                    Amat2,Amat4,Amat6 = output_batch[1:]
                else:
                    Amat2 = np.append(Amat2, output_batch[1], axis=0)
                    Amat4 = np.append(Amat4, output_batch[2], axis=0)
                    Amat6 = np.append(Amat6, output_batch[3], axis=0)

            if i % 2 == 0:

                print("\nEPOCHS:", i)

                print(Amat2.shape, Amat4.shape, Amat6.shape)

                epochs_ranges.append(i)

                print("\nTesting...")
                test_accuracy,prediction_tf,Amat2_test,Amat4_test,Amat6_test = evaluate(X_test, y_test_one_hot)

                print("\nTest Accuracy = {:.3f}".format(test_accuracy))
                
                
                # Appending tensorflow accuracies to the tf_only
                tf_only.append(test_accuracy)

                '''

                in this model, there is no test data.
                it is only using train data and evaluate on the train data.
                so no need to worry about dividing into two sets

                in the next for loop, I used the term train and test but
                i just copied them from CIFAR10-tf-smaller.ipynb and didn't change the words.

                '''

                print("\nTesting with SVM")
                for train_mat_type in mat_type_list: # only 6 and 16

                    if train_mat_type == "2":
                        train_matrix = Amat2.copy()
                        test_matrix  = Amat2_test.copy()
                    elif train_mat_type == "4":
                        train_matrix = Amat4.copy()
                        test_matrix  = Amat4_test.copy()
                    elif train_mat_type == "6":
                        train_matrix = Amat6.copy()
                        test_matrix  = Amat6_test.copy()                   

                    # With linear kernel
                    svc = SVC(kernel='linear')
                    svc.fit(train_matrix, y_train)
                    
                    print("Y_train:",y_train.shape)
                    print("Y_train:",y_test.shape)

                    prediction = svc.predict(test_matrix)
                    
                    print("prediction:",prediction.shape)
                    
                    svc_linear_accuracy = np.sum(prediction==y_test.ravel())/len(prediction)

                    print('\n train_mat_type: {}, Accuracy by SVC (linear): {}'.format(train_mat_type, svc_linear_accuracy))

                    accuracy_svc_linear[train_mat_type].append(svc_linear_accuracy)

                    # append accuracy to svm_only#
                    if train_mat_type == "2":
                        tf_svm2.append(svc_linear_accuracy)
                    elif train_mat_type == "4":
                        tf_svm4.append(svc_linear_accuracy)
                    elif train_mat_type == "6":
                        tf_svm6.append(svc_linear_accuracy)

        for train_mat_type in mat_type_list:
            plt.figure(figsize=(10,5))
            plt.plot(epochs_ranges, tf_only, 'o-', label="TF")
            plt.plot(epochs_ranges, accuracy_svc_linear[train_mat_type], 'o-',label="SVC (linear)")

            a = np.argmax(tf_only)
            plt.annotate(tf_only[a], (epochs_ranges[a], tf_only[a]))

            a = np.argmax(accuracy_svc_linear[train_mat_type])
            plt.annotate(accuracy_svc_linear[train_mat_type][a], (epochs_ranges[a], accuracy_svc_linear[train_mat_type][a]))

            plt.legend()
            title  = "{}-{}-{}.png".format(filename, SIZE, train_mat_type)
            title1 = title

            plt.title(title1)
            plt.grid()
            plt.savefig(str(title))

            # plt.show()

    pickle.dump({"tf_only":tf_only, "tf_svm2":tf_svm2, "tf_svm4":tf_svm4, "tf_svm6":tf_svm6}, open("{}-{}.p".format(filename, SIZE),"wb"))

if __name__ == "__main__":
    main(sys.argv)