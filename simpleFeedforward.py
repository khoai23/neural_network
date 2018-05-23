import tensorflow
# Preparing training data (inputs-outputs)  
training_inputs = tensorflow.placeholder(shape=[None, 3], dtype=tensorflow.float32)  
training_outputs = tensorflow.placeholder(shape=[None, 1], dtype=tensorflow.float32) #Desired outputs for each input  
   
# Preparing neural network parameters (weights and bias) using TensorFlow Variables  
weights = tensorflow.Variable(initial_value=[[.3], [.1], [.8]], dtype=tensorflow.float32)  
bias = tensorflow.Variable(initial_value=[[1]], dtype=tensorflow.float32)  
  
# Preparing inputs of the activation function  
af_input = tensorflow.matmul(training_inputs, weights) + bias  
  
# Activation function of the output layer neuron  
predictions = tensorflow.nn.sigmoid(af_input)  
  
# Measuring the prediction error of the network after being trained  
prediction_error = tensorflow.reduce_sum(training_outputs - predictions)  
  
# Minimizing the prediction error using gradient descent optimizer  
train_op = tensorflow.train.GradientDescentOptimizer(learning_rate=0.05).minimize(prediction_error)  
  
# Creating a TensorFlow Session  
sess = tensorflow.Session()  
  
# Initializing the TensorFlow Variables (weights and bias)  
sess.run(tensorflow.global_variables_initializer())  
  
# Training data inputs  
training_inputs_data = [[255, 0, 0],  
                        [248, 80, 68],  
                        [0, 0, 255],  
                        [67, 15, 210]]  
  
# Training data desired outputs  
training_outputs_data = [[1],  
                         [1],  
                         [0],  
                         [0]]  
  
# Training loop of the neural network  
for step in range(10000):  
    sess.run(fetches=[train_op], feed_dict={
                                   training_inputs: training_inputs_data,  
                                   training_outputs: training_outputs_data})  
  
# Class scores of some testing data  
print("Expected Scores : ", sess.run(fetches=predictions, feed_dict={training_inputs: [[248, 80, 68],
                                                                   [0, 0, 255]]}))  
 
# Closing the TensorFlow Session to free resources  
sess.close()