import numpy as np

class Activation():
    def __init__(self, function):
        self.function = function
        self.name = function

    def forward(self, Z):
        if self.function == "sigmoid":
            """
            Implements the sigmoid activation in numpy
            
            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation
            
            Returns:
            A -- output of sigmoid(z), same shape as Z
            
            """

            # GRADED FUNCTION: sigmoid_forward
            ### START CODE HERE ### (≈ 8 lines of code)
            A = np.zeros(shape=[Z.shape[0],Z.shape[1]], dtype=np.float64)
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    if(Z[i][j]>=0):
                        A[i][j] = 1.0 / (1.0 + np.exp(-Z[i][j]))
                    else:
                        A[i][j] = np.exp(Z[i][j]) / (1.0 + np.exp(Z[i][j]))
            self.cache = Z
            ### END CODE HERE ###
            
            return A

        elif self.function == "softmax":
            """
            Implements the softmax activation in numpy
            
            Arguments:
            Z -- numpy array of any shape (dim 0: number of classes, dim 1: number of samples)
            self.cache -- stores Z as well, useful during backpropagation
            
            Returns:
            A -- output of softmax(z), same shape as Z
            """

            # GRADED FUNCTION: softmax_forward
            ### START CODE HERE ### (≈ 2 lines of code)  
            #???            
            self.cache = Z
            
            A = np.zeros(shape=[Z.shape[1],Z.shape[0]], dtype=np.float64)
            cnt=0
            Z = Z.transpose()
            for i in Z:
                i-=np.max(i)
                A[cnt] = np.exp(i)/np.sum(np.exp(i))
                cnt+=1
            A = A.transpose()
            ### END CODE HERE ###
            
            return A

        elif self.function == "relu":
            """
            Implement the RELU function in numpy
            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation
            Returns:
            A -- output of relu(z), same shape as Z
            
            """
            
            # GRADED FUNCTION: relu_forward
            ### START CODE HERE ### (≈ 2 lines of code)
            A = np.maximum(0,Z)
            self.cache = Z 
            ### END CODE HERE ###
            
            assert(A.shape == Z.shape)
            
            return A

    def backward(self, dA=None, Y=None):
        if self.function == "sigmoid":
            """
            Implement the backward propagation for a single SIGMOID unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """
            
            # GRADED FUNCTION: sigmoid_backward
            ### START CODE HERE ### (≈ 9 lines of code)
            Z = self.cache            
            A = Activation("sigmoid").forward(Z)
            B = np.ones(shape=[Z.shape[0],Z.shape[1]], dtype=np.float64)
                                    
            dZ = dA * (A)*(B-A)
            ### END CODE HERE ###
            
            assert (dZ.shape == Z.shape)
            
            return dZ

        elif self.function == "relu":
            """
            Implement the backward propagation for a single RELU unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """
            
            # GRADED FUNCTION: relu_backward
            ### START CODE HERE ### (≈ 3 lines of code)
            Z = self.cache
            dZ = dA
            dZ[Z <= 0] = 0
            ### END CODE HERE ###
            
            assert (dZ.shape == Z.shape)
            
            return dZ

        elif self.function == "softmax":
            """
            Implement the backward propagation for a [SOFTMAX->CCE LOSS] unit.
            Arguments:
            Y -- true "label" vector (one hot vector, for example: [[1], [0], [0]] represents rock, [[0], [1], [0]] represents paper, [[0], [0], [1]] represents scissors 
                                      in a Rock-Paper-Scissors image classification), shape (number of classes, number of examples)
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """
            
            # GRADED FUNCTION: softmax_CCE_backward
            ### START CODE HERE ### (≈ 3 lines of code)
            Z = self.cache            
            s = Activation("softmax").forward(Z)            
            dZ = s-Y
            ### END CODE HERE ###
            
            assert (dZ.shape == Z.shape)
            
            return dZ