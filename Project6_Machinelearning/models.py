import nn
from backend import Dataset
import math

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return -1 if nn.as_scalar(self.run(x)) < 0 else 1

    def train(self, dataset: Dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        isAllCorrect = False
        
        # A while loop = 1 epoch
        while not isAllCorrect:
            isAllCorrect = True
            
            # Each weight evaluation and modification will be done upon each data point.
            for x, y in dataset.iterate_once(1):
                y = nn.as_scalar(y)

                if self.get_prediction(x) != y:
                    self.get_weights().update(x, y) # w -= x.data * y
                    isAllCorrect = False
                

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_layer_size = 512 # equivalent to number of the hidden layer's nodes (features)
        # 1 hidden layer (has weights denoted by w1 matrix)
        self.w1 = nn.Parameter(1, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)
        # 1 node in output layer (has weights denoted by w2 matrix)
        self.w2 = nn.Parameter(hidden_layer_size, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        linear_multi = nn.Linear(x, self.w1)
        z1 = nn.AddBias(linear_multi, self.b1)
        a1 = nn.ReLU(z1)

        linear_multi = nn.Linear(a1, self.w2)
        z2 = nn.AddBias(linear_multi, self.b2)
        
        return z2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)

        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 200
        learning_rate = 0.05

        # Calculate batch number of every loop
        batch_number = 0
        total_batch = math.ceil(dataset.x.shape[0] / batch_size)
        # 1 epoch = the whole training dataset, thus epoch number will be increased by 1
        # when the batch number is assigned with 1 again.
        epoch = 0

        # Determine if the loss function's value is admissible (<=0.02 - defined by autograde)
        def isAcceptable(loss_obj):
            loss_function_value = loss_obj.data

            return True if loss_function_value <= 0.02 else False
            
        
        # evaluate weights by loss function and modify them until the loss function's value is admissible
        for x, y in dataset.iterate_forever(batch_size):
            batch_number = batch_number % total_batch + 1
            
            if batch_number == 1: epoch += 1

            params = [self.w1, self.w2, self.b1, self.b2]
            gradients = nn.gradients(self.get_loss(x, y), params)
            
            for i in range(len(gradients)):
                params[i].update(gradients[i], -(learning_rate))  
            
            if isAcceptable(self.get_loss(x, y)):
                print("Total epoch: %s" % epoch)
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_layer_size = 200

        self.w1 = nn.Parameter(784, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)
        self.w2 = nn.Parameter(hidden_layer_size, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        linear_multi = nn.Linear(x, self.w1)
        z1 = nn.AddBias(linear_multi, self.b1)
        a1 = nn.ReLU(z1)
        
        linear_multi = nn.Linear(a1, self.w2)
        z2 = nn.AddBias(linear_multi, self.b2)
        
        return z2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)

        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 100
        learning_rate = 0.5

        # Calculate batch number of every loop
        batch_number = 0
        total_batch = math.ceil(dataset.x.shape[0] / batch_size)
        # 1 epoch = the whole training dataset, thus epoch number will be increased by 1
        # when the batch number is assigned with 1 again.
        epoch = 0

        # Determine if the loss function's value is admissible (<=0.02 - defined by autograder)
        def isAcceptable(): return dataset.get_validation_accuracy() >= 0.98
                
        # evaluate weights by loss function and modify them until the loss function's value is admissible
        for x, y in dataset.iterate_forever(batch_size):
            batch_number = batch_number % total_batch + 1

            if isAcceptable():
                # print("Total epochs: %s" % epoch)
                return

            if batch_number == 1: 
                epoch += 1
                print(f"Processing epoch {epoch}... Achieved validation accuracy so far: {dataset.get_validation_accuracy()}")

            # print(f"Batch number: {batch_number}")

            param_list = [self.w1, self.b1, self.w2, self.b2]
            gradients = nn.gradients(self.get_loss(x, y), param_list)
            
            for i in range(len(param_list)):
                param_list[i].update(gradients[i], -learning_rate) #param -= gradient * learning_rate


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
