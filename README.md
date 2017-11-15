# Perceptron_algorithm

How perceptron works? An implementation from the scratch

Perceptron
Perceptron is the popular traditional  supervised binary classification technique that performs a linear mapping of input feature values to identify the class label. Perceptron can be treated as a basic building block of a neural network called nodes. The node performs a weighted sum of input feature values together with a bias term on which a non-linear mapping is performed using activation functions to produce the output. A simple perceptron (Fig. 2) performs linear transformation of input features and a sign thresholding function is used to fire the class label.  The learned decision boundary will be a line in two dimensional feature space. The decision boundary acts like a plane if the input features are three tuple vectors. When the dimension of input feature is more than three, the decision boundary becomes a hyperplane in a higher dimensional feature space (beyond imagination). A simple perceptron with a single node works well only if the data points are linearly separable.  Figure below shows examples for linearly separable data points and non-linearly separable data points.
The python code for generating the below plots are available at https://github.com/AneeshCEN/data_generation/blob/master/data_generation.py

(a)                                                                            (b)

(c)
Fig 1. Data points distribution examples for binary classification scenarios 

			Fig 2. Architecture of simple perceptron
When the data is nonlinear in nature as shown in Fig 1.b and Fig 1.c, the simple perceptron will fail. Hence, multilayer perceptron was introduced. However, for two class problems (or binary class problems) where data is linearly distinguishable, simple perceptron will work.  Considering the case of simple perceptron, the input data vector  . Each is an n-tuple vector in the n-dimensional input space. Given m set of  with known labels , algorithm learns an optimum weight vector such that,
For a new sample , algorithm predicts the label as +1 if , else predict the label as -1.
ie 

Where , 
                                                               
Expanding the equation for hypothesis representation,

Hypothesis representation predicts the label as positive when  ie,

Taking theta to the left hand side, equation becomes,

Replacing the bias term theta with ,

Perceptron fires positive class label for a feature vector , when the above equation gets satisfied, else it predicts the label as negative.


How the decision boundary can be visualized in 2-d space?
Let as consider a training set of 2-d samples each , as shown in the figure below

Fig 2. Representation of decision boundary in 2-d input features 
The hypothesis representation of perceptron predicts the class label +1 if

The process of training with known labels associated with  derived the model parameter  as
= 1,  = 1, and = 4
Then the equation becomes,
Predicts  when 
Or (1). 
And 
Predicts  when 
Or     (2)
Neuron will predict class label for all the newly coming previously unknown  samples if its satisfies the above equation (1) and will predict negative label if it satisfies equation (2)
The straight line  represents the linear decision boundary shown in the figure No 3. 

How the weights are being updated?
The process of training is an iterative procedure which identifies the optimum weight vector such that the learned weight vector minimizes the error difference between the hypothesis predicted value and actual value over the training data. 
The whole process can be explained as,
1. Initialize the number of iteration, learning rate eta, Initialize the weight vector w to all zeros or random numbers. The length of the weight vector should be equal to one plus the dimension of feature vector. Additional co-efficient represents the bias term. 
2. For every training sample, calculate the error difference between actual label and  hypothesis predicted label and update the weight as,

Where eta is the learning rate usually in between 0 and 1. This factor controls the rate of change for the weight vector. 

What is the intuition behind the above equation? Let us examine it in different scenarios
In the case of false positives 
This is the case when the algorithm predicts the class label as +1 while the actual label of the sample is   -1. As we know, the algorithm predicts the label as +1 only when  is positive,
 can be written as 
 is always positive, and  becomes positive only when theta is in between 0 and 90 degree. Means the angle between  and will be in between 0 and 90 degree. When the training happens weights get updated as,

Where  is the actual label -1 and  is the predicted label +1. Substituting this to 

In effect, the weights get decreased and the direction of weight vector changes such that the angle between  and  gets adjusted such that  becomes negative.  returns negative when angle between  and  is in between 90 and 180 and the hypothesis returns -1 in the next iteration . The weight updating process can be illustrated as below.
                                                 
                                                       
                                                                                          
                                                                                                     

Fig :- Transformation for weight vector while training (False positive scenario)
In the case of false negatives
This is the scenario when the algorithm predicts -1 while the actual label is +1. In such cases
Perceptron’s hypothesis function  will be negative ie,

 can’t be negative,  will only be negative when  is negative, means when theta is in between 0 and 90. 
The weight update equation when false negative case is,

Where - the actual label +1 and  the predicted label is -1, so  becomes,


The training process increases the weight vector to make the angle between w and x is in between 90 and 180 degree and  becomes positive, becomes positive and the algorithm classifies the sample correctly in the next iteration.
The illustration of weight updation process can be represented as below,
                                                   


                   
Fig :- Transformation for weight vector while training (False negative scenario)
In the case of true positive and true negative
This is the case when hypothesis function correctly predicts the class label and the weight vector doesn’t get updated.
True positive scenarios:
                
True negative scenarios:

In each iteration, this process of weight adjustment happens for all of the training samples  and an optimum weight vector is derived such that false positive and negative samples gets reduced (All the samples gets correctly separated).
Figure below shows the learned decision boundary for linearly separable data points. Black line indicates the decision boundary learned without using library and the green one indicates the learned decision boundary using scikit learn python library.  Red colored sample below the black decision boundary is an example for false negative sample. 

The entire python code for perceprtron algorithm is available at
https://github.com/AneeshCEN/Perceptron_algorithm
