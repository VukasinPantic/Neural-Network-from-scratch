import nnfs.datasets
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from nnfs.datasets import sine_data 
import os
import cv2

nnfs.init() #set default data type for numpy use, ovo je zbog toga da dobijemo iste vrednosti kao sam ovim 

class Layer_Dense:

    def __init__(self,n_inputs,n_neurons,weight_regularizer_l1=0,weight_regularizer_l2=0,bias_regularizer_l1=0,
                 bias_regularizer_l2=0):

        self.weights = 0.01*np.random.randn(n_inputs,n_neurons) # za input size misli na broj neurona 
        self.biases = np.zeros((1,n_neurons))
        
        #Setting lambda values
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    
    def forward(self,inputs,training):
        
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
        
    def backward(self,dvalues):

        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        


        # Gradients for regularization

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2*self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues,self.weights.T) 

class Activation_ReLU:

    def forward(self,inputs,training):

        self.output = np.maximum(0,inputs)
        self.inputs = inputs

    def backward(self,dvalues):

        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0] = 0

    def predictions(self,outputs):
        return outputs

class Activation_Sigmoid:

    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = 1/(1+np.exp(-inputs))
    
    def backward(self,dvalues):
        self.dinputs = dvalues * (1-self.output) * self.output

    def predictions(self,outputs):
        return (outputs > 0.5) * 1

class Activation_Softmax:

    def forward(self,inputs,training):

        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities

    def backward(self,dvalues):

        self.dinputs = np.empty_like(dvalues)
        for index,(single_output,single_dvalues) in enumerate(zip(self.single_output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
        
    def predictions(self,outputs):
        return np.argmax(outputs,axis=1)

class Loss:
    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self,output,y,*,include_regularization=False):

        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self,*,include_regularization=False):
        
        data_loss = self.accumulated_sum / self.accumulated_count
        
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularization_loss(self):

        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1>0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(layer.weights*layer.weights)

            if layer.weight_regularizer_l2>0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights*layer.weights)

            if layer.bias_regularizer_l1>0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(layer.biases*layer.biases)

            if layer.bias_regularizer_l2>0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases*layer.biases)
        return regularization_loss


    # def regularization_loss(self,layer):

    #     regularization_loss =0 
        
    #     if layer.weight_regularizer_l1>0:
    #         regularization_loss += layer.weight_regularizer_l1 * np.sum(layer.weights*layer.weights)

    #     if layer.weight_regularizer_l2>0:
    #         regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights*layer.weights)

    #     if layer.bias_regularizer_l1>0:
    #         regularization_loss += layer.bias_regularizer_l1 * np.sum(layer.biases*layer.biases)


    #     if layer.bias_regularizer_l2>0:
    #         regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases*layer.biases)
        
    #     return regularization_loss
    
class Loss_CategoricalCrossentropy(Loss):

    def forward(self,y_pred,y_true):

        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidecens = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidecens = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidecens)
        return negative_log_likelihoods
    
    def backward(self,dvalues,y_true):
        #dvalues su ovde izlazi iz softmax-a
        samples = len(dvalues)
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true/dvalues;
        self.dinputs = self.dinputs/samples

class Activation_Softmax_Loss_CategoricalCrossentropy():

    def backward(self,dvalues,y_true):
        
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -=1
        self.dinputs = self.dinputs/samples
        

class Optimizer_SGD:

    def __init__(self,learning_rate=1.0,decay=0.,momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1./(1.+self.decay*self.iterations))

    def update_params(self,layer):

        if self.momentum:

            if not hasattr(layer,'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate*layer.dweights
            bias_updates = -self.current_learning_rate*layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates

        layer.weights += -self.current_learning_rate*layer.dweights
        layer.biases += -self.current_learning_rate*layer.dbiases
    
    def post_update_params(self):
        self.iterations +=1




class Optimizer_Adagrad:

    def __init__(self,learning_rate=1.0,decay=0.,epsilon=1e-7):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.learning_rate*(1./(1.+self.decay*self.iterations))

    def update_params(self,layer):

        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases/(np.sqrt(layer.bias_cache)+self.epsilon)
        
    def post_update_params(self):
        self.iterations +=1



class Optimizer_RMSprop:

    def __init__(self,learning_rate=0.001,decay=0.,epsilon=1e-7,rho=0.9):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.learning_rate*(1./(1.+self.decay*self.iterations))

    def update_params(self,layer):


        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho)*layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho)*layer.dbiases**2

        layer.weights += -self.current_learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases/(np.sqrt(layer.bias_cache)+self.epsilon)
        
    def post_update_params(self):
        self.iterations +=1


class Optimizer_Adam:

    def __init__(self,learning_rate=0.001,decay=0.,epsilon=1e-7,rho=0.9,beta1=0.9,beta2=0.999):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        self.beta_1 = beta1
        self.beta_2 = beta2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1./(1.+self.decay*self.iterations))

    def update_params(self,layer):

        if not hasattr(layer,'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums/(1-self.beta_1**(self.iterations+1))
        bias_momentus_corrected = layer.bias_momentums/(1-self.beta_1**(self.iterations+1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache/(1-self.beta_2**(self.iterations+1))
        bias_cache_corrected = layer.bias_cache/(1-self.beta_2**(self.iterations+1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected/(np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentus_corrected/(np.sqrt(bias_cache_corrected)+self.epsilon)


    def post_update_params(self):
        self.iterations +=1


class Layer_Dropout:

    def __init__(self,rate):
        self.rate = 1-rate
    
    def forward(self,inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1,self.rate, size=inputs.shape)/self.rate 
        self.output = inputs * self.binary_mask
    
    def backward(self,dvalues):
        self.dinputs = dvalues * self.binary_mask


class Loss_BinaryCrossentropy(Loss):
    

    def forward(self,y_pred,y_true):

        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1-y_true) * np.log(1-y_pred_clipped))
        sample_losses = np.mean(sample_losses,axis=-1)

        return sample_losses
    
    def backward(self,dvalues,y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues,1e-7,1-1e-7)
        self.dinputs = -(y_true/clipped_dvalues - (1-y_true)/(1-clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples

class Activation_Linear:

    def forward(self,inputs,training):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self,outputs):
        return outputs

class Loss_MeanSquaredError(Loss):

    def forward(self,y_pred,y_true):

        sample_losses = np.mean((y_true-y_pred)**2,axis=-1)
        return sample_losses
    
    def backward(self,dvalues,y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true-dvalues)/outputs
        self.dinputs = self.dinputs/samples

class Loss_MeanAbsoulteError(Loss):

    def forward(self,y_pred,y_true):

        sample_losses = np.mean(np.abs(y_true-y_pred),axis=-1)
        return sample_losses
    
    def backward(self,dvalues,y_true):
        
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true-dvalues)/outputs
        self.dinputs = self.dinputs/samples
     

class Layer_Input:

    def forward(self,inputs,training):
        self.output = inputs

class Accuracy:

    def calculate(self, predictions, y):

        comparasions = self.compare(predictions, y)
        accuracy = np.mean(comparasions) # treats ​True​ as ​1​ and ​False​ as ​0

        self.accumulated_sum += np.sum(comparasions)
        self.accumulated_count += len(comparasions)

        return accuracy
    
    def calculate_accumulated(self):

        accuracy =self.accumulated_sum / self.accumulated_count

        return accuracy
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
class Accuracy_Categorical(Accuracy):

    def init(self,y):
        pass
    
    def compare(self,predictions,y):
        if len(y.shape) == 2:
            y = np.argmax(y,axis=1)
        return predictions == y

class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self,y,reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y)/250

    def compare(self,precitions,y):
        return np.absolute(precitions-y) < self.precision
    

class Model:

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self,layer):
        self.layers.append(layer)
    
    def set(self,*,loss,optimizer,accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):

        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count-1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i],'weights'):
                self.trainable_layers.append(self.layers[i])
        
        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
    
    def forward(self,X,training):

        self.input_layer.forward(X,training)
        for layer in self.layers:
            layer.forward(layer.prev.output,training)
    
        return layer.output
    
    def backward(self,output,y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output,y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        self.loss.backward(output,y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self,X,y,*,epochs=1,batch_size=None,print_every=1,validation_data=None):
        
        self.accuracy.init(y)
        train_steps = 1
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
        
        if batch_size is not None:
            train_steps = len(X) // batch_size

            if train_steps*batch_size < len(X):
                train_steps += 1
            
            if validation_data is not None:
                validation_steps = len(X_val) //  batch_size

                if validation_steps*batch_size < len(X_val):
                    validation_steps += 1


        for epoch in range(1,epochs+1):
            
            print(f'epoch: {epoch}')
            
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X,training=True)

                data_loss, regularization_loss = self.loss.calculate(output,batch_y,include_regularization=True)

                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output) # Prediction method depends on type of layer
                accuracy = self.accuracy.calculate(predictions,batch_y)

                self.backward(output,batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'epoch: {epoch}, ' + 
                        f'acc: {accuracy:.3f}, ' + 
                        f'loss: {loss:.3f} ' + 
                        f'data_loss: {data_loss:.3f}, ' + 
                        f'reg_loss: {regularization_loss:.3f}, '+ 
                        f'lr: {self.optimizer.current_learning_rate} ')
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' + 
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' + 
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')
            
            if validation_data is not None:
                
                self.loss.new_pass()
                self.accuracy.new_pass()

                for step in range(validation_steps):
                    
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        batch_X = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_val[step*batch_size:(step+1)*batch_size]
                    
                    output = self.forward(batch_X, training=False)
                    self.loss.calculate(output,batch_y)
                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions,batch_y)
                    validation_loss = self.loss.calculate_accumulated()
                    validation_accuracy = self.accuracy.calculate_accumulated()

                    print(f'validation, '+
                          f'acc: {validation_accuracy}, '+
                          f'loss: {validation_loss}')
                    

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path,dataset))
    x = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path,dataset,label)):

            image = cv2.imread(os.path.join(path,dataset,label,file), cv2.IMREAD_UNCHANGED)
            x.append(image)
            y.append(label)

def create_data_mnist(path):
    X,y = load_mnist_dataset('train',path)
    X_test, y_test = load_mnist_dataset('test',path)
    return X,y,X_test,y_test

