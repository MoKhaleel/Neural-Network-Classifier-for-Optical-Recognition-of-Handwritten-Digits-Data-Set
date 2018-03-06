# import neural network libs
import numpy
import timeit
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')


print("\n********************\nClassifing with CNN\n********************\n")

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
trainset = numpy.loadtxt("optdigits.tra", delimiter=",")
testset = numpy.loadtxt("optdigits.tes", delimiter=",")

# split into input (data) and output (labels) variables
data = trainset[:,0:64]
labels = trainset[:,64]

data_testset = testset[:,0:64]
labels_testset = testset[:,64]
labels_test = labels_testset

# reshape to be [samples][pixels][width][height]
data = data.reshape(data.shape[0], 1, 8, 8).astype('float32')
data_testset = data_testset.reshape(data_testset.shape[0], 1, 8, 8).astype('float32')

# one hot encode outputs
labels = np_utils.to_categorical(labels)
labels_testset = np_utils.to_categorical(labels_testset)
num_classes = labels_testset.shape[1]

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 8, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model  
    

# validation
    
# Fit the model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=10, batch_size=20, verbose=0)

# evaluate the model using kFold cross validation with 20% of the data for testing and 80% for training
kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(estimator, data, labels, cv=kfold)
print("\nOverall Validation accuracy of CNN: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# start the timer for training the neural network
start = timeit.default_timer()

# testing

# build the model
model = baseline_model()

# Fit the model
model.fit(data, labels, validation_data=(data_testset, labels_testset), epochs=10, batch_size=20, verbose=0)

# Final evaluation of the model
scores = model.evaluate(data_testset, labels_testset, verbose=0)
print("\nThe Overall accuracy of the nerual network with the test dataset of CNN:  %.2f" % (scores[1]*100))

# stop the timer after the neural network has been trained and tested
stop = timeit.default_timer()


# accuracy per category
model.fit(data, labels, epochs=10, batch_size=20, verbose=0)
predictions = model.predict(data_testset)
#print("\nPredeiction of CNN: \n" , predictions)
#print("\nThe actual labels of the test set of CNN:\n" , labels_test)

finalPredictions = []
for i in range (0, predictions.shape[0]):
    index = 0
    for j in range (0, num_classes):
        if (0.5 < predictions[i,j]):
            index = j
    finalPredictions.append(index)     

countPredectCat = [0,0,0,0,0,0,0,0,0,0]
countActualCat = [0,0,0,0,0,0,0,0,0,0]
for i in range (0, len(finalPredictions)):
    if (finalPredictions[i] == labels_test[i]):
        countPredectCat[int(finalPredictions[i])] += 1
    countActualCat[int(labels_test[i])] += 1
        
for i in range (0, len(countPredectCat)):
    print("\nThe accuracy of category ", i , " of CNN equal:  %.2f" % ((countPredectCat[i]/countActualCat[i])*100))

# build the confusion matrix after classifing the test data
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, finalPredictions)
print ("\nThe confusion matrix of CNN when apply the test set on the trained nerual network:\n" , cm)

print("\nThe total time of training the convolutional neural network: " , stop - start , "second")