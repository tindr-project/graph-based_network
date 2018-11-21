# Copyright 2018 H. Gaspar hagax8@gmail.com
import numpy as np
import keras
import keras.backend as K
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def FloatOrZero(value):
    try:
        return int(value)
    except:
        return 0


def ProteinModel(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(32, (5, 5), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs=X_input, outputs=X, name='ProteinModel')
    return model


def proteinCNN(X, Y, output):
    #X_train, X_test, y_train, y_test = train_test_split(
    #                                    X, Y, test_size=0.10, random_state=42)
    #X = X_train
    #Y = y_train
    #print(X_train.shape)
    #print("Excluding 20%% of the data for future testing...")
    #print(X_test.shape)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    cvscores = []
    for train, test in kfold.split(X, Y):
        proteinModel = ProteinModel(X.shape[1:])
        proteinModel.compile(loss="binary_crossentropy",
                             optimizer="Adam", metrics=["accuracy"])
        proteinModel.fit(x=X[train], y=Y[train],
                         epochs=100, batch_size=20, verbose=0)
        preds = proteinModel.evaluate(x=X[test], y=Y[test], verbose=0)
        print("%s: %.2f%%" % (proteinModel.metrics_names[1], preds[1]*100))
        cvscores.append(preds[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    #prediction = proteinModel.predict(X[test])
    #y_classes = np.round(prediction).astype(int)
    #np.savetxt(output+"_predictions.txt", prediction)
    #np.savetxt(output+"_classes.txt", y_classes)
    # print()
    #print ("Loss = " + str(preds[0]))
    #print ("Test Accuracy = " + str(preds[1]))
    plot_model(proteinModel, to_file=output+'model.png')
