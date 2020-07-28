import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

def get_best_accuracy(input_data, output_data, attempts):
    best = 0
    best_model = None
    for _ in range(attempts):
        train_data, test_data, train_out, test_out = sklearn.model_selection.train_test_split(input_data, output_data, test_size = 0.1)
        linear = linear_model.LinearRegression()
        linear.fit(train_data, test_data)
        accuracy = linear.score(train_out, test_out)
        if accuracy > best:
            best = accuracy
            print (best)
            best_model = linear
    return best, best_model

col_names=["mpg", "cylinders", "displacement", "horsepower", "weight", 
        "acceleration", "model_year", "origin", "car_name"]
data = pd.read_table("auto-mpg.data", sep = " ", names = col_names)

predict = "mpg"

# Set the training data
input_data = np.array(data.drop([predict, "car_name"], 'columns')).astype(int)
output_data = np.array(data[predict]).astype(int)

#best, linear = get_best_accuracy(input_data, output_data, 100000)

#with open("car_mpg_model.pickle", "wb") as f:
#    pickle.dump(linear, f)

pickle_in = open("car_mpg_model.pickle", "rb")
linear = pickle.load(pickle_in)

train_data, test_data, train_out, test_out = sklearn.model_selection.train_test_split(input_data, output_data, test_size = 0.1)

#predictions = linear.predict(test_data)
#for i in range(len(predictions)):
#    print(predictions[i], test_data[i], test_out[i])

# Plotting

style.use("seaborn")

print (data.head())

attr = "model_year"

pyplot.scatter(data[attr], data[predict])
pyplot.xlabel(attr)
pyplot.ylabel(predict)
pyplot.show()

