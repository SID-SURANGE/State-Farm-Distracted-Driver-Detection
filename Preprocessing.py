# coding = utf-8

import imports


def pre_process(training_data):
    training_data = imports.random.shuffle(training_data)
    x = [], y = []

    for features, label in training_data:
        x.append(features)
        y.append(label)

    y_cat = imports.np_utils.to_categorical(y, num_classes=10)

    X = imports.np.array(x).reshape(-1, 240, 240, 1)
    print(X.shape)

    x_train, x_test, y_train, y_test = imports.train_test_split(X, y_cat, test_size=0.3, random_state=50)
    print("Shape of train images is:", x_train.shape)
    print("Shape of validation images is:", x_test.shape)
    print("Shape of labels is:", y_train.shape)
    print("Shape of labels is:", y_test.shape)

    return x_train, x_test, y_train, y_test
