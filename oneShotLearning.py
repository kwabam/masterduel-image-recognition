# importing keras modules
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Input
from keras import backend as K
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense

# importing other modules
import time
import numpy as np
from sklearn.utils import shuffle
import numpy.random as rng
import os
import matplotlib.pyplot as plt
from PIL import Image

def loadimgs(path, n=0):
    X = []
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    fixed_size = (105, 153)
    num_channels = 3

    alphabet = "yugioh"
    print("loading alphabet: " + alphabet)
    lang_dict[alphabet] = [curr_y, None]

    for letter in os.listdir(path):
        cat_dict[curr_y] = (alphabet, letter)
        category_images = []
        letter_path = os.path.join(path, letter)

        for filename in os.listdir(letter_path):
            image_path = os.path.join(letter_path, filename)
            image = Image.open(image_path).convert('RGB').resize(fixed_size)
            img_array = np.array(image).reshape(*fixed_size, num_channels)
            category_images.append(img_array)
            y.append(curr_y)
        X.append(np.stack(category_images))
        curr_y += 1
        lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X, y, lang_dict



def plot_images(ch_num, images):
    """
        Plot all 20 samples of a particular character
    """
    images_list = images[ch_num,:,:,:]
    num_rows, num_cols = 5, 4
    num_images = images_list.shape[0]

    f, axarr = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for i in range(num_rows):
        for j in range(num_cols):
            img_idx = i * num_cols + j
            if img_idx < num_images:
                axarr[i, j].imshow(images_list[img_idx, :, :])
            else:
                axarr[i, j].axis('off')

    plt.show()


def get_batch(batch_size, images):
    """
    Create batch of n pairs, half same class, half different class
    """
    n_classes, n_examples, w, h, num_channels = images.shape

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, w, h, num_channels)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        category = rng.choice(n_classes)
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = images[category, idx_1]
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to
            # ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = images[category_2, idx_2]

    return pairs, targets



def plot_batch(batch_pairs):
    """
        Plot all pairs of a particular batch
    """
    f, axarr = plt.subplots(batch_pairs[0].shape[0], 2, figsize=(100, 100))
    for i in range(batch_pairs[0].shape[0]):
        for j in range(2):
            axarr[i, j].imshow(batch_pairs[j][i, :, :, 0])
    plt.show()

def generate(batch_size, s="train"):
    """
    a generator for batches, so model.fit_generator can be used.
    """
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)


def get_siamese_model(model, input_shape):
    """
        Siamese network architecture created from passed base model
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


def get_base_cnn_model(input_shape):
    """
        Base CNN model architecture
    """
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64,
                     (10, 10),
                     activation='relu',
                     input_shape=input_shape,
                     kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128,
                     (7, 7),
                     activation='relu',
                     kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128,
                     (4, 4),
                     activation='relu',
                     kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256,
                     (4, 4),
                     activation='relu',
                     kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096,
                    activation='sigmoid',
                    kernel_regularizer=l2(1e-3)))

    return model


def get_siamese_cnn_model(input_shape):
    cnn_model = get_base_cnn_model(input_shape)
    siamese_model = get_siamese_model(cnn_model, input_shape)
    return siamese_model


def make_oneshot_task(N, images):
    """
    Create pairs of test image, support set for testing N way one-shot learning.
    """
    n_classes, n_examples, w, h, num_channels = images.shape

    indices = rng.randint(0, n_examples, size=(N,))

    categories = rng.choice(range(n_classes), size=(N,), replace=False)

    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
    test_image = np.asarray([images[true_category, ex1, :, :]] * N).reshape(N, w, h, num_channels)


    support_set = images[categories, indices, :, :]
    support_set[0, :, :] = images[true_category, ex2]
    support_set = support_set.reshape(N, w, h, num_channels)

    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image, support_set]
    return pairs, targets


def test_oneshot(model, N, k, images, verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N, images)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
    return percent_correct


def train_model(
        model,
        images_train,
        images_eval,
        evaluate_every=10,  # interval for evaluating on one-shot tasks
        loss_every=20,  # interval for printing loss (iterations)
        batch_size=32,
        n_iter=20000,
        N_way=20,  # how many classes for testing one-shot tasks>
        n_val=250,  # how many one-shot tasks to validate on?
        best=-1,
        model_path="./models"
):
    print("Starting training process!")
    print("-------------------------------------")
    weights_path = os.path.join(model_path, "model_weights.h5")

    # Create the directory if it doesn't already exist
    os.makedirs(model_path, exist_ok=True)

    t_start = time.time()
    for i in range(1, n_iter):
        (inputs, targets) = get_batch(batch_size, images_train)
        loss = model.train_on_batch(inputs, targets)
        print("\n ------------- \n")
        print("Loss: {0}".format(loss))
        if i % evaluate_every == 0:
            print("Time for {0} iterations: {1}".format(i, time.time() - t_start))
            val_acc = test_oneshot(model, N_way, n_val, images_eval, verbose=True)
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                print("Saving weights to: {0} \n".format(weights_path))
                model.save_weights(weights_path)
                best = val_acc

        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i, loss))

    model.load_weights(weights_path)
    return model
