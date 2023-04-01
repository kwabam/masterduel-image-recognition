from oneShotLearning import *

if __name__ == '__main__':
    train_folder = "./cardDatabase"
    eval_folder = "./evalSimilar"

    images_train, classes_train, lang_dict_train = loadimgs(train_folder)
    images_eval, classes_eval, lang_dict_eval = loadimgs(eval_folder)
    print(images_train.shape)
    print(images_eval.shape)

    print(classes_train.shape)
    print(classes_train)

    # Let's create and compile the model with an optimizer
    model = get_siamese_cnn_model((images_train.shape[2], images_train.shape[3], images_train.shape[4]))

    learning_rate = 0.001  # You can adjust this value as needed
    beta_1 = 0.9  # The default value for the first moment estimate, you can adjust if needed
    beta_2 = 0.999  # The default value for the second moment estimate, you can adjust if needed
    epsilon = 1e-07  # A small constant for numerical stability, you can adjust if needed

    adam_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    model.compile(loss="binary_crossentropy", optimizer=adam_optimizer)

    model = train_model(model, images_train, images_eval)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
