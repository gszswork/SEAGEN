import numpy as np

if __name__ == '__main__':
    fold0_test = np.load('fold0_x_test_id.npy')
    fold0_train = np.load('fold0_x_train_id.npy')
    whole = np.concatenate((fold0_train, fold0_test), axis=0)

    length = len(whole)

    fold0_x_test = whole[:int(0.2 * length)]
    fold0_x_train = whole[int(0.2 * length):]

    fold1_x_test = whole[int(0.2 * length) : int(0.4 * length)]
    fold1_x_train = np.concatenate((whole[:int(0.2 * length)], whole[int(0.4 * length):]))

    fold2_x_test = whole[int(0.4 * length) : int(0.6 * length)]
    fold2_x_train = np.concatenate((whole[:int(0.4 * length)], whole[int(0.6 * length):]))

    fold3_x_test = whole[int(0.6 * length) : int(0.8 * length)]
    fold3_x_train = np.concatenate((whole[:int(0.6 * length)], whole[int(0.8 * length):]))

    fold4_x_test = whole[int(0.8 * length):]
    fold4_x_train = whole[:int(0.8 * length)]

    print(len(fold0_x_test), len(fold0_x_train))
    print(len(fold1_x_test), len(fold1_x_train))
    print(len(fold2_x_test), len(fold2_x_train))
    print(len(fold3_x_test), len(fold3_x_train))
    print(len(fold4_x_test), len(fold4_x_train))

    np.save('fold0_x_test.npy', fold0_x_test)
    np.save('fold0_x_train.npy', fold0_x_train)
    np.save('fold1_x_test.npy', fold1_x_test)
    np.save('fold1_x_train.npy', fold1_x_train)
    np.save('fold2_x_test.npy', fold2_x_test)
    np.save('fold2_x_train.npy', fold2_x_train)
    np.save('fold3_x_test.npy', fold3_x_test)
    np.save('fold3_x_train.npy', fold3_x_train)
    np.save('fold4_x_test.npy', fold4_x_test)
    np.save('fold4_x_train.npy', fold4_x_train)
