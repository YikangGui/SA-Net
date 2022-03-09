import os
from os import listdir
from os.path import isfile, join
import pandas as pd


PWD = './Data'


def rename(train_files, test_files):
    df_train = pd.DataFrame(columns=['frame', 'eef', 'grasp'])
    df_test = pd.DataFrame(columns=['frame', 'eef', 'grasp'])

    for train_id, file in enumerate(train_files):
        os.rename(f'{PWD}/train/{file[1]}', f'{PWD}/train/{train_id}.jpg')
        df_train.loc[train_id] = [f'{train_id}.jpg', '', '']

    del train_id

    for test_id, file in enumerate(test_files):
        os.rename(f'{PWD}/test/{file[1]}', f'{PWD}/test/{test_id}.jpg')
        df_test.loc[test_id] = [f'{test_id}.jpg', '', '']

    df_train.to_csv(f'{PWD}/train.csv', index=False, header=None)
    df_test.to_csv(f'{PWD}/test.csv', index=False, header=None)


def rename_test(train_files, test_files):
    df_train = pd.DataFrame(columns=['frame', 'eef', 'grasp'])
    df_test = pd.DataFrame(columns=['frame', 'eef', 'grasp'])

    for train_id, file in enumerate(train_files):
        os.rename(f'{PWD}/rgb_images1/{file[1]}', f'{PWD}/rgb_images1/{train_id}.jpg')
        df_train.loc[train_id] = [f'{train_id}.jpg', '', '']

    for test_id, file in enumerate(test_files):
        os.rename(f'{PWD}/rgb_images2/{file[1]}', f'{PWD}/rgb_images2/{test_id + train_id + 1}.jpg')
        df_test.loc[test_id] = [f'{test_id + train_id + 1}.jpg', '', '']

    assert PWD == '.'
    df_train.to_csv(f'{PWD}/train.csv', index=False, header=None)
    df_test.to_csv(f'{PWD}/test.csv', index=False, header=None)


if __name__ == '__main__':
    # train_pwd = f'{PWD}/train'
    # test_pwd = f'{PWD}/test'
    # train_files = sorted([(int(f[5:-4]), f) for f in listdir(train_pwd)], key=lambda x: x[0])
    # test_files = sorted([(int(f[5:-4]), f) for f in listdir(test_pwd)], key=lambda x: x[0])
    # rename(train_files, test_files)
    PWD = '.'
    train_pwd = f'{PWD}/rgb_images1'
    test_pwd = f'{PWD}/rgb_images2'
    train_files = sorted([(int(f[5:-4]), f) for f in listdir(train_pwd)], key=lambda x: x[0])
    test_files = sorted([(int(f[5:-4]), f) for f in listdir(test_pwd)], key=lambda x: x[0])
    rename_test(train_files, test_files)