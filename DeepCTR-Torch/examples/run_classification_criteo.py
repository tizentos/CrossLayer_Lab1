# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
import numpy as np



def process_data(csv_file, embedding_size, column = None):
    
    data = csv_file
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    if column:
        sparse_features.remove(column)
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    if column:
        data = data.drop(columns=column)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),embedding_dim=embedding_size)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns


    return data,dnn_feature_columns,linear_feature_columns


if __name__ == "__main__":

    data = pd.read_csv('./train1.txt')
    test_data1 = pd.read_csv('./test1.txt')
    test_data2 = pd.read_csv('./test2.txt')
    test_data3 = pd.read_csv('./test3.txt')
    target = ['label']

    columns = ['C2','C14','C18','C19','C22','C24']


    for column in columns:

        train,dnn_feature_columns,linear_feature_columns =  process_data(data,128,column=column)
        test1,_,_ = process_data(test_data1,128, column=column)
        test2,_,_ = process_data(test_data2,128,column=column)
        test3,_,_ = process_data(test_data3,128,column=column)

        tests =[test1,test2,test3]

        batch_size = 256

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model

        # train, test = train_test_split(data, test_size=0.2)


        train_model_input = {name: train[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate

        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    task='binary',
                    l2_reg_embedding=1e-5, device=device)

        model.compile("adagrad", "binary_crossentropy",
                    metrics=["binary_crossentropy", "auc","accuracy"], )
        model.fit(train_model_input, train[target].values,
                batch_size=batch_size, epochs=10, validation_split=0.0, verbose=2, use_double= True)
        i = 1
        print("******************COLUMN %s*******************"%column)
        for test in tests:

            print("For test%d.txt"%i)
            test_model_input = {name: test[name] for name in feature_names}

            pred_ans = model.predict(test_model_input, batch_size,use_double=True)
            acc = round(accuracy_score(test[target].values, np.where(pred_ans > 0.5,1,0)),4)

            print("")
            print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
            print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
            print("test Accuracy", accuracy_score(test[target].values, np.where(pred_ans > 0.5, 1,0))*100)
            i+=1
        print("*****************************************************************************")
