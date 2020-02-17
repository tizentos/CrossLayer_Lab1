# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
import os
import numpy as np



def process_data(csv_file, embedding_size,column=None):
    
    data = csv_file
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    if column:
        sparse_features.remove(column)
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    if column:
        data = data.drop(columns = column)

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

    print(os.getcwd())
    data = pd.read_csv('./DeepCTR-Torch/examples/criteo_sample.txt')
    # test_data = pd.read_csv('./examples/test1.txt')
    target = ['label']
    i =43

    print("Tests%d.txt"%i)
    train,dnn_feature_columns,linear_feature_columns =  process_data(data,8)
    # test,_,_ = process_data(test_data,8)

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)



    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics={"binary_crossentropy","auc","accuracy"}, )
    model.fit(train_model_input, train[target].values,
              batch_size=256, epochs=10, validation_split=0.0, verbose=2)

    pred_ans = model.predict(test_model_input, 256)

    a = test[target]
    b = test[target].values

    # print("Saving model...\n")
    # torch.save(model,'Model-model.h5')
    # torch.save(model.state_dict(),"Model-weights.h5")
    # model.load_state_dict(torch.load("Model-weights.h5"))
       

    # model = torch.load('Model-model.h5')

    print(test[target].values)
    print(pred_ans)

    new_pred = np.where(pred_ans > 0.5, 1, 0)
    c = accuracy_score(test[target].values,np.where(pred_ans > 0.5, 1, 0))
    print(c)
    # acc = model.metrics["accuracy"]
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    print("test Accuracy", accuracy_score(test[target].values,np.where(pred_ans > 0.5, 1, 0))*100)

