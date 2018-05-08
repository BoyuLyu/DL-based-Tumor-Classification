#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN
from skimage import io
from sklearn.feature_selection import VarianceThreshold


def feature_selection(big_table, annotation_path, preprocessed_file_path):
    feature_name = list(big_table)
    feature_name = feature_name[1:]
    labels = np.array(big_table.iloc[:, 0])
    annotation = pd.read_csv(annotation_path, dtype=str)
    gene_id_annotation = list(annotation.loc[:, "GeneID"])
    gene_id_chr_annotation = list(annotation.loc[:, "chromosome"])
    gene_id_original = []
    idx1 = []
    idx1_annotation = []
    k = 0
    for name in feature_name:
        symbol, gene_id = name.split("|", 1)
        gene_id_original.append(gene_id)
        if gene_id in gene_id_annotation:
            idx1.append(k)
            idx1_annotation.append(gene_id_annotation.index(gene_id))
        print('compare with annotation, progress :', k / len(feature_name))
        k = k + 1
    features_raw = np.array(big_table.iloc[:, 1:], dtype=float)
    features = np.log2(1.0 + features_raw)
    features[np.where(features <= 1)] = 0
    # values corresponding to existing genes (1st filtering)
    features_filtered = np.array(list(features[:, idx1_tmp] for idx1_tmp in idx1)).transpose()
    feature_name_filtered = list(feature_name[i] for i in idx1)
    gene_id_chr = list(gene_id_chr_annotation[i] for i in idx1_annotation)
    # sort the features based on the chr number
    idx_sorted = sort_feature(gene_id_chr)
    feature_name_sorted = list(feature_name_filtered[j] for j in idx_sorted)
    features_sorted = np.array(list(features_filtered[:, i] for i in idx_sorted)).transpose()
    print('features have been sorted based on chromosome')
    selector = VarianceThreshold(threshold=6)
    selector.fit(features_sorted)
    idx2 = selector.get_support()
    idx2_num = selector.get_support(indices=True)
    # numpy is different from list
    features = features_sorted[:, idx2]
    feature_name_final = list(feature_name_sorted[i] for i in idx2_num)
    feature_name_path = os.path.join(preprocessed_file_path, 'feature_name.csv')
    pd.DataFrame(feature_name_final).to_csv(feature_name_path)
    print('features are selected, the selected gene name are saved at', feature_name_path)
    return features, labels


def sort_feature(chr_filtered):
    idx_list = list(range(len(chr_filtered)))
    big_list_chr = zip(idx_list, chr_filtered)  # combine a list of indices and a list of chromosomes
    tmp = sorted(big_list_chr, key=lambda x: x[1])  # sort based on the chromosome
    idx, _ = zip(*tmp)
    return idx


def kfold_split(big_table, annotation_path, preprocessed_file_path, folds):
    features, labels = feature_selection(big_table, annotation_path, preprocessed_file_path)
    print('features have been read')
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    index_list = list(skf.split(features, labels))
    return features, labels, index_list


def over_sampling(training_data, training_label):
    """
    This function is used only for classification
    :param training_data:
    :param training_label:
    :return: The training data and training label after oversampling using SMOTE algorithm
    """
    training_data_resampled, training_label_resampled = SMOTE(ratio='minority', random_state=42, kind='svm', n_jobs=12)\
        .fit_sample(training_data, training_label)
    return training_data_resampled, training_label_resampled


def embedding_2d(features_train, features_test, fold_idx, preprocessed_file_path, labels_train, labels_test):
    print('embedding to 2D image, fold:', fold_idx)
    features_padded = np.zeros(102*102)
    subfolder_path = os.path.join(preprocessed_file_path + '/img_fold' + str(fold_idx))
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    subfolder_train_path = subfolder_path + '/train'
    if not os.path.exists(subfolder_train_path):
        os.makedirs(subfolder_train_path)
    subfolder_test_path = subfolder_path + '/test'
    if not os.path.exists(subfolder_test_path):
        os.makedirs(subfolder_test_path)
    pd.DataFrame(labels_train).to_csv(os.path.join(subfolder_train_path, 'labels_train.csv'))
    pd.DataFrame(labels_test).to_csv(os.path.join(subfolder_test_path, 'labels_test.csv'))
    for i in range(features_train.shape[0]):
        features_padded[range(features_train.shape[1])] = features_train[i, :]/max(features_train[i, :])
        features_train_tmp = features_padded.reshape(102, 102)
        file_path = subfolder_train_path + '/'+str(i) + '.png'
        io.imsave(file_path, features_train_tmp)
    for j in range(features_test.shape[0]):
        features_padded[range(features_test.shape[1])] = features_test[j, :]/max(features_test[j, :])
        features_test_tmp = features_padded.reshape(102, 102)
        file_path = subfolder_test_path + '/' + str(j) + '.png'
        io.imsave(file_path, features_test_tmp)




