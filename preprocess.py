import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to original data csv file.", type=str, default="./aug_train.csv")
    parser.add_argument("--save_path", help="Save path for preprocessed csv file.", type=str, default="./aug_train_preprocessed.csv")
    parser.add_argument("--method", help="Use one hot encoder or label encoder. (one_hot or label_encode)", type=str, default="one_hot")
    return parser


parser = get_parser()
args = parser.parse_args()
data_path = args.data_path
save_path = args.save_path
method = args.method

data = pd.read_csv(data_path)
data = data.fillna(data.mode().iloc[0])

encode_labels = ["city", "gender", "relevent_experience", "enrolled_university", "education_level", "major_discipline", "experience", "company_size", "company_type", "last_new_job"]

if method == "one_hot":
    for c in encode_labels:
        onehot_encode = OneHotEncoder()
        transformed = onehot_encode.fit_transform(data[[c]])
        data[onehot_encode.categories_[0]] = transformed.toarray()
    final_data = data.drop(encode_labels, axis=1)
    final_data.drop(["enrollee_id"], axis=1, inplace=True)
    final_data.to_csv(save_path, index=False)

else:
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()

    # Define the Labels to be Encoded
    encode_labels = ["gender", "relevent_experience", "enrolled_university", "education_level", "major_discipline", "experience", "company_size", "company_type", "last_new_job"]

    # Encoding
    to_encode = data[encode_labels]
    data_temp = to_encode.astype("str").apply(label_encoder.fit_transform)  # encode given labels
    data["city"] = data["city"].apply(lambda x: int(x.split("_")[1]))  # replace 'city_xxx' (str) -> xxx (int)
    data.drop(encode_labels, axis=1, inplace=True)

    encoded_data = data_temp.join(data)  # combine encoded columns with other columns

    # Scaling
    scaled_data = scaler.fit_transform(encoded_data)  # scale data to range [0, 1]
    final_data = pd.DataFrame(scaled_data, columns=encoded_data.columns)

    # Drop Data
    final_data.drop(["enrollee_id"], axis=1, inplace=True)  # drop unwanted column 'enrollee_id'

    # Output
    final_data.to_csv(save_path, index=False)
