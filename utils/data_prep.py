from utils.packages import *


def analyze_observations(data):
    for col_name in data["df"].columns:
        col_type = data["df"][col_name].dtype
        if col_type != 'int64':
            print('---------------------')
            print(f'{col_name}')
            print(data["df"][col_name].value_counts())
            print('---------------------')
    return(data)



def analyze_features(data):
    for col_name in data["df"].columns:
        col_type = data["df"][col_name].dtype
        if col_type != 'int64':
            print('---------------------')
            print(f'{col_name} : unique vals :  {len(set(data["df"][col_name]))}')
            print(set(data["df"][col_name]))
            print('---------------------')
    return(data)


def split_x_y(data):
    
    data["x"] = data["df"].loc[:, data["df"].columns != data["target_col"]]
    data["y"] = data["df"][data["target_col"]]
    
    data["categorical_columns"] = [c for c in data["x"].columns if data["x"][c].dtype =="object"]
    data["numerical_columns"] = [c for c in data["df"].columns if data["df"][c].dtype =="int64"]
    
    print("----------------------------")
    print(f'categorical_columns : {data["categorical_columns"]}')
    print(f'numerical_columns : {data["numerical_columns"]}')
    print("----------------------------")
    
    return(data)


def encode_data(data):
    data["encoder"] = LabelEncoder()
    data["encoder"].fit(data["y"])
    data["y_encoded"] = data["encoder"].transform(data["y"])
    data["y_map"] = dict(zip(data["encoder"].transform(data["encoder"].classes_),data["encoder"].classes_))
    
    data['x_dummied'] = pd.get_dummies(data["x"])
    
    print("----------------------------")
    print(f'x_dummied df ready')
    print(f'y_encoded series ready')
    print("----------------------------")
    return(data)


def prep_data_for_tf(data):
    data['y_encoded_t'] = utils.to_categorical(data['y_encoded'])

    scaler_obj = StandardScaler()
    scaler_obj.fit(data["x_dummied"].values)
    data['x_scaled'] = scaler_obj.transform(data["x_dummied"].values)

    data["train_x_t"], data["valid_x_t"], data["train_y_t"], data["valid_y_t"] = train_test_split(data["x_scaled"], data['y_encoded_t'],train_size = 0.8,random_state = 1)
    
    return(data)