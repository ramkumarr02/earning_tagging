from utils.packages import *

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


def train_model(data):    
    data["clf"] = data['classifier'].fit(data["train_x"], data["train_y"])
    data["predictions"] = data["clf"].predict(data["valid_x"])    
    data["acc"] = metrics.accuracy_score(data["predictions"], data["valid_y"])    
    return (data)


def feature_importance(data):
    x = data["df"].loc[:, data["df"].columns != data["target_col"]]
    y = data["df"][data["target_col"]]    
    
    data['x_labels'] = x.apply(LabelEncoder().fit_transform)
    data['y_labels'] = pd.DataFrame(y).apply(LabelEncoder().fit_transform)
    rfe = RFE(estimator= data['rfe_clf'], n_features_to_select = data['rfe_col_num'])
    rfe.fit(data['x_labels'], data['y_labels'])
    rfe.transform(data['x_labels'])

    temp = {}
    data["feature_importance"] = {}
    
    cols = data['x_labels'].columns

    for i, imp_val in enumerate(rfe.ranking_):
        temp[cols[i]] = imp_val


    max_val = np.max(list(temp.values()))

    for i in np.arange(1,max_val+1):        
        data["feature_importance"][i] = []    
        for k, v in temp.items():
            if i == v:
                data["feature_importance"][i].append(k)
            
    return(data)