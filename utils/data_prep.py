from utils.packages import *
from config.variables import *

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


def prep_data_for_map(data):
    ref_df = px.data.gapminder()
    ref_df = ref_df[['country', 'iso_alpha']]
    ref_dict = pd.Series(ref_df.iso_alpha.values,index=ref_df.country).to_dict()

    temp_df0 = data["df"]
    temp_df0 = temp_df0.drop(temp_df0[temp_df0['origin_country'].isin(['laos','other', 'others', 'south korea', 'yugoslavia'])].index)
    temp_df0['origin_country'] = temp_df0['origin_country'].map(temp_dict)

    temp = temp_df0[temp_df0['income_tagging'] == 'less than or equals to $50000']['origin_country'].value_counts()
    temp_df1 = pd.DataFrame()
    temp_df1['cntry'] = temp.keys()
    temp_df1['count'] = temp.values
    temp_df1['type'] = 'below' 

    temp = temp_df0[temp_df0['income_tagging'] == 'more than $50000']['origin_country'].value_counts()
    temp_df = pd.DataFrame()
    temp_df['cntry'] = temp.keys()
    temp_df['count'] = temp.values
    temp_df['type'] = 'above' 

    temp_df = temp_df.append(temp_df1)
    temp_df = temp_df.reset_index(drop=True)

    temp_df['country_code'] = temp_df['cntry'].replace(ref_dict)
    return(temp_df)


def impute_cols(data):

    temp_df = data["df"].apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]),index=series[series.notnull()].index))

    imputer = KNNImputer(weights="uniform")
    temp_imputed = imputer.fit_transform(temp_df)
    temp_imputed = np.round(temp_imputed, 0)

    l = data['impute_cols']

    for col_val in tqdm(l):
        index_dict = {}
        map_dict = {}
        col_position = list(data["df"].columns).index(col_val)

        for cnt in set(data['df'][col_val]):
            if cnt is np.nan:
                pass
            else:
                ind = data['df'][data['df'][col_val].str.find(cnt) == 0].index[0]
                index_dict[cnt] = ind

        for k, v in index_dict.items():
            label_val = temp_imputed[v][col_position]
            map_dict[label_val] = k 

        df_ind_list = data['df'][data['df'][col_val].isnull()].index

        for index_val in df_ind_list:
            impute_val = temp_imputed[index_val][col_position]
            data['df'].at[index_val, col_val] = map_dict[impute_val]

    print('Null values after imputaion')
    print(data["df"].isnull().sum())
    return(data)