from utils.packages import *

data = {}

data["target_col"] = 'income_tagging'

data["models"] = {}
data["accuracy"] = {}

data["models"]['LOG'] = linear_model.LogisticRegression()
data["models"]['RF'] = ensemble.RandomForestClassifier()
data["models"]['KNN'] = KNeighborsClassifier()
data["models"]['DT'] = DecisionTreeClassifier()
data["models"]['XG'] = xgboost.XGBClassifier(eval_metric = 'logloss')


data['edu_rank'] = {'kindergarten':1,
            'pre-highschool':2,
            'highschool':3,
            'professional school':4,
            'associate':5,
            'community college':6,
            'bachelors':7,
            'masters':8,
            'doctorate':9}


data["edu_type"] = {'kindergarten':'school',
            'pre-highschool':'school',
            'highschool':'school',
            'professional school':'school',
            'associate':'school',
            'community college':'grad',
            'bachelors':'grad',
            'masters':'pgrad',
            'doctorate':'doc'
           }