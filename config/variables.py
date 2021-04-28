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


data["epoch_val"]         = 10
data["batch_size_val"]    = 64
data["verbose_val"]       = 2
data["workers_val"]       = -1


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


data['cnt_cont'] = {
                    'cambodia':'asia',
                    'canada':'north_america',
                    'china':'asia',
                    'columbia':'south_america',
                    'cuba':'caribbean',
                    'dominican republic':'caribbean',
                    'ecuador':'south_america',
                    'el salvador':'central_america',
                    'england':'europe',
                    'france':'europe',
                    'germany':'europe',
                    'greece':'europe',
                    'guatemala':'central_america',
                    'haiti':'caribbean',
                    'honduras':'central_america',
                    'hong kong':'asia',
                    'hungary':'europe',
                    'india':'asia',
                    'iran':'middle_east',
                    'ireland':'europe',
                    'italy':'europe',
                    'jamaica':'caribbean',
                    'japan':'asia',
                    'laos':'asia',
                    'mexico':'north_america',
                    'nicaragua':'central_america',
                    'other':'other',
                    'others':'other',
                    'peru':'south_america',
                    'philippines':'asia',
                    'poland':'europe',
                    'portugal':'europe',
                    'puerto rico':'central_america',
                    'scotland':'europe',
                    'south korea':'asia',
                    'taiwan':'asia',
                    'thailand':'asia',
                    'trinadad and tobago':'caribbean',
                    'usa':'north_america',
                    'vietnam':'asia',
                    'yugoslavia':'europe',
                    }


temp_dict = {'cambodia':'Cambodia',
'canada':'Canada',
'china':'China',
'columbia':'Columbia',
'cuba':'Cuba',
'dominican republic':'Dominican republic',
'ecuador':'Ecuador',
'el salvador':'El salvador',
'england':'United Kingdom',
'france':'France',
'germany':'Germany',
'greece':'Greece',
'guatemala':'Guatemala',
'haiti':'Haiti',
'honduras':'Honduras',
'hong kong':'Hong Kong China',
'hungary':'Hungary',
'india':'India',
'iran':'Iran',
'ireland':'Ireland',
'italy':'Italy',
'jamaica':'Jamaica',
'japan':'Japan',
'laos':'Laos',
'mexico':'Mexico',
'nicaragua':'Nicaragua',
'other':'Other',
'others':'Others',
'peru':'Peru',
'philippines':'Philippines',
'poland':'Poland',
'portugal':'Portugal',
'puerto rico':'Puerto rico',
'scotland':'United Kingdom',
'south korea':'South korea',
'taiwan':'Taiwan',
'thailand':'Thailand',
'trinadad and tobago':'Trinadad and Tobago',
'usa':'United States',
'vietnam':'Vietnam',
'yugoslavia':'Yugoslavia',
}

