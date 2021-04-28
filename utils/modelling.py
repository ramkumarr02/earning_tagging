from utils.packages import *

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


def result_report(data):
    
    data["clf"] = data['selected_model'].fit(data["train_x"], data["train_y"])
    data["predictions"] = data["clf"].predict(data["valid_x"])    
    data["acc"] = metrics.accuracy_score(data["predictions"], data["valid_y"])    
    print('--------------------------------')
    print(f'selected model : {data["selected_model"]}')
    print('--------------------------------')
    print(f'Accuracy       : {data["acc"]}')
    print('--------------------------------')
    return(data)

def print_classification_report(data):
    
    target_names = list(data['y_map'].values())
    print(classification_report(data["valid_y"], data['predictions'], target_names=target_names))
    
    return(data)


def print_confusion_matrix(data):
    labels = [data["y_map"][x] for x in data["clf"].classes_]

    fig, ax = plt.subplots(figsize=(8, 8))
    cm = confusion_matrix(data["valid_y"], data['predictions'], labels = data["clf"].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels)
    disp.plot(ax=ax)

    return(data)



def print_roc_curve(data):
    fpr, tpr, threshold = metrics.roc_curve(data["valid_y"], data['predictions'])
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return(data)