# -*- coding: utf-8 -*-

import copy
import itertools
import numpy as np
import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score

import xgboost as xgb

sb.set_context("talk")


class Xy:
    """
    Class to hold dataframe (train or test) and help 
    execute the pertinent operations of it 
    """
    
    def __init__(self, path):
        df = pd.read_csv(path)
        df = df.rename(columns={'Unnamed: 0': 'ID'})
        df = df.sort_values('ID').reset_index(drop=True)        
        self._df = df
        

    def resume(self):
        df = self.df()

        toprint = ''
       
        toprint += 'DATA FRAME:'
        toprint += '\n\n{}'.format(df)

        toprint += '\n\nMISSING VALUES:'
        toprint += '\n\n{}'.format(pd.DataFrame(df.isnull().sum()).T)

        toprint += '\n\nDESCRIBE:'
        toprint += '\n\n{}'.format(df.describe())     

        print(toprint)


    def df(self):
        return copy.deepcopy(self._df)
    
    
    def update(self, df):
        self._df = df
    
    
    def X_cat(self):
        df = self.df()
        df = df.select_dtypes(include=['object', 'uint8'])
        return df
    
  
    def X_num(self):
        df = self.df()
        df = df.select_dtypes(include=['float64', 'int64'])
        return df  

    
    def X(self):
        df = self.df().iloc[:,:-1]
        return df  

    
    def y(self):
        df = self.df().iloc[:,-1]
        return df  
    
    
    def _replace_nan_numeric(self, mean):
        df = self.df().fillna(mean)
        
        other = copy.deepcopy(self)
        other.update(df)
            
        return other    
    
    
    def _normalize_numeric(self, min_X_num, max_X_num):
        df = self.df()
        
        X_num = self.X_num()
        X_num = (X_num-min_X_num)/(max_X_num-min_X_num)
        for col in X_num:
            df[col] = X_num[col]            
            
        other = copy.deepcopy(self)
        other.update(df)
                
        return other 
    
    
    def _one_hot_encoding(self):
        df = self.df()
        X_cat = self.X_cat()
        dummies = pd.get_dummies(X_cat, dummy_na=True)
        
        for col in dummies.columns:
            name, surname = col.split('_')
            if name in df.columns:
                del df[name]
        
        df = pd.concat([dummies, df], axis=1)
            
        other = copy.deepcopy(self)
        other.update(df)
            
        return other    
    
    
    def _remove_index_columns(self, columns):
        df = self.df()
        for column in columns:
            del df[column]
            
        other = copy.deepcopy(self)
        other.update(df)
            
        return other
        
        
    def _remove_outliers(self, number_of_sigmas, mean, std):    
        factor = number_of_sigmas
        df = self.df()
        X_num = self.X_num()
        for col in X_num:
            series = df[col]
            _mean = mean[col]
            _std = std[col]
            mask = (series <  factor*_std + _mean); series = series[mask]
            mask = (series > -factor*_std + _mean); series = series[mask]

            df = df.loc[series.index]

        other = copy.deepcopy(self)
        other.update(df)
            
        return other
    
    
    def _remove_collinearity(self, columns):
        return self._remove_index_columns(columns)
 

class Ensemble:
    def __init__(self, models, names, weights):
         self.models = models
         self.names = names
         self.weights = weights
         
    
    def name(self):
        return '{}'.format(list(zip(self.names, self.weights)))
      
         
    def predict_proba(self, X):
        y_prob = np.zeros((len(X), 2, ))
        for weight, clf in zip(self.weights, self.models):
            y_prob += weight*clf.predict_proba(X)
        return y_prob / sum(self.weights)
    
    
    def predict(self, X):
        y_prob = self.predict_proba(X)
        return y_prob[:,1] > 0.5
    
    
class PLPreprocessing:
    """
    Class to guarantee that the same preprocessing procedures 
    are applied to the data (pipeline for preprocessing)
    """
    @classmethod
    def procedure_1(cls, train_csv_path, test_csv_path):
        ### TRAINNING DATA ###
        # load data
        train = Xy(train_csv_path)
        
        # removing index features (columns)
        train = train._remove_index_columns(['ID', 'id'])
        
        # removing outliers
        mean = train.X_num().mean()
        std = train.X_num().std()
        train = train._remove_outliers(3, mean, std)
        
        # removing collinear feature
        train = train._remove_collinearity(['tiwrsloh'])
        
        # normalizing numerical data
        min_X_num = train.X_num().min()
        max_X_num = train.X_num().max()
        train = train._normalize_numeric(min_X_num, max_X_num)        
        
        # one hot encoding categorical data
        train = train._one_hot_encoding()
        
        # just for precaution
        df = train.df()
        train.update(df.dropna())
        
        #train.resume()
        
        
        ### TESTING DATA ###
        # load data
        test = Xy(test_csv_path)
        
        # removing index features (columns)
        test = test._remove_index_columns(['ID', 'id'])
        
        # removing outliers
        # attention, we are using standard deviation information
        # from the training dataset. Therefore any information from
        # the testing dataset is considered and, hence, this should not
        # be considered data leaking
        test = test._remove_outliers(3, mean, std)
        
        # removing collinear feature
        test = test._remove_collinearity(['tiwrsloh'])
        
        # normalizing with information of the training data
        # the same situation as the one from outliers remotion
        test = test._normalize_numeric(min_X_num, max_X_num)  
        
        # one hot encoding categorical data
        test = test._one_hot_encoding()
        
        # just for precaution
        df = test.df()        
        test.update(df.dropna())
        
        return train, test     
    
    
    @classmethod
    def procedure_2(cls, train_csv_path, test_csv_path):
        ### TRAINNING DATA ###
        # load data
        train = Xy(train_csv_path)
        
        # removing index features (columns)
        train = train._remove_index_columns(['ID', 'id'])
        
        # replaing the missing values from numerical features by the
        # mean
        mean = train.X_num().mean()
        train = train._replace_nan_numeric(mean)
        
        # removing outliers
        std = train.X_num().std()
        train = train._remove_outliers(3, mean, std)
        
        # removing collinear feature
        train = train._remove_collinearity(['tiwrsloh'])
        
        # normalizing numerical data
        min_X_num = train.X_num().min()
        max_X_num = train.X_num().max()
        train = train._normalize_numeric(min_X_num, max_X_num)        
        
        # one hot encoding categorical data
        train = train._one_hot_encoding()
        
        # just for precaution
        df = train.df()
        train.update(df.dropna())
        
        #train.resume()
        
        
        ### TESTING DATA ###
        # load data
        test = Xy(test_csv_path)
        
        # removing index features (columns)
        test = test._remove_index_columns(['ID', 'id'])
        
        # replace missing numerical values by the mean
        # attention, this is not data leaking since the
        # information used were obtainend from the 
        # train data
        test = test._replace_nan_numeric(mean)
        
        # removing outliers
        # attention, we are using standard deviation information
        # from the training dataset. Therefore any information from
        # the testing dataset is considered and, hence, this should not
        # be considered data leaking
        test = test._remove_outliers(3, mean, std)
        
        # removing collinear feature
        test = test._remove_collinearity(['tiwrsloh'])
        
        # normalizing with information of the training data
        # the same situation as the one from outliers remotion
        test = test._normalize_numeric(min_X_num, max_X_num)  
        
        # one hot encoding categorical data
        test = test._one_hot_encoding()
        
        # just for precaution
        df = test.df()        
        test.update(df.dropna())
        
        return train, test    


class PLHyperparams:
    """
    Class to help the cross-validation prodecure and the 
    hyperparametrization of the models considered
    """
    @classmethod
    def randomforest(cls, train):    
        criterion_lst = ['gini', 'entropy']        
        n_estimators_lst = [500, 1000, 1500]
        class_weight_lst = [{0:1, 1:1}, {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}]
        
        # criterion_lst = ['gini']        
        # n_estimators_lst = [500]
        # class_weight_lst = [{0:1, 1:1}]
        
        hyper_params_dic = {}
        
        for tup in itertools.product(criterion_lst, n_estimators_lst, class_weight_lst):
            criterion, n_estimators, class_weight = tup
            
            clf = RandomForestClassifier( n_estimators=n_estimators
                                        , criterion=criterion
                                        , class_weight=class_weight
                                        , max_depth=None
                                        , min_samples_split=2
                                        , random_state=0
                                        , n_jobs=4
                                        )
           
            scores = cross_val_score(clf, train.X(), train.y(), cv=5)
            toprint = ''
            toprint += '(estimators, criterion, class_weight)\n\t= ({:04d}, {:7s}, {})'\
                .format(n_estimators, criterion, class_weight)
            toprint += ' | scores\n\t\t= {}'.format(scores)
            toprint += ' | mean_score = {:0.4f}'.format(np.mean(scores))

    
            print(toprint)    
    
            hyper_params_dic[np.mean(scores)] = {
                  'scores': scores
                , 'mean_score': np.mean(scores)
                , 'std_score': np.std(scores)                
                
                }
            hyper_params_dic[np.mean(scores)]['params'] = {
                  'criterion': criterion
                , 'n_estimators': n_estimators
                , 'class_weight': class_weight
                } 
            
        return hyper_params_dic
    
    
    @classmethod
    def adaboost(cls, train):
        learning_rate_lst = [0.75, 1.0, 1.25]    
        n_estimators_lst = [500, 1000, 1500]
        
        # learning_rate_lst = [1.0]    
        # n_estimators_lst = [500]

        hyper_params_dic = {}
        
        for tup in itertools.product(learning_rate_lst, n_estimators_lst):
            learning_rate, n_estimators = tup
            
            clf = AdaBoostClassifier(  n_estimators=n_estimators
                                     , learning_rate=learning_rate
                                    )
            
            scores = cross_val_score(clf, train.X(), train.y(), cv=5)
            
            toprint = ''
            toprint += '(estimators, learning_rate)\n\t= ({:04d}, {:.2f})'\
                .format(n_estimators, learning_rate)
            toprint += ' | scores\n\t\t= {}'.format(scores)
            toprint += ' | mean_score = {:0.4f}'.format(np.mean(scores))

    
            print(toprint)    
            
            hyper_params_dic[np.mean(scores)] = {
                  'scores': scores
                , 'mean_score': np.mean(scores)
                , 'std_score': np.std(scores)                
                
                }
            hyper_params_dic[np.mean(scores)]['params'] = {
                  'n_estimators': n_estimators
                , 'learning_rate': learning_rate
                } 
 
        return hyper_params_dic

    
    @classmethod
    def logisticregression(cls, train):
        penalty_lst = ['l1', 'l2']
        class_weight_lst = [{0:1, 1:1}, {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}]

        
        hyper_params_dic = {}
        
        for tup in itertools.product(penalty_lst, class_weight_lst):
            penalty, class_weight = tup
            
            clf = LogisticRegression(  penalty=penalty
                                     , class_weight=class_weight
                                     , solver='liblinear' 
                                    )
            
            scores = cross_val_score(clf, train.X(), train.y(), cv=5)
            
            toprint = ''
            toprint += '(penalty, class_weight)\n\t= ({:5s}, {})'\
                .format(penalty, class_weight)
            toprint += ' | scores\n\t\t= {}'.format(scores)
            toprint += ' | mean_score = {:0.4f}'.format(np.mean(scores))

    
            print(toprint)    
            
            hyper_params_dic[np.mean(scores)] = {
                  'scores': scores
                , 'mean_score': np.mean(scores)
                , 'std_score': np.std(scores)                
                
                }
            hyper_params_dic[np.mean(scores)]['params'] = {
                  'penalty': penalty
                , 'class_weight': class_weight
                } 
 
        return hyper_params_dic            
    
    
    @classmethod
    def xgboost(cls, train):
        eta_lst = [0.01, 0.05, 0.1, 0.15, 0.2]
        max_depth_lst = [3, 4, 5, 6]
        gamma_lst = [0, 0.01, 1, 10]
        
        # eta_lst = [0.2]
        # max_depth_lst = [4]
        # gamma_lst = [1]
        
        hyper_params_dic = {}
        
        for tup in itertools.product(eta_lst, max_depth_lst, gamma_lst):
            eta, max_depth, gamma = tup
        
            clf = xgb.XGBClassifier(eval_metric='error', use_label_encoder=False
                                    , eta=eta
                                    , max_depth=max_depth
                                    , gamma=gamma
                                    )

            scores = cross_val_score(clf, train.X(), train.y(), cv=5)
           
            toprint = ''
            toprint += '(eta, max_depth, gamma)\n\t= ({}, {}, {})'\
            .format(eta, max_depth, gamma)
            toprint += ' | scores\n\t\t= {}'.format(scores)
            toprint += ' | mean_score = {:0.4f}'.format(np.mean(scores))

            print(toprint)  
            
            hyper_params_dic[np.mean(scores)] = {
                  'scores': scores
                , 'mean_score': np.mean(scores)
                , 'std_score': np.std(scores)                
                
                }
            hyper_params_dic[np.mean(scores)]['params'] = {
                  'eta': eta
                , 'max_depth': max_depth
                , 'gamma': gamma
                } 
        
        return hyper_params_dic


    @classmethod
    def ensembleweights(cls, train, models):
        
        names_lst, models_lst = zip(*models.items())        
        
        weights = range(1,6)
        weights_lst =[]
        for _ in range(len(models_lst)):
            weights_lst.append(weights)
            
        y_true_dic = {}
        y_prob_dic = {}        
        weights_dic = {}
        
        for weights in itertools.product(*weights_lst):
            if sum(weights) != 10:
                continue
                
            clf = Ensemble(models_lst, names_lst, weights)                
            name = clf.name()
            
            y_pred = clf.predict(train.X())
            y_prob = clf.predict_proba(test.X()); y_prob_dic[name] = y_prob
            y_true = train.y().to_numpy(); y_true_dic[name] = y_true       
                
            TP = 0; TN = 0
            FP = 0; FN = 0
            for true, pred in zip(y_true, y_pred):
                if pred == 1:
                    if pred == true:
                        TP += 1                        
                    else:
                        FP += 1
                else:
                    if pred == true:
                        TN += 1
                    else:
                        FN += 1            
            
            toprint = ''            
            toprint += '\n{}'.format(name.upper())
            toprint += '\t --> SCORE = {}'.format((TP+TN)/(TP+TN+FP+FN))
            toprint += '\t --> PRECISION = {}'.format(TP/(TP+FP))
            toprint += '\t --> RECALL = {}'.format(TP/(TP+FN))
            toprint += '\n________________________' 
            print(toprint)
            
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            myscore = 0.4*precision + 0.6*recall 
            weights_dic[myscore] = dict(zip(names_lst, weights))
                
        return weights_dic



class PLModel:
    """
    Class to help training the model with the best hyperparameters
    """
    @classmethod
    def randomforest(cls, train, hyper_params):
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        n_estimators = hyper_params['params']['n_estimators']
        criterion = hyper_params['params']['criterion']        
        class_weight = hyper_params['params']['class_weight']
        
        clf = RandomForestClassifier(  n_estimators=n_estimators
                                     , criterion=criterion
                                     , class_weight=class_weight
                                     , max_depth=None
                                     , min_samples_split=2
                                     , random_state=0
                                     , n_jobs=4
                                    )
        
        clf.fit(train.X(), train.y())
        
        return clf
    
    
    @classmethod
    def adaboot(cls, train, hyper_params):
        n_estimators = hyper_params['params']['n_estimators']
        learning_rate = hyper_params['params']['learning_rate']     
        
        clf = AdaBoostClassifier(  n_estimators=n_estimators
                                 , learning_rate=learning_rate
                                )
        
        clf.fit(train.X(), train.y())
        
        return clf
    
    
    @classmethod
    def logisticregression(cls, train, hyper_params):   
        penalty = hyper_params['params']['penalty']
        class_weight = hyper_params['params']['class_weight']
 
        clf = LogisticRegression(  penalty=penalty
                                 , class_weight=class_weight
                                 , solver='liblinear' 
                                )
        
        clf.fit(train.X(), train.y())
        
        return clf
    
    
    @classmethod
    def xgboost(cls, train, hyper_params):        
        eta = hyper_params['params']['eta']
        max_depth = hyper_params['params']['max_depth']
        gamma = hyper_params['params']['gamma']
        
        clf = xgb.XGBClassifier(eval_metric='error', use_label_encoder=False
                        , eta=eta
                        , max_depth=max_depth
                        , gamma=gamma
                        )
        
        clf = xgb.XGBClassifier(eval_metric='error', use_label_encoder=False)

        clf.fit(train.X(), train.y())
  
        return clf
    
class PLTest:
    @classmethod
    def metrics(cls, test, models):
        y_true_dic = {}
        y_prob_dic = {}
        
        for name in models:
            clf = models[name]
        
            y_pred = clf.predict(test.X())
            y_prob = clf.predict_proba(test.X()); y_prob_dic[name] = y_prob
            y_true = test.y().to_numpy(); y_true_dic[name] = y_true
    
            # TP = 0; TN = 0
            # FP = 0; FN = 0
            # for true, pred in zip(y_true, y_pred):
            #     if pred == 1:
            #         if pred == true:
            #             TP += 1                        
            #         else:
            #             FP += 1
            #     else:
            #         if pred == true:
            #             TN += 1
            #         else:
            #             FN += 1
                        
            toprint = ''            
            toprint += '\n\n{}'.format(name.upper())
            toprint += '\n________________________' 
            toprint += '\n\n{}'.format(clf)
            #toprint += '\n\t --> SCORE     = {}'.format((TP+TN)/(TP+TN+FP+FN))
            #toprint += '\n\t --> PRECISION = {}'.format(TP/(TP+FP))
            #toprint += '\n\t --> RECALL    = {}'.format(TP/(TP+FN))
            print(toprint)            
                        
            target_names = ['rich', 'poor']
            print(classification_report(y_true, y_pred, target_names=target_names))
            
            fig, axs = plt.subplots(1,2,figsize=(15, 5))
            
            fig.suptitle(name.upper())
            
            cm = confusion_matrix(y_true, y_pred)            
            sb.heatmap(cm, annot=True, fmt='d', cbar=False, ax=axs[0])
            axs[0].set_title('Confusion matrix')
            axs[0].set_ylabel('True label')
            axs[0].set_xlabel('Predicted label')
            
            fpr, tpr, thresholds = roc_curve(y_true, y_prob[:,1])
            
            auc = roc_auc_score(y_true, y_prob[:,1])
            
            axs[1].plot(fpr, tpr, label='{} (auc = {:.3f})'.format(name, auc))
            axs[1].set_xlabel('False positive rate')
            axs[1].set_ylabel('True positive rate')
            axs[1].set_title('ROC curve')
            axs[1].legend(loc='best')
            
            print('________________________')
            
        fig, axs = plt.subplots(1,1, figsize=(7.5, 7.5))        
        for key in y_true_dic:
            name = key
            y_true = y_true_dic[name]
            y_prob = y_prob_dic[name]
            fpr, tpr, thresholds = roc_curve(y_true, y_prob[:,1])

            auc = roc_auc_score(y_true, y_prob[:,1])

            axs.plot(fpr, tpr, label='{} (auc = {:.3f})'.format(name, auc))
        axs.set_xlabel('False positive rate')
        axs.set_ylabel('True positive rate')
        axs.set_title('ROC curve')
        axs.legend(loc='best')    



                
               
        
if __name__ == '__main__':
    train_1, test_1 = PLPreprocessing.procedure_1('./dataset/Exercise_train (4) (2).csv', './dataset/Exercise_test (4) (2).csv')
    
    #train_2, test_2 = PLPreprocessing.procedure_2('./dataset/Exercise_train (4) (2).csv', './dataset/Exercise_test (4) (2).csv')

    train = train_1; test = test_1
    
    models = {}
    hyper_params = {}

    ### LOGISTICREGRESSION
    _hyper_params = PLHyperparams.logisticregression(train)
    _hyper_params = _hyper_params[max(_hyper_params.keys())]

    hyper_params['LogisticRegression'] = _hyper_params    
    models['LogisticRegression'] = PLModel.logisticregression(train, _hyper_params)

    ### RANDOM FOREST    
    _hyper_params = PLHyperparams.randomforest(train)
    _hyper_params = _hyper_params[max(_hyper_params.keys())]
    
    hyper_params['RandomForest'] = _hyper_params    
    models['RandomForest'] = PLModel.randomforest(train, _hyper_params)
    
    ### ADA BOOST
    _hyper_params = PLHyperparams.adaboost(train)
    _hyper_params = _hyper_params[max(_hyper_params.keys())]
    
    hyper_params['AdaBoost'] = _hyper_params  
    models['AdaBoost'] = PLModel.adaboot(train, _hyper_params)
    
    ### XGBOOST
    _hyper_params = PLHyperparams.xgboost(train)
    _hyper_params = _hyper_params[max(_hyper_params.keys())]

    hyper_params['XGBoost'] = _hyper_params  
    models['XGBoost'] = PLModel.xgboost(train, _hyper_params)
    
    ### ENSEMBLE
    _weights = PLHyperparams.ensembleweights(train, models)
    _names, _models = zip(*models.items())
    weights = []
    for _name in _names: 
        weights.append(_weights[max(_weights)][_name])
    models['Ensemble'] = Ensemble(_models, _names, weights)
    
    ### METRICS
    PLTest.metrics(test, models)