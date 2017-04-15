from collections import namedtuple
import os.path
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample
import common.time as time
from common.io import load_hkl_file, save_hkl_file
from sklearn import model_selection, preprocessing
from sklearn.metrics import roc_curve, auc
from channels_analysis import ChannelsImportance        
from seizure.transforms import FreqSlice
from copy import deepcopy

subj = None
significant_channels = None
TaskCore = namedtuple('TaskCore', ['cached_data_loader', 'data_dir', 'target', 'pipeline', 'classifier_name',
                                   'classifier', 'normalize', 'gen_ictal', 'cv_ratio'])

class Task(object):
    """
    A Task computes some work and outputs a dictionary which will be cached on disk.
    If the work has been computed before and is present in the cache, the data will
    simply be loaded from disk and will not be pre-computed.
    """
    def __init__(self, task_core):
        self.task_core = task_core

    def filename(self):
        raise NotImplementedError("Implement this")

    def run(self):
        return self.task_core.cached_data_loader.load(self.filename(), self.load_data)


class LoadIctalDataTask(Task):
    """
    Load the ictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y, 'latencies': latencies}
    """
    def filename(self):
        return 'data_ictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, 'ictal', self.task_core.pipeline,
                           self.task_core.gen_ictal)


class LoadInterictalDataTask(Task):
    """
    Load the interictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """
    def filename(self):
        return 'data_interictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, 'interictal', self.task_core.pipeline)


class LoadTestDataTask(Task):
    """
    Load the test mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X}
    """
    def filename(self):
        return 'data_test_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        global significant_channels
        filename = 'data-cache/significant_channels_%s' % self.task_core.target
        significant_channels = load_hkl_file(filename)
        print 'LoadTestDataTask', significant_channels
        return parse_input_data(self.task_core.data_dir, self.task_core.target, 'test', self.task_core.pipeline)
        
class LoadIctalDataTask_ACS(Task):
    
    def filename(self):
        return 'data_ictal_ACS_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data_ACS(self.task_core.data_dir, self.task_core.target, 'ictal', self.task_core.pipeline,
                           self.task_core.gen_ictal)


class LoadInterictalDataTask_ACS(Task):

    def filename(self):
        return 'data_interictal_ACS_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data_ACS(self.task_core.data_dir, self.task_core.target, 'interictal', self.task_core.pipeline)



# for subject having too many channels, try to elim non-significant data
# save significant channels number into data-cache/subject_significant_channel.hkl
class SignificantChannels():
    def __init__(self,task_core):
        self.task_core = task_core    
    def filename(self):
        return 'significant_channels_%s' % self.task_core.target
    def load_data(self):   
        ictal_data = LoadIctalDataTask_ACS(self.task_core).run()
        interictal_data = LoadInterictalDataTask_ACS(self.task_core).run()
        data = prepare_training_data_ACS(ictal_data, interictal_data, self.task_core.cv_ratio)    
        nchannels = pd.read_csv('nchannels.csv',index_col=0)        
        n = nchannels['n'][subj]

        #print subj, 'is using ', n, 'channels..................'
        return ChannelsImportance(data['X_train'],data['y_train'],[('RandomForest',1000)],int(n)).get_channels_importance()

class TrainingDataTask(Task):
    """
    Creating a training set and cross-validation set from the transformed ictal and interictal data.
    """
    def filename(self):
        return None  # not cached, should be fast enough to not need caching

    def load_data(self):
        ictal_data = LoadIctalDataTask(self.task_core).run()
        interictal_data = LoadInterictalDataTask(self.task_core).run()
        return prepare_training_data(ictal_data, interictal_data, self.task_core.cv_ratio)


class CrossValidationScoreTask(Task):
    """
    Run a classifier over a training set, and give a cross-validation score.
    """
    
    def filename(self):
        return 'score_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        global significant_channels
        global subj
        subj = self.task_core.target # to be used in auc record
        start = time.get_seconds()
        significant_channels = SignificantChannels(self.task_core).load_data()
        print significant_channels
        print 'ACS time is %d s' % (time.get_seconds() - start)
        #print aa
        data = TrainingDataTask(self.task_core).run()
        classifier_data = train_classifier(self.task_core.classifier, data, normalize=self.task_core.normalize)
        del classifier_data['classifier'] # save disk space
        return classifier_data
        

class TrainClassifierTask(Task):
    """
    Run a classifier over the complete data set (training data + cross-validation data combined)
    and save the trained models.
    """
    def filename(self):
        return 'classifier_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):    
        
        data = TrainingDataTask(self.task_core).run()
        return train_classifier(self.task_core.classifier, data, use_all_data=True, normalize=self.task_core.normalize)


class MakePredictionsTask(Task):
    """
    Make predictions on the test data.
    """
    def filename(self):
        return 'predictions_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        global significant_channels
        global subj
        subj = self.task_core.target # to be used in auc record
        start = time.get_seconds()
        filename = 'data-cache/significant_channels_%s' % self.task_core.target
        significant_channels = SignificantChannels(self.task_core).load_data()
        save_hkl_file(filename,significant_channels)
        print significant_channels
        print 'ACS time is %d s' % (time.get_seconds() - start)        
        
        start = time.get_seconds()
        data = TrainingDataTask(self.task_core).run()
        y_classes = data.y_classes
        del data
        
        point = time.get_seconds()
        time_prepare = (point - start)
        print 'Time to prepare data for %s is %f seconds.' % (self.task_core.target,time_prepare)   

        classifier_data = TrainClassifierTask(self.task_core).run()
        
        point2 = time.get_seconds()
        time_training = (point2 - point)
        print 'Time to train data for %s is %f seconds.' % (self.task_core.target,time_training)
        
        test_data = LoadTestDataTask(self.task_core).run()
        X_test = flatten(test_data.X)

        return make_predictions(self.task_core.target, X_test, y_classes, classifier_data)

# a list of pairs indicating the slices of the data containing full seizures
# e.g. [(0, 5), (6, 10)] indicates two ranges of seizures
def seizure_ranges_for_latencies(latencies):
    indices = np.where(latencies == 0)[0]

    ranges = []
    for i in range(1, len(indices)):
        ranges.append((indices[i-1], indices[i]))
    ranges.append((indices[-1], len(latencies)))

    return ranges


#generator to iterate over competition mat data
def load_mat_data(data_dir, target, component):
    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        filename = '%s/%s_%s_segment_%d.mat' % (dir, target, component, i)
        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)
            yield(data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True

# process all of one type of the competition mat data
# data_type is one of ('ictal', 'interictal', 'test')
def parse_input_data_ACS(data_dir, target, data_type, pipeline, gen_ictal=False):
    ictal = data_type == 'ictal'

    mat_data = load_mat_data(data_dir, target, data_type)

    def process_raw_data(mat_data, with_latency):

        X = []
        y = []
        
        for segment in mat_data:
            data = segment['data']
            if with_latency:
                y.append(0)
            elif y is not None:
                y.append(1)
            transformed_data = FreqSlice(1,48).apply(data)
            X.append(transformed_data)

        X = np.array(X)        
        y = np.array(y)

        print 'X', X.shape, 'y', y.shape
        return X, y

    X, y = process_raw_data(mat_data, with_latency=ictal)
    return {
        'X': X,
        'y': y
    }


# process all of one type of the competition mat data
# data_type is one of ('ictal', 'interictal', 'test')
def parse_input_data(data_dir, target, data_type, pipeline, gen_ictal=False):
    ictal = data_type == 'ictal'
    interictal = data_type == 'interictal'

    mat_data = load_mat_data(data_dir, target, data_type)
      
    # for each data point in ictal, interictal and test,
    # generate (X, <y>, <latency>) per channel
    def process_raw_data(mat_data, with_latency):
        start = time.get_seconds()
        print 'Loading data',
        X = []
        y = []
        latencies = []

        for segment in mat_data:
            data = segment['data']
            if (significant_channels is not None):
                data = data[significant_channels]
            if data.shape[-1] > 400:
                data = resample(data, 400, axis=data.ndim - 1)

            if with_latency:
                # this is ictal
                latency = segment['latency'][0]
                if latency <= 15:
                    y_value = 0 # ictal <= 15
                else:
                    y_value = 1 # ictal > 15

                y.append(y_value)
                latencies.append(latency)

                prev_latency = latency
            elif y is not None:
                y.append(2)

            
            transformed_data = pipeline.apply(data)
            X.append(transformed_data)

        print '(%ds)' % (time.get_seconds() - start)

        X = np.array(X)
        y = np.array(y)
        latencies = np.array(latencies)

        if ictal:
            print 'X', X.shape, 'y', y.shape, 'latencies', latencies.shape
            return X, y, latencies
        elif interictal:
            print 'X', X.shape, 'y', y.shape
            return X, y
        else:
            print 'X', X.shape
            return X

    data = process_raw_data(mat_data, with_latency=ictal)

    if len(data) == 3:
        X, y, latencies = data
        return {
            'X': X,
            'y': y,
            'latencies': latencies
        }
    elif len(data) == 2:
        X, y = data
        return {
            'X': X,
            'y': y
        }
    else:
        X = data
        return {
            'X': X
        }


# flatten data down to 2 dimensions for putting through a classifier
def flatten(data):
    if data.ndim > 2:
        return data.reshape((data.shape[0], np.product(data.shape[1:])))
    else:
        return data

# split up ictal and interictal data into training set and cross-validation set
def prepare_training_data_ACS(ictal_data, interictal_data, cv_ratio):
    print 'Preparing training data ...',
    ictal_X, ictal_y = (ictal_data.X), ictal_data.y
    interictal_X, interictal_y = (interictal_data.X), interictal_data.y

    def concat(a, b):
        return np.concatenate((a, b), axis=0)

    X_train = concat(ictal_X, interictal_X)
    y_train = concat(ictal_y, interictal_y)
    y_classes = np.unique(y_train)

    print 'X_train:', np.shape(X_train)
    print 'y_train:', np.shape(y_train)
    
    print 'y_classes:', y_classes

    return {
        'X_train': X_train,
        'y_train': y_train,
        'y_classes': y_classes
    }

# split up ictal and interictal data into training set and cross-validation set
def prepare_training_data(ictal_data, interictal_data, cv_ratio):
    print 'Preparing training data ...',
    ictal_X, ictal_y = flatten(ictal_data.X), ictal_data.y
    interictal_X, interictal_y = flatten(interictal_data.X), interictal_data.y

    seizure_ranges = seizure_ranges_for_latencies(ictal_data.latencies)
    num_seizures = len(seizure_ranges)

    def concat(a, b):
        return np.concatenate((a, b), axis=0)

    X_train_l = []
    y_train_l = []
    X_cv_l = []
    y_cv_l = []
    ictal_X_cv_l = []
    ictal_y_cv_l = []
    ictal_l_cv_l = []
    interictal_X_cv_l = []
    interictal_y_cv_l = []

    for fold in range(num_seizures):
        tmp_ranges = deepcopy(seizure_ranges)
        cv_ictal_range = tmp_ranges[fold]
        print ('cv_ictal_range', cv_ictal_range)
        del tmp_ranges[fold]
        print ('train_ictal_range', tmp_ranges)

        ictal_X_cv, ictal_y_cv, ictal_l_cv = ictal_X[cv_ictal_range[0]:cv_ictal_range[1]], ictal_y[cv_ictal_range[0]:
        cv_ictal_range[1]], ictal_data.latencies[cv_ictal_range[0]:cv_ictal_range[1]]

        ictal_X_train_chunks = []
        ictal_y_train_chunks = []
        ictal_l_train_chunks = []
        for r in tmp_ranges:
            _ictal_X_train, _ictal_y_train, _ictal_l_train = ictal_X[r[0]:r[1]], ictal_y[r[0]:r[1]], ictal_data.latencies[r[0]:r[1]]
            ictal_X_train_chunks.append(_ictal_X_train)
            ictal_y_train_chunks.append(_ictal_y_train)
            ictal_l_train_chunks.append(_ictal_l_train)
        ictal_X_train = np.concatenate(ictal_X_train_chunks)
        ictal_y_train = np.concatenate(ictal_y_train_chunks)

        interictal_X_train, interictal_y_train, interictal_X_cv, interictal_y_cv = split_train_random(interictal_X,
                                                                                                      interictal_y,
                                                                                                      1.0 / num_seizures)

        X_train = concat(ictal_X_train, interictal_X_train)
        y_train = concat(ictal_y_train, interictal_y_train)
        X_cv = concat(ictal_X_cv, interictal_X_cv)
        y_cv = concat(ictal_y_cv, interictal_y_cv)

        y_classes = np.unique(concat(y_train, y_cv))

        X_train_l.append(X_train)
        y_train_l.append(y_train)
        X_cv_l.append(X_cv)
        y_cv_l.append(y_cv)
        ictal_X_cv_l.append(ictal_X_cv)
        ictal_y_cv_l.append(ictal_y_cv)
        ictal_l_cv_l.append(ictal_l_cv)
        interictal_X_cv_l.append(interictal_X_cv)
        interictal_y_cv_l.append(interictal_y_cv)

    return {
        'X_train': X_train_l,
        'y_train': y_train_l,
        'X_cv': X_cv_l,
        'y_cv': y_cv_l,
        'y_classes': y_classes
    }


# split interictal segments at random for training and cross-validation
def split_train_random(X, y, cv_ratio):
    X_train, X_cv, y_train, y_cv = model_selection.train_test_split(X, y, test_size=cv_ratio, random_state=0)
    return X_train, y_train, X_cv, y_cv


# train classifier for cross-validation
def train(classifier, X_train, y_train, X_cv, y_cv, y_classes):
    print "Training ..."

    print 'Dim', 'X', np.shape(X_train), 'y', np.shape(y_train), 'X_cv', np.shape(X_cv), 'y_cv', np.shape(y_cv)
    start = time.get_seconds()
    classifier.fit(X_train, y_train)
    print "Scoring..."
    S, E = score_classifier_auc(classifier, X_cv, y_cv, y_classes)
    score = 0.5 * (S + E)

    elapsedSecs = time.get_seconds() - start
    print "t=%ds score=%f" % (int(elapsedSecs), score)
    return score, S, E


# train classifier for predictions
def train_all_data(classifier, X_train, y_train, X_cv, y_cv):
    print "Training ..."
    X = np.concatenate((X_train, X_cv), axis=0)
    y = np.concatenate((y_train, y_cv), axis=0)
    print 'Dim', np.shape(X), np.shape(y)
    start = time.get_seconds()
    classifier.fit(X, y)
    elapsedSecs = time.get_seconds() - start
    print "t=%ds" % int(elapsedSecs)


# sub mean divide by standard deviation
def normalize_data(X_train, X_cv):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)
    return X_train, X_cv

# depending on input train either for predictions or for cross-validation
def train_classifier(classifier, data, use_all_data=False, normalize=False):
    X_train = data.X_train
    y_train = data.y_train
    X_cv = data.X_cv
    y_cv = data.y_cv

    n_folds = len(X_train)

    if normalize:
        X_train, X_cv = normalize_data(X_train, X_cv)

    if not use_all_data:
        scores = []
        Ss=[]
        Es=[]
        for i in range(n_folds):
            score, S, E = train(classifier, X_train[i], y_train[i], X_cv[i], y_cv[i], data.y_classes)
            scores.append(score)
            Ss.append(S)
            Es.append(E)
        return {
            'classifier': classifier,
            'score': scores,
            'S_auc': Ss,
            'E_auc': Es
        }
    else:
        X_train = np.concatenate(X_train,axis=0)
        y_train = np.concatenate(y_train, axis=0)
        X_cv = np.concatenate(X_cv, axis=0)
        y_cv = np.concatenate(y_cv, axis=0)
        train_all_data(classifier, X_train, y_train, X_cv, y_cv)
        return {
            'classifier': classifier
        }


# convert the output of classifier predictions into (Seizure, Early) pair
def translate_prediction(prediction, y_classes):
    if len(prediction) == 3:
        ictalLTE15, ictalGT15, interictal = prediction
        S = ictalLTE15 + ictalGT15
        E = ictalLTE15
        return S, E
    elif len(prediction) == 2:
        # 1.0 doesn't exist for Patient_4, i.e. there is no late seizure data
        if not np.any(y_classes == 1.0):
            ictalLTE15, interictal = prediction
            S = ictalLTE15
            E = ictalLTE15
            return S, E
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


# use the classifier and make predictions on the test data
def make_predictions(target, X_test, y_classes, classifier_data):
    classifier = classifier_data.classifier
    predictions_proba = classifier.predict_proba(X_test)

    lines = []
    for i in range(len(predictions_proba)):
        p = predictions_proba[i]
        S, E = translate_prediction(p, y_classes)
        lines.append('%s_test_segment_%d.mat,%.15f,%.15f' % (target, i+1, S, E))

    return {
        'data': '\n'.join(lines),
	'predictions': predictions_proba,
	'y_classes': y_classes
    }


# the scoring mechanism used by the competition leaderboard
def score_classifier_auc(classifier, X_cv, y_cv, y_classes):
    predictions = classifier.predict_proba(X_cv)
    S_predictions = []
    E_predictions = []
    S_y_cv = [1.0 if (x == 0.0 or x == 1.0) else 0.0 for x in y_cv]
    E_y_cv = [1.0 if x == 0.0 else 0.0 for x in y_cv]

    for i in range(len(predictions)):
        p = predictions[i]
        S, E = translate_prediction(p, y_classes)
        S_predictions.append(S)
        E_predictions.append(E)

    fpr, tpr, thresholds = roc_curve(S_y_cv, S_predictions)
    S_roc_auc = auc(fpr, tpr)
    fpr, tpr, thresholds = roc_curve(E_y_cv, E_predictions)
    E_roc_auc = auc(fpr, tpr)

    return S_roc_auc, E_roc_auc

