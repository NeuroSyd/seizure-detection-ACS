import json
import os.path
import numpy as np
from common import time
from common.data import CachedDataLoader, makedirs
from common.pipeline import Pipeline
from seizure.transforms import FFT, Slice, Magnitude, Log10, FFTWithTimeFreqCorrelation, MFCC, Resample, Stats, \
    DaubWaveletStats, TimeCorrelation, FreqCorrelation, TimeFreqCorrelation
from seizure.tasks import TaskCore, CrossValidationScoreTask, MakePredictionsTask, TrainClassifierTask
from seizure.scores import get_score_summary, print_results

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

def run_seizure_detection(build_target):
    """
    The main entry point for running seizure-detection cross-validation and predictions.
    Directories from settings file are configured, classifiers are chosen, pipelines are
    chosen, and the chosen build_target ('cv', 'predict', 'train_model') is run across
    all combinations of (targets, pipelines, classifiers)
    """

    with open('SETTINGS.json') as f:
        settings = json.load(f)

    data_dir = str(settings['competition-data-dir'])
    cache_dir = str(settings['data-cache-dir'])
    submission_dir = str(settings['submission-dir'])

    makedirs(submission_dir)

    cached_data_loader = CachedDataLoader(cache_dir)

    ts = time.get_millis()

    targets = [
        'Dog_1',
        'Dog_2',
        'Dog_3',
        'Dog_4',
        'Patient_1',
        'Patient_2',
        'Patient_3',
        'Patient_4',
        'Patient_5',
        'Patient_6',
        'Patient_7',
        'Patient_8'
    ]

    pipelines = [
        Pipeline(gen_ictal=False, pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')])
    ]
    classifiers = [
        (RandomForestClassifier(n_estimators=3000, min_samples_split=2, bootstrap=False, n_jobs=4, random_state=0), 'rf3000')
    ]
    cv_ratio = 0.5

    def should_normalize(classifier):
        clazzes = [LogisticRegression]
        return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)

    def train_full_model(make_predictions):
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                guesses = ['clip,seizure,early']
                classifier_filenames = []
                for target in targets:
                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_ictal=pipeline.gen_ictal,
                                         cv_ratio=cv_ratio)

                    if make_predictions:
                        predictions = MakePredictionsTask(task_core).run()
                        guesses.append(predictions.data)
                    else:
                        task = TrainClassifierTask(task_core)
                        task.run()
                        classifier_filenames.append(task.filename())

                if make_predictions:
                    filename = 'submission%d-%s_%s.csv' % (ts, classifier_name, pipeline.get_name())
                    filename = os.path.join(submission_dir, filename)
                    with open(filename, 'w') as f:
                        print >> f, '\n'.join(guesses)
                    print 'wrote', filename
                else:
                    print 'Trained classifiers ready in %s' % cache_dir
                    for filename in classifier_filenames:
                        print os.path.join(cache_dir, filename + '.pickle')

    def do_cross_validation():
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                scores = []
                for target in targets:
                    print 'Processing %s (classifier %s)' % (target, classifier_name)

                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_ictal=pipeline.gen_ictal,
                                         cv_ratio=cv_ratio)

                    data = CrossValidationScoreTask(task_core).run()
                    score = data.score
                    scores.append(score)

                    print target, 'Seizure_AUC=', data.S_auc, 'Early_AUC=', data.E_auc

    if build_target == 'cv':
        do_cross_validation()
    elif build_target == 'train_model':
        train_full_model(make_predictions=False)
    elif build_target == 'make_predictions':
        train_full_model(make_predictions=True)
    else:
        raise Exception("unknown build target %s" % build_target)
