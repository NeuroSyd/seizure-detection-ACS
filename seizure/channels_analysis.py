import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
    
class ChannelsImportance():
	def __init__(self,X,y,classifiers,n_keep):
		self.X = X
		self.y = y
		self.classifiers = classifiers
		self.n_keep = n_keep
	def get_channels_importance(self):
		print 'Running Channels Analysis'
		if self.X.shape[1] <= self.n_keep:
			return None			
		print self.classifiers
		importances_agg = [0.0]*self.X.shape[1]

		for (classifier, n_estimators) in self.classifiers:
			if classifier=='GradBoost':
				clf = GradientBoostingClassifier(n_estimators=n_estimators,min_samples_split=2,learning_rate=0.1, random_state=0)
			elif classifier=='AdaBoost':
				clf = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=0.1, random_state=0)
			elif classifier=='ExtraTree':
				clf = ExtraTreesClassifier(n_estimators=n_estimators,learning_rate=0.1, random_state=0)
			elif classifier=='RandomForest':			
				clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=2, bootstrap=False, n_jobs=4, random_state=0)
			clf.fit(self.X.reshape(-1,self.X.shape[1]*self.X.shape[2]),self.y)
			importances = clf.feature_importances_.reshape(self.X.shape[1],self.X.shape[2])
			#print np.sum(importances,axis=1)
			importances_agg += np.sum(importances,axis=1)				
		return importances_agg.argsort()[-self.n_keep:][::-1]
	
