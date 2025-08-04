from sklearn.ensembly import RandomForestClassifier
from sklearn.dataset import make_classification 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_scorr, classification_report
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
