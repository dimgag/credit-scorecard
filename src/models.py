from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, gini_score
from sklearn.model_selection import train_test_split

class CreditScoreModelTrainer:
    def __init__(self, X, y):
        self.X = X  # Features
        self.y = y  # Target variable

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        # Initialize models
        self.logreg = LogisticRegression()
        self.dt = DecisionTreeClassifier()
        self.rf = RandomForestClassifier()
        self.gb = GradientBoostingClassifier()
        self.svm = SVC()

    def train_models(self):
        # Train models
        self.logreg.fit(self.X_train, self.y_train)
        self.dt.fit(self.X_train, self.y_train)
        self.rf.fit(self.X_train, self.y_train)
        self.gb.fit(self.X_train, self.y_train)
        self.svm.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        # Make predictions
        logreg_pred = self.logreg.predict(self.X_test)
        dt_pred = self.dt.predict(self.X_test)
        rf_pred = self.rf.predict(self.X_test)
        gb_pred = self.gb.predict(self.X_test)
        svm_pred = self.svm.predict(self.X_test)

        # Calculate Gini coefficient or other evaluation metrics
        logreg_gini = gini_score(self.y_test, logreg_pred)
        dt_gini = gini_score(self.y_test, dt_pred)
        rf_gini = gini_score(self.y_test, rf_pred)
        gb_gini = gini_score(self.y_test, gb_pred)
        svm_gini = gini_score(self.y_test, svm_pred)

        # Print evaluation results
        print("Logistic Regression Gini:", logreg_gini)
        print("Decision Tree Gini:", dt_gini)
        print("Random Forest Gini:", rf_gini)
        print("Gradient Boosting Gini:", gb_gini)
        print("Support Vector Machine Gini:", svm_gini)

# Example usage
X = # Your feature matrix
y = # Your target variable

model_trainer = CreditScoreModelTrainer(X, y)
model_trainer.train_models()
model_trainer.evaluate_models()
