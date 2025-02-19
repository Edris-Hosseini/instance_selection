from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score
class i_s:
    def model_selection(self, model,X_train,y_train,k=5):
        if model=='LR':
            train_model=LogisticRegression()
        elif model=='KNN':
            train_model = KNeighborsClassifier(n_neighbors=k)
        elif model=='SVM':
            train_model = SVC()
        elif model=='RF':
            train_model=RandomForestClassifier()
        elif model == 'DT':
            train_model = DecisionTreeClassifier()

        train_model.fit(X_train,y_train)
        return train_model

    def eval(self, model,X_train,y_train, X_test, y_test,k=5):
        # print(y_train)
        train_model = self.model_selection(model,X_train,y_train,k)
        y_pred = train_model.predict(X_test)
        f1 = f1_score(y_test, y_pred,average='weighted')

        return train_model.score(X_test,y_test),f1
