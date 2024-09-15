import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, mean_squared_error
from joblib import Parallel, delayed
import numpy as np

def plot_roc_curve(y_test, y_prob, label_encoder, model_name):
    num_classes = len(label_encoder.classes_)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(9, 7.2), dpi=100)
    for i, color in zip(range(num_classes), plt.cm.rainbow(np.linspace(0, 1, num_classes))):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} (area = {1:0.2f})'
                       ''.format(label_encoder.inverse_transform([i])[0], roc_auc[i]))
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    return fig

def evaluate_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test, label_encoder):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    # Ensure the predicted probabilities have the correct number of columns
    num_classes = len(label_encoder.classes_)
    if y_prob.shape[1] != num_classes:
        zero_prob = np.zeros((y_prob.shape[0], num_classes))
        zero_prob[:, :y_prob.shape[1]] = y_prob
        y_prob = zero_prob

    # Normalize probabilities to sum up to 1.0
    y_prob /= y_prob.sum(axis=1, keepdims=True)

    # Calculate metrics for the overall model
    overall_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    fscore = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate metrics only for the labels present in y_test
    present_labels = np.unique(y_test).astype(int)
    auc_scores = {label_encoder.inverse_transform([label])[0]: roc_auc_score(y_test == label, y_prob[:, label]) for label in present_labels}
    fscore_per_class = {label_encoder.inverse_transform([label])[0]: f1_score(y_test == label, y_pred == label) for label in present_labels}
    accuracy_per_class = {label_encoder.inverse_transform([label])[0]: accuracy_score(y_test == label, y_pred == label) for label in present_labels}
    
    conf_matrix = confusion_matrix(y_test, y_pred, labels=present_labels)
    
    # Plot ROC curve
    fig = plot_roc_curve(y_test, y_prob, label_encoder, name)
    
    return name, (y_test, y_pred, overall_auc, auc_scores, fscore_per_class, accuracy_per_class, fscore, accuracy, mse, conf_matrix, fig)

def train_evaluate_voting(X_train_scaled, y_train, X_test_scaled, y_test, label_encoder):
    estimators = [
        ('svc', SVC(kernel='linear', probability=True, random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('knn', KNeighborsClassifier())
    ]

    # Train individual models
    results = dict(Parallel(n_jobs=-1)(delayed(evaluate_model)(name, model, X_train_scaled, y_train, X_test_scaled, y_test, label_encoder) for name, model in estimators))

    # Train the voting classifier
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    voting_clf.fit(X_train_scaled, y_train)
    y_pred_voting = voting_clf.predict(X_test_scaled)
    y_prob_voting = voting_clf.predict_proba(X_test_scaled)
    
    # Ensure the predicted probabilities have the correct number of columns
    num_classes = len(label_encoder.classes_)
    if y_prob_voting.shape[1] != num_classes:
        zero_prob = np.zeros((y_prob_voting.shape[0], num_classes))
        zero_prob[:, :y_prob_voting.shape[1]] = y_prob_voting
        y_prob_voting = zero_prob
    
    # Normalize probabilities to sum up to 1.0
    y_prob_voting /= y_prob_voting.sum(axis=1, keepdims=True)
    
    # Calculate metrics for the overall model
    overall_auc_voting = roc_auc_score(y_test, y_prob_voting, multi_class='ovr')
    fscore_voting = f1_score(y_test, y_pred_voting, average='weighted')
    accuracy_voting = accuracy_score(y_test, y_pred_voting)
    mse_voting = mean_squared_error(y_test, y_pred_voting)
    
    # Calculate metrics only for the labels present in y_test
    present_labels = np.unique(y_test).astype(int)
    auc_scores_voting = {label_encoder.inverse_transform([label])[0]: roc_auc_score(y_test == label, y_prob_voting[:, label]) for label in present_labels}
    fscore_per_class_voting = {label_encoder.inverse_transform([label])[0]: f1_score(y_test == label, y_pred_voting == label) for label in present_labels}
    accuracy_per_class_voting = {label_encoder.inverse_transform([label])[0]: accuracy_score(y_test == label, y_pred_voting == label) for label in present_labels}
    
    conf_matrix_voting = confusion_matrix(y_test, y_pred_voting, labels=present_labels)
    
    # Plot ROC curve for Voting Classifier
    fig = plot_roc_curve(y_test, y_prob_voting, label_encoder, 'Voting Classifier')
    
    results['Voting Classifier'] = (y_test, y_pred_voting, overall_auc_voting, auc_scores_voting, fscore_per_class_voting, accuracy_per_class_voting, fscore_voting, accuracy_voting, mse_voting, conf_matrix_voting, fig)
    
    return results, fig
