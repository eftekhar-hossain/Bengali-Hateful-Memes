from sklearn.metrics import confusion_matrix,classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


'''Evaluation Parameters'''

def PrintMetrics(true,pred,class_names):
    print("Confusion Matrix:\n", confusion_matrix(true,pred))
    print("Classification Report:\n ", classification_report(true, pred, target_names = class_names, digits = 4))
    print("ROC-AUC: ",round(roc_auc_score(true, pred),4)) 