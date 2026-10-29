"""Reproducible student experiment: train-only tuning, fixed holdout, saved serving pipeline."""
import json, hashlib, platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import precision_recall_curve, recall_score, precision_score, f1_score, fbeta_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from preprocessing import build_preprocessor
ROOT=Path(__file__).resolve().parents[1]

def choose_threshold(y,p,target=.975):
    precision,recall,thresholds=precision_recall_curve(y,p)
    f2=5*precision[:-1]*recall[:-1]/np.maximum(4*precision[:-1]+recall[:-1],1e-12)
    eligible=np.where(recall[:-1]>=target)[0]
    i=eligible[np.argmax(f2[eligible])]
    return float(thresholds[i])

def metrics(y,p,t):
    pred=p>=t
    return dict(threshold=float(t),recall=float(recall_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred)),f2=float(fbeta_score(y,pred,beta=2)),roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),confusion_matrix=confusion_matrix(y,pred).tolist())

def main(output_dir=None):
    out=Path(output_dir) if output_dir else ROOT/'outputs/models';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/raw/healthcare-dataset-stroke-data.csv'
    data=pd.read_csv(source).drop(columns=['id'],errors='ignore').drop_duplicates()
    X,y=data.drop(columns='stroke'),data.stroke
    tx,vx,ty,vy=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    folds=StratifiedKFold(5,shuffle=True,random_state=42)
    configs={
      'Logistic Regression':(LogisticRegression(max_iter=2000,random_state=42),{'classifier__C':[.01,.1,1,10],'classifier__class_weight':[None,'balanced']}),
      'Random Forest':(RandomForestClassifier(n_estimators=180,random_state=42,n_jobs=1),{'classifier__max_depth':[3,5,7,None],'classifier__min_samples_leaf':[5,15,30],'classifier__max_features':[.6,1.0],'classifier__class_weight':[None,'balanced']}),
      'XGBoost':(XGBClassifier(n_estimators=180,eval_metric='logloss',random_state=42,n_jobs=1),{'classifier__max_depth':[2,3,4],'classifier__learning_rate':[.02,.05],'classifier__min_child_weight':[5,15],'classifier__reg_lambda':[5,20],'classifier__subsample':[.8,1.0],'classifier__scale_pos_weight':[1,5,15]}),
      'LightGBM':(LGBMClassifier(n_estimators=140,random_state=42,n_jobs=1,verbosity=-1),{'classifier__max_depth':[3,4,5],'classifier__num_leaves':[7,12],'classifier__learning_rate':[.02,.05],'classifier__min_child_samples':[25,50],'classifier__reg_lambda':[5,20],'classifier__scale_pos_weight':[1,5,15]})}
    results=[];fitted=[]
    for name,(clf,params) in configs.items():
        pipe=Pipeline([('preprocessor',build_preprocessor()),('smote','passthrough'),('classifier',clf)])
        params['smote']=['passthrough',SMOTE(random_state=42,sampling_strategy=.5)]
        search=RandomizedSearchCV(pipe,params,n_iter=12 if name!='Logistic Regression' else 8,cv=folds,scoring='roc_auc',random_state=42,n_jobs=2,error_score='raise')
        search.fit(tx,ty)
        best=search.best_estimator_
        oof=cross_val_predict(best,tx,ty,cv=folds,method='predict_proba',n_jobs=2)[:,1]
        cutoff=choose_threshold(ty,oof)
        best.decision_threshold_=cutoff
        best.model_name_=name
        best.training_metadata_={'seed':42,'target_training_recall':.975,'features':list(tx.columns),'rows':len(data),'cv_folds':5,'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        cv=metrics(ty,oof,cutoff)
        result={'model':name,'cv':cv,'params':{k:str(v) for k,v in search.best_params_.items()}}
        results.append(result);fitted.append((cv['precision'],cv['roc_auc'],name,best))
        print(name,'TRAIN OOF',cv,flush=True)
    # Select on training OOF precision under the predeclared recall constraint.
    selected=max([v for v in fitted if v[2]!='Logistic Regression'],key=lambda v:(v[0],v[1]))
    _,_,name,best=selected
    for result,(_,_,_,candidate) in zip(results,fitted):
        prob=candidate.predict_proba(vx)[:,1]
        result['holdout']=metrics(vy,prob,candidate.decision_threshold_)
    import shap
    processed=best.named_steps['preprocessor'].transform(tx.sample(n=min(200,len(tx)),random_state=42))
    values=np.asarray(shap.TreeExplainer(best.named_steps['classifier']).shap_values(processed))
    if values.ndim==3:values=values[:,:,1]
    names=best.named_steps['preprocessor'].get_feature_names_out()
    best.global_shap_={str(k):float(v) for k,v in zip(names,np.abs(values).mean(axis=0))}
    best.holdout_metrics_=next(r['holdout'] for r in results if r['model']==name)
    report={'selected':name,'selection':'training OOF precision with recall >= 0.975, tree ensembles only','holdout_warning':'Development holdout reused across project iterations; external clinical validation is still required.','results':results,'metadata':best.training_metadata_,'python':platform.python_version()}
    (out/'validation_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for filename in ['model_pipeline.pkl','best_model_week5.pkl']:
        temporary=out/(filename+'.tmp');joblib.dump(best,temporary);temporary.replace(out/filename)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for _,_,label,candidate in fitted:
        prob=candidate.predict_proba(vx)[:,1];fpr,tpr,_=roc_curve(vy,prob);axes[0].plot(fpr,tpr,label=label)
    axes[0].plot([0,1],[0,1],'k--');axes[0].legend(fontsize=8);axes[0].set(title='Development holdout ROC',xlabel='False positive rate',ylabel='Recall')
    cm=confusion_matrix(vy,best.predict_proba(vx)[:,1]>=best.decision_threshold_);axes[1].imshow(cm,cmap='Blues')
    for (i,j),v in np.ndenumerate(cm):axes[1].text(j,i,str(v),ha='center',va='center')
    axes[1].set(title=name+' confusion matrix',xlabel='Predicted',ylabel='Actual',xticks=[0,1],yticks=[0,1]);fig.tight_layout();fig.savefig(out/'evaluation.png',dpi=160);plt.close(fig)
    print('SELECTED',name,json.dumps(next(r for r in results if r['model']==name)),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output-dir');main(p.parse_args().output_dir)
