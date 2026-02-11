import xgboost as xgb


def get_xgb_model(scale_weight):
    return xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_weight,
        random_state=42,
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=500,
    )
