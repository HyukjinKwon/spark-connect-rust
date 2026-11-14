//! The classification classes of the completeness test in `ml_params_table.rs`.

use spark_connect::ml::*;

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    (
        "DecisionTreeClassificationModel",
        DecisionTreeClassificationModel::PARAM_NAMES,
    ),
    (
        "DecisionTreeClassifier",
        DecisionTreeClassifier::PARAM_NAMES,
    ),
    (
        "GBTClassificationModel",
        GBTClassificationModel::PARAM_NAMES,
    ),
    ("GBTClassifier", GBTClassifier::PARAM_NAMES),
    (
        "RandomForestClassificationModel",
        RandomForestClassificationModel::PARAM_NAMES,
    ),
    (
        "RandomForestClassifier",
        RandomForestClassifier::PARAM_NAMES,
    ),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    ("FMClassificationModel", "pending: classification"),
    ("FMClassifier", "pending: classification"),
    ("LinearSVC", "pending: classification"),
    ("LinearSVCModel", "pending: classification"),
    ("LogisticRegression", "pending: classification"),
    ("LogisticRegressionModel", "pending: classification"),
    (
        "MultilayerPerceptronClassificationModel",
        "pending: classification",
    ),
    ("MultilayerPerceptronClassifier", "pending: classification"),
    ("NaiveBayes", "pending: classification"),
    ("NaiveBayesModel", "pending: classification"),
    ("OneVsRest", "pending: classification"),
    ("OneVsRestModel", "pending: classification"),
];
