// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of the decision tree and random forest
//! classification models. The decision tree is deterministic, so its values are PySpark 4.3.0's
//! on the golden datasets. A random forest's trees depend on the partitioning of the server, so
//! its values are checked against invariants (the predictions, the summaries' counts) and the
//! golden values that do not depend on it. Set SPARK_REMOTE to run against a live Spark Connect
//! server.
mod common;
#[path = "common/tree_classification.rs"]
mod tree_classification;

use common::live_session;
use serde_json::{json, Value};
use spark_connect::column::col;
use spark_connect::ml::{
    BinaryClassificationSummary, ClassificationModel, ClassificationSummary,
    DecisionTreeClassificationModel, DecisionTreeClassifier, DecisionTreeModel, Estimator,
    GBTClassificationModel, GBTClassifier, MlSummary, MlVector, Model, PredictionModel,
    ProbabilisticClassificationModel, RandomForestClassificationModel, RandomForestClassifier,
    RandomForestSummary, RandomForestTrainingSummary, TrainingSummary, Transformer,
    TreeEnsembleModel,
};
use spark_connect::DataFrame;
use tree_classification::{assert_close, dataset, expected, vector_json};

/// The golden of the models and their datasets, with the summaries under `summaries`.
fn golden() -> Value {
    let mut models = tree_classification::golden("classification_models.json");
    let summaries = tree_classification::golden("classification_summaries.json");
    models["summaries"] = summaries["summaries"].clone();
    models
}

/// The rows of a golden dataset's features, in order: the dense vectors of `binary_eval`.
fn features_of(golden: &Value, name: &str) -> Vec<MlVector> {
    golden["datasets"][name]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            MlVector::Dense(
                row[2]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            )
        })
        .collect()
}

/// The features of the golden's predict calls: dense, dense again, and sparse.
fn probes() -> Vec<(&'static str, MlVector)> {
    vec![
        ("dense", MlVector::Dense(vec![0.5, 1.5])),
        ("dense2", MlVector::Dense(vec![3.0, -0.25])),
        (
            "sparse",
            MlVector::Sparse {
                size: 2,
                indices: vec![1],
                values: vec![2.0],
            },
        ),
    ]
}

fn scratch_path(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "spark-connect-rust-tree-api-{}-{name}",
        std::process::id()
    ));
    dir.to_string_lossy().into_owned()
}

fn check(golden: &Value, section: &str, key: &str, member: &str, actual: Value) {
    assert_close(
        &actual,
        expected(golden, section, key, member),
        &format!("{key}.{member}"),
    );
}

/// A model's description with its uid taken out (`uid=X,` or `uid = X,`): the uid is the
/// instance's, not the server's.
fn without_uid(text: &str) -> String {
    let start = text.find("uid").expect("a uid");
    let end = start + text[start..].find(',').expect("the uid's end");
    format!("{}uid*{}", &text[..start], &text[end..])
}

/// The values of `name` of `frame`, in the order of its `id`.
fn column(frame: DataFrame, name: &str) -> Vec<f64> {
    frame
        .order_by(vec![col("id").asc().expression().clone()])
        .select([col(name)])
        .collect()
        .unwrap()
        .into_iter()
        .map(|row| row.get(0).unwrap().as_f64().unwrap())
        .collect()
}

/// The weighted fraction of the rows of `frame` whose label is its prediction, as the summary's
/// accuracy is (the weight column of the golden datasets).
fn accuracy_of(frame: DataFrame) -> f64 {
    let labels = column(frame.clone(), "label");
    let predictions = column(frame.clone(), "prediction");
    let weights = column(frame, "weight");
    let total: f64 = weights.iter().sum();
    let right: f64 = labels
        .iter()
        .zip(&predictions)
        .zip(&weights)
        .filter(|((label, prediction), _)| label == prediction)
        .map(|(_, weight)| weight)
        .sum();
    right / total
}

/// The members of a classification summary (`_ClassificationSummary`), against the golden
/// summary `key`, for the members that do not depend on the fit's partitioning. The DataFrames
/// are compared by their row count.
fn check_classification<S: ClassificationSummary>(s: &S, golden: &Value, key: &str) {
    let section = "summaries";
    let c = |member: &str, actual: Value| check(golden, section, key, member, actual);
    c("labelCol", json!(s.label_col().unwrap()));
    c("predictionCol", json!(s.prediction_col().unwrap()));
    c("weightCol", json!(s.weight_col().unwrap()));
    c("labels", json!(s.labels().unwrap()));
    let accuracy = s.accuracy().unwrap();
    assert!((accuracy - accuracy_of(s.predictions().unwrap())).abs() < 1e-9);
    let classes = s.labels().unwrap().len();
    assert_eq!(s.precision_by_label().unwrap().len(), classes);
    assert_eq!(s.recall_by_label().unwrap().len(), classes);
    assert!(s.weighted_f_measure(1.0).unwrap().is_finite());
    let rows = s.predictions().unwrap().collect().unwrap().len();
    let want = expected(golden, section, key, "predictions")["rows"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(rows, want, "{key}.predictions rows");
}

/// The members of a binary classification summary, against the golden summary `key`.
fn check_binary<S: BinaryClassificationSummary>(s: &S, golden: &Value, key: &str) {
    let section = "summaries";
    check(
        golden,
        section,
        key,
        "scoreCol",
        json!(s.score_col().unwrap()),
    );
    let auc = s.area_under_roc().unwrap();
    assert!((0.0..=1.0).contains(&auc), "{key}.areaUnderROC {auc}");
    let curves = [
        ("roc", s.roc().unwrap()),
        ("pr", s.pr().unwrap()),
        ("fMeasureByThreshold", s.f_measure_by_threshold().unwrap()),
        ("precisionByThreshold", s.precision_by_threshold().unwrap()),
        ("recallByThreshold", s.recall_by_threshold().unwrap()),
    ];
    // The curves depend on the forest's predictions, which vary with the server's partitioning.
    for (member, frame) in curves {
        assert!(!frame.collect().unwrap().is_empty(), "{key}.{member}");
    }
}

/// The members of a training summary (`_TrainingSummary`), against the golden summary `key`.
fn check_training<S: TrainingSummary>(s: &S, golden: &Value, key: &str) {
    let section = "summaries";
    check(
        golden,
        section,
        key,
        "objectiveHistory",
        json!(s.objective_history().unwrap()),
    );
    check(
        golden,
        section,
        key,
        "totalIterations",
        json!(s.total_iterations().unwrap()),
    );
}

#[test]
fn decision_tree_model_matches_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let mut estimator = DecisionTreeClassifier::new()
        .set_max_depth(3)
        .set_seed(42)
        .set_weight_col("weight");
    let model = estimator.fit(&dataset(&s, &g, "multi_train")).unwrap();
    let key = "DecisionTreeClassifier";
    let c = |member: &str, actual: Value| check(&g, "models", key, member, actual);
    c("depth", json!(model.depth().unwrap()));
    c("numNodes", json!(model.num_nodes().unwrap()));
    c("numClasses", json!(model.num_classes().unwrap()));
    c("numFeatures", json!(model.num_features().unwrap()));
    c(
        "featureImportances",
        vector_json(&model.feature_importances().unwrap()),
    );
    assert_close(
        &json!(without_uid(&model.to_debug_string().unwrap())),
        &json!(without_uid(
            expected(&g, "models", key, "toDebugString")
                .as_str()
                .unwrap()
        )),
        "DecisionTreeClassifier.toDebugString",
    );
    c("supportedImpurities", json!(model.supported_impurities()));
    for (name, features) in probes() {
        c(
            &format!("predict-{name}"),
            json!(model.predict(&features).unwrap()),
        );
        c(
            &format!("predictLeaf-{name}"),
            json!(model.predict_leaf(&features).unwrap()),
        );
        c(
            &format!("predictRaw-{name}"),
            vector_json(&model.predict_raw(&features).unwrap()),
        );
        c(
            &format!("predictProbability-{name}"),
            vector_json(&model.predict_probability(&features).unwrap()),
        );
    }
}

#[test]
fn decision_tree_model_and_estimator_round_trip_through_save_and_load() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let estimator = DecisionTreeClassifier::new()
        .set_max_depth(3)
        .set_seed(42)
        .set_weight_col("weight");
    let estimator_path = scratch_path("dt-estimator");
    estimator
        .write()
        .session(&s)
        .overwrite()
        .save(&estimator_path)
        .unwrap();
    let loaded = DecisionTreeClassifier::load(&s, &estimator_path).unwrap();
    let mut before: Vec<String> = estimator
        .params()
        .to_proto()
        .params
        .keys()
        .cloned()
        .collect();
    let mut after: Vec<String> = loaded.params().to_proto().params.keys().cloned().collect();
    before.sort();
    after.sort();
    assert_eq!(before, after, "the estimator's params survive the save");

    let model_path = scratch_path("dt-model");
    let mut fitted = estimator.clone();
    let model = fitted.fit(&dataset(&s, &g, "multi_train")).unwrap();
    model
        .write()
        .unwrap()
        .overwrite()
        .save(&model_path)
        .unwrap();
    let loaded = DecisionTreeClassificationModel::load(&s, &model_path).unwrap();
    let key = "DecisionTreeClassifier";
    check(&g, "models", key, "depth", json!(loaded.depth().unwrap()));
    check(
        &g,
        "models",
        key,
        "predict-dense",
        json!(loaded.predict(&MlVector::Dense(vec![0.5, 1.5])).unwrap()),
    );
}

/// The number of values of a vector.
fn size_of(vector: &MlVector) -> usize {
    match vector {
        MlVector::Dense(values) => values.len(),
        MlVector::Sparse { size, .. } => *size as usize,
    }
}

/// A random forest of three trees with the golden's seed and weight column, and every feature
/// considered at each split, so that the forest does not depend on the server's partitioning.
fn forest() -> RandomForestClassifier {
    RandomForestClassifier::new()
        .set_num_trees(3)
        .set_max_depth(3)
        .set_seed(42)
        .set_feature_subset_strategy("all")
        .set_weight_col("weight")
}

#[test]
fn random_forest_binary_model_matches_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let eval = dataset(&s, &g, "binary_eval");
    let mut model = forest().fit(&dataset(&s, &g, "binary_train")).unwrap();
    let key = "RandomForestClassifier.binary";
    let c = |member: &str, actual: Value| check(&g, "models", key, member, actual);
    assert!(model.has_summary());
    c("numClasses", json!(model.num_classes().unwrap()));
    c("numFeatures", json!(model.num_features().unwrap()));
    c("getNumTrees", json!(model.get_num_trees().unwrap()));
    c("treeWeights", json!(model.tree_weights().unwrap()));
    c(
        "supportedFeatureSubsetStrategies",
        json!(model.supported_feature_subset_strategies()),
    );
    c("supportedImpurities", json!(model.supported_impurities()));

    // The forest is the sum of its trees, and each prediction is the transform's.
    let trees = model.trees().unwrap();
    assert_eq!(trees.len(), 3);
    let nodes: i32 = trees.iter().map(|t| t.num_nodes().unwrap()).sum();
    assert_eq!(model.total_num_nodes().unwrap(), nodes);
    let predicted = column(model.transform(&eval).unwrap(), "prediction");
    for (i, features) in features_of(&g, "binary_eval").iter().enumerate() {
        assert_eq!(model.predict(features).unwrap(), predicted[i], "row {i}");
        let MlVector::Dense(values) = model.predict_probability(features).unwrap() else {
            panic!("a dense probability")
        };
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-9, "row {i}");
        let leaf = model.predict_leaf(features).unwrap();
        assert_eq!(size_of(&leaf), 3, "a leaf of each tree, row {i}");
    }
    for (name, features) in probes() {
        assert!(model.predict(&features).unwrap().is_finite(), "{name}");
    }
    assert_eq!(trees[0].num_classes().unwrap(), 2);
    assert_eq!(trees[0].num_features().unwrap(), 2);

    let summary = match model.summary().unwrap() {
        RandomForestTrainingSummary::Binary(summary) => summary,
        other => panic!("two classes give a binary summary, got {other:?}"),
    };
    let training = "RandomForestClassifier.binary.training";
    check_classification(&summary, &g, training);
    check_binary(&summary, &g, training);
    check_training(&summary, &g, training);
    assert_eq!(
        summary.summary_target().id(),
        format!("{}.summary", summary.summary_target().root().unwrap().id())
    );

    let evaluated = match model.evaluate(&eval).unwrap() {
        RandomForestSummary::Binary(summary) => summary,
        other => panic!("two classes give a binary summary, got {other:?}"),
    };
    let evaluation = "RandomForestClassifier.binary.evaluate";
    check_classification(&evaluated, &g, evaluation);
    check_binary(&evaluated, &g, evaluation);
}

#[test]
fn random_forest_multiclass_model_matches_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let model = forest().fit(&dataset(&s, &g, "multi_train")).unwrap();
    let key = "RandomForestClassifier.multiclass";
    let c = |member: &str, actual: Value| check(&g, "models", key, member, actual);
    c("numClasses", json!(model.num_classes().unwrap()));
    c("getNumTrees", json!(model.get_num_trees().unwrap()));
    c("treeWeights", json!(model.tree_weights().unwrap()));
    for (name, features) in probes() {
        assert!(model.predict(&features).unwrap().is_finite(), "{name}");
        assert_eq!(
            size_of(&model.predict_raw(&features).unwrap()),
            3,
            "a score of each class"
        );
    }

    let summary = match model.summary().unwrap() {
        RandomForestTrainingSummary::Multiclass(summary) => summary,
        other => panic!("three classes give a multiclass summary, got {other:?}"),
    };
    let training = "RandomForestClassifier.multiclass.training";
    check_classification(&summary, &g, training);
    check_training(&summary, &g, training);

    let evaluated = match model.evaluate(&dataset(&s, &g, "multi_eval")).unwrap() {
        RandomForestSummary::Multiclass(summary) => summary,
        other => panic!("three classes give a multiclass summary, got {other:?}"),
    };
    check_classification(&evaluated, &g, "RandomForestClassifier.multiclass.evaluate");
}

#[test]
fn random_forest_model_loads_without_a_training_summary() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let model = forest().fit(&dataset(&s, &g, "binary_train")).unwrap();
    let path = scratch_path("rf-model");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = RandomForestClassificationModel::load(&s, &path).unwrap();
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());
    assert_eq!(
        loaded.predict(&MlVector::Dense(vec![0.5, 1.5])).unwrap(),
        model.predict(&MlVector::Dense(vec![0.5, 1.5])).unwrap(),
    );
}

/// A gradient-boosted classifier of three iterations, with the golden's depth, seed, step size,
/// loss and weight column. Boosting has no sampling, so the model is the golden's on any server.
fn boosted() -> GBTClassifier {
    GBTClassifier::new()
        .set_max_depth(2)
        .set_max_iter(3)
        .set_seed(42)
        .set_step_size(0.1)
        .set_loss_type("logistic")
        .set_feature_subset_strategy("all")
        .set_weight_col("weight")
}

#[test]
fn gradient_boosted_model_matches_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let eval = dataset(&s, &g, "binary_eval");
    let model = boosted().fit(&dataset(&s, &g, "binary_train")).unwrap();
    let key = "GBTClassifier";
    let c = |member: &str, actual: Value| check(&g, "models", key, member, actual);
    c("numClasses", json!(model.num_classes().unwrap()));
    c("numFeatures", json!(model.num_features().unwrap()));
    c("getNumTrees", json!(model.get_num_trees().unwrap()));
    c("totalNumNodes", json!(model.total_num_nodes().unwrap()));
    c("treeWeights", json!(model.tree_weights().unwrap()));
    c(
        "featureImportances",
        vector_json(&model.feature_importances().unwrap()),
    );
    assert_close(
        &json!(without_uid(&model.to_debug_string().unwrap())),
        &json!(without_uid(
            expected(&g, "models", key, "toDebugString")
                .as_str()
                .unwrap()
        )),
        "GBTClassifier.toDebugString",
    );
    c(
        "supportedFeatureSubsetStrategies",
        json!(model.supported_feature_subset_strategies()),
    );
    c("supportedImpurities", json!(model.supported_impurities()));
    c("supportedLossTypes", json!(model.supported_loss_types()));
    c(
        "evaluateEachIteration",
        json!(model.evaluate_each_iteration(&eval).unwrap()),
    );
    for (name, features) in probes() {
        c(
            &format!("predict-{name}"),
            json!(model.predict(&features).unwrap()),
        );
        c(
            &format!("predictLeaf-{name}"),
            vector_json(&model.predict_leaf(&features).unwrap()),
        );
        c(
            &format!("predictRaw-{name}"),
            vector_json(&model.predict_raw(&features).unwrap()),
        );
        c(
            &format!("predictProbability-{name}"),
            vector_json(&model.predict_probability(&features).unwrap()),
        );
    }

    // The first tree is a regression tree of its own.
    let trees = model.trees().unwrap();
    assert_eq!(trees.len(), 3);
    let tree_key = "GBTClassifier.trees[0]";
    let t = |member: &str, actual: Value| check(&g, "models", tree_key, member, actual);
    t("depth", json!(trees[0].depth().unwrap()));
    t("numNodes", json!(trees[0].num_nodes().unwrap()));
    t("numFeatures", json!(trees[0].num_features().unwrap()));
    t(
        "featureImportances",
        vector_json(&trees[0].feature_importances().unwrap()),
    );
    for (name, features) in probes() {
        t(
            &format!("predict-{name}"),
            json!(trees[0].predict(&features).unwrap()),
        );
    }
}

#[test]
fn gradient_boosted_model_round_trips_through_save_and_load() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let model = boosted().fit(&dataset(&s, &g, "binary_train")).unwrap();
    let path = scratch_path("gbt-model");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = GBTClassificationModel::load(&s, &path).unwrap();
    check(
        &g,
        "models",
        "GBTClassifier",
        "predict-dense",
        json!(loaded.predict(&MlVector::Dense(vec![0.5, 1.5])).unwrap()),
    );
    assert_eq!(loaded.trees().unwrap().len(), 3);
}
