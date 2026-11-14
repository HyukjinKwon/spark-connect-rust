// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of the native regression estimators, as PySpark's
//! `pyspark.ml.regression` has them over Spark Connect (see [`super::object`]).
use super::*;

native_model! {
    PyDecisionTreeRegressionModel = "DecisionTreeRegressionModel"
        in "pyspark.ml.connect.regression",
        jvm: "org.apache.spark.ml.regression.DecisionTreeRegressionModel";
    params: [
    "cacheNodeIds" => ("getCacheNodeIds", get_cache_node_ids, param_cache_node_ids),
    "checkpointInterval" => (
        "getCheckpointInterval",
        get_checkpoint_interval,
        param_checkpoint_interval
    ),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "impurity" => ("getImpurity", get_impurity, param_impurity),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "leafCol" => ("getLeafCol", get_leaf_col, param_leaf_col),
    "maxBins" => ("getMaxBins", get_max_bins, param_max_bins),
    "maxDepth" => ("getMaxDepth", get_max_depth, param_max_depth),
    "maxMemoryInMB" => ("getMaxMemoryInMB", get_max_memory_in_mb, param_max_memory_in_mb),
    "minInfoGain" => ("getMinInfoGain", get_min_info_gain, param_min_info_gain),
    "minInstancesPerNode" => (
        "getMinInstancesPerNode",
        get_min_instances_per_node,
        param_min_instances_per_node
    ),
    "minWeightFractionPerNode" => (
        "getMinWeightFractionPerNode",
        get_min_weight_fraction_per_node,
        param_min_weight_fraction_per_node
    ),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "seed" => ("getSeed", get_seed, param_seed),
    "varianceCol" => ("getVarianceCol", get_variance_col, param_variance_col),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    setters: [
        "setFeaturesCol" => (set__features_col, "featuresCol"),
        "setLeafCol" => (set__leaf_col, "leafCol"),
        "setPredictionCol" => (set__prediction_col, "predictionCol"),
        "setVarianceCol" => (set__variance_col, "varianceCol"),
    ];
    properties: [
        "depth" => (depth, "depth"),
        "featureImportances" => (feature_importances, "featureImportances"),
        "numFeatures" => (num_features, "numFeatures"),
        "numNodes" => (num_nodes, "numNodes"),
        "toDebugString" => (to_debug_string, "toDebugString"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictLeaf" => (predict_leaf, "predictLeaf", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: [];
    /// The supportedImpurities of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedImpurities() -> Vec<&'static str> {
        vec!["variance"]
    }
}

/// Register the typed regression models and summaries.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDecisionTreeRegressionModel>()?;
    Ok(())
}
