// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of the decision tree, random forest and
//! gradient-boosted tree classification models, and of their estimators, as PySpark's
//! `pyspark.ml.classification` and `pyspark.ml.tree` have them.
//!
//! A random forest's and a gradient-boosted model's trees are server-side objects of their own
//! ([`TreeEnsembleModel::tree_of`]). A fit of a random forest sets its training summary to the
//! `<model>.summary` object, with the transform of the training dataset, from which the server
//! recreates the summary when it has lost it.

#[cfg(test)]
mod golden_tests;

use spark_connect_core::error::{Result, SparkError};

use super::{
    BinaryClassificationSummary, ClassificationModel, ClassificationSummary, MlSummary,
    ProbabilisticClassificationModel, TrainingSummary,
};
use crate::dataframe::DataFrame;
use crate::ml::tree_api::{as_f64s, fetch_value};
use crate::ml::{
    fetch, fit_via_command, read_model, read_operator, DecisionTreeClassificationModel,
    DecisionTreeClassifier, DecisionTreeModel, Estimator, FetchArg, GBTClassificationModel,
    GBTClassifier, MlOperator, MlRef, MlTarget, Model, OperatorType, Params, PredictionModel,
    RandomForestClassificationModel, RandomForestClassifier, TreeEnsembleModel,
};
use crate::SparkSession;

const DECISION_TREE_CLASSIFIER: &str = "org.apache.spark.ml.classification.DecisionTreeClassifier";
const DECISION_TREE_MODEL: &str =
    "org.apache.spark.ml.classification.DecisionTreeClassificationModel";
const GBT_CLASSIFIER: &str = "org.apache.spark.ml.classification.GBTClassifier";
const GBT_MODEL: &str = "org.apache.spark.ml.classification.GBTClassificationModel";
const RANDOM_FOREST_CLASSIFIER: &str = "org.apache.spark.ml.classification.RandomForestClassifier";
const RANDOM_FOREST_MODEL: &str =
    "org.apache.spark.ml.classification.RandomForestClassificationModel";

/// The impurities of a classification tree (`_TreeClassifierParams.supportedImpurities`).
const CLASSIFIER_IMPURITIES: &[&str] = &["entropy", "gini"];
/// The impurity of a gradient-boosted tree (`_HasVarianceImpurity.supportedImpurities`).
const VARIANCE_IMPURITY: &[&str] = &["variance"];
/// The strategies of the features of each tree of an ensemble
/// (`_TreeEnsembleParams.supportedFeatureSubsetStrategies`).
const FEATURE_SUBSET_STRATEGIES: &[&str] = &["auto", "all", "onethird", "sqrt", "log2"];
/// The loss types of a gradient-boosted classifier (`_GBTClassifierParams.supportedLossTypes`).
const LOSS_TYPES: &[&str] = &["logistic"];

// The estimators: the typed fit, the persistence and the client-side lists of PySpark.

impl DecisionTreeClassifier {
    /// Fit the decision tree classification model of `df` (`fit`), typed.
    pub fn fit(&mut self, df: &DataFrame) -> Result<DecisionTreeClassificationModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator that `path` was saved with (`DecisionTreeClassifier.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(
            session,
            DECISION_TREE_CLASSIFIER,
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
    }

    /// The impurities that the tree supports (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        CLASSIFIER_IMPURITIES
    }
}

impl RandomForestClassifier {
    /// Fit the random forest classification model of `df` (`fit`), typed. Its training summary
    /// is the summary of its `<model>.summary` object, which the server recreates from `df`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<RandomForestClassificationModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        model.training_summary = Some(
            MlTarget::object(&model.cached)
                .child("summary")
                .with_summary_dataset(df),
        );
        Ok(model)
    }

    /// The estimator that `path` was saved with (`RandomForestClassifier.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(
            session,
            RANDOM_FOREST_CLASSIFIER,
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
    }

    /// The strategies of the features of each tree (`supportedFeatureSubsetStrategies`).
    pub fn supported_feature_subset_strategies(&self) -> &'static [&'static str] {
        FEATURE_SUBSET_STRATEGIES
    }

    /// The impurities that the trees support (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        CLASSIFIER_IMPURITIES
    }
}

impl GBTClassifier {
    /// Fit the gradient-boosted classification model of `df` (`fit`), typed.
    pub fn fit(&mut self, df: &DataFrame) -> Result<GBTClassificationModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator that `path` was saved with (`GBTClassifier.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, GBT_CLASSIFIER, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }

    /// The strategies of the features of each tree (`supportedFeatureSubsetStrategies`).
    pub fn supported_feature_subset_strategies(&self) -> &'static [&'static str] {
        FEATURE_SUBSET_STRATEGIES
    }

    /// The impurity that the trees support (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        VARIANCE_IMPURITY
    }

    /// The loss types that the classifier supports (`supportedLossTypes`).
    pub fn supported_loss_types(&self) -> &'static [&'static str] {
        LOSS_TYPES
    }
}

// The decision tree classification model.

impl DecisionTreeClassificationModel {
    /// The model of the server-side object `ml_ref`, with the params `params`, which the trees of
    /// a forest are built of. It has a new uid.
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(DECISION_TREE_MODEL, OperatorType::Model),
            params,
            cached: ml_ref,
        }
    }

    /// The model that `path` was saved with (`DecisionTreeClassificationModel.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(session, DECISION_TREE_MODEL, path)?;
        Ok(Self {
            operator,
            ..Self::from_ml_ref(object, params)
        })
    }

    /// The impurities that the tree supports (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        CLASSIFIER_IMPURITIES
    }
}

impl DecisionTreeModel for DecisionTreeClassificationModel {}

impl ClassificationModel for DecisionTreeClassificationModel {}

impl ProbabilisticClassificationModel for DecisionTreeClassificationModel {}

impl PredictionModel for DecisionTreeClassificationModel {}

// The random forest classification model and its summaries.

impl RandomForestClassificationModel {
    /// The model of the server-side object `ml_ref`, with the params `params`. It has a new uid
    /// and no training summary.
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(RANDOM_FOREST_MODEL, OperatorType::Model),
            params,
            cached: ml_ref,
            training_summary: None,
        }
    }

    /// The model that `path` was saved with (`RandomForestClassificationModel.load`). It has no
    /// training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(session, RANDOM_FOREST_MODEL, path)?;
        Ok(Self {
            operator,
            ..Self::from_ml_ref(object, params)
        })
    }

    /// Whether the model has a training summary (`hasSummary`): only a model that a fit returned
    /// has one.
    pub fn has_summary(&self) -> bool {
        self.training_summary.is_some()
    }

    /// The training summary (`summary`): the binary one when the model has two classes or fewer,
    /// as PySpark chooses it by `numClasses`. Errors when the model has no training summary.
    pub fn summary(&self) -> Result<RandomForestTrainingSummary> {
        let target = self.training_summary.clone().ok_or_else(|| {
            SparkError::value_msg("the random forest model has no training summary")
        })?;
        if self.num_classes()? <= 2 {
            Ok(RandomForestTrainingSummary::Binary(
                BinaryRandomForestClassificationTrainingSummary::new(target),
            ))
        } else {
            Ok(RandomForestTrainingSummary::Multiclass(
                RandomForestClassificationTrainingSummary::new(target),
            ))
        }
    }

    /// The summary of the model on `dataset` (`evaluate`), of the class that `summary` chooses.
    pub fn evaluate(&self, dataset: &DataFrame) -> Result<RandomForestSummary> {
        let model = self.model_target()?;
        let id =
            fetch(&model, "evaluate", vec![FetchArg::Input(dataset.clone())])?.into_string()?;
        let target = MlTarget::unowned(model.session(), &id);
        if self.num_classes()? <= 2 {
            Ok(RandomForestSummary::Binary(
                BinaryRandomForestClassificationSummary::new(target),
            ))
        } else {
            Ok(RandomForestSummary::Multiclass(
                RandomForestClassificationSummary::new(target),
            ))
        }
    }

    /// The strategies of the features of each tree (`supportedFeatureSubsetStrategies`).
    pub fn supported_feature_subset_strategies(&self) -> &'static [&'static str] {
        FEATURE_SUBSET_STRATEGIES
    }

    /// The impurities that the trees support (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        CLASSIFIER_IMPURITIES
    }
}

impl TreeEnsembleModel for RandomForestClassificationModel {
    type Tree = DecisionTreeClassificationModel;

    fn tree_of(&self, tree: MlRef) -> DecisionTreeClassificationModel {
        DecisionTreeClassificationModel::from_ml_ref(tree, Params::new())
    }
}

impl ClassificationModel for RandomForestClassificationModel {}

impl ProbabilisticClassificationModel for RandomForestClassificationModel {}

impl PredictionModel for RandomForestClassificationModel {}

/// The training summary of a random forest classification model, of the class that PySpark
/// chooses by `numClasses`: binary for two classes or fewer.
#[derive(Debug, Clone)]
pub enum RandomForestTrainingSummary {
    /// The summary of a binary model (`BinaryRandomForestClassificationTrainingSummary`).
    Binary(BinaryRandomForestClassificationTrainingSummary),
    /// The summary of a model with more than two classes
    /// (`RandomForestClassificationTrainingSummary`).
    Multiclass(RandomForestClassificationTrainingSummary),
}

/// The summary of a random forest classification model on a dataset, of the class that PySpark
/// chooses by `numClasses`: binary for two classes or fewer.
#[derive(Debug, Clone)]
pub enum RandomForestSummary {
    /// The summary of a binary model (`BinaryRandomForestClassificationSummary`).
    Binary(BinaryRandomForestClassificationSummary),
    /// The summary of a model with more than two classes (`RandomForestClassificationSummary`).
    Multiclass(RandomForestClassificationSummary),
}

/// The summary of a random forest classification model on a dataset
/// (`RandomForestClassificationSummary`): the members of a classification summary.
#[derive(Debug, Clone)]
pub struct RandomForestClassificationSummary {
    target: MlTarget,
}

impl RandomForestClassificationSummary {
    /// The summary of `target`, the target of the summary's server-side object (as PySpark's
    /// summary constructors take the object of the summary).
    pub fn new(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for RandomForestClassificationSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for RandomForestClassificationSummary {}

/// The training summary of a random forest classification model with more than two classes
/// (`RandomForestClassificationTrainingSummary`).
#[derive(Debug, Clone)]
pub struct RandomForestClassificationTrainingSummary {
    target: MlTarget,
}

impl RandomForestClassificationTrainingSummary {
    /// The summary of `target`, the target of the summary's server-side object (as PySpark's
    /// summary constructors take the object of the summary).
    pub fn new(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for RandomForestClassificationTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for RandomForestClassificationTrainingSummary {}

impl TrainingSummary for RandomForestClassificationTrainingSummary {}

/// The summary of a binary random forest classification model on a dataset
/// (`BinaryRandomForestClassificationSummary`): the members of a binary classification summary.
#[derive(Debug, Clone)]
pub struct BinaryRandomForestClassificationSummary {
    target: MlTarget,
}

impl BinaryRandomForestClassificationSummary {
    /// The summary of `target`, the target of the summary's server-side object (as PySpark's
    /// summary constructors take the object of the summary).
    pub fn new(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for BinaryRandomForestClassificationSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for BinaryRandomForestClassificationSummary {}

impl BinaryClassificationSummary for BinaryRandomForestClassificationSummary {}

/// The training summary of a binary random forest classification model
/// (`BinaryRandomForestClassificationTrainingSummary`).
#[derive(Debug, Clone)]
pub struct BinaryRandomForestClassificationTrainingSummary {
    target: MlTarget,
}

impl BinaryRandomForestClassificationTrainingSummary {
    /// The summary of `target`, the target of the summary's server-side object (as PySpark's
    /// summary constructors take the object of the summary).
    pub fn new(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for BinaryRandomForestClassificationTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for BinaryRandomForestClassificationTrainingSummary {}

impl BinaryClassificationSummary for BinaryRandomForestClassificationTrainingSummary {}

impl TrainingSummary for BinaryRandomForestClassificationTrainingSummary {}

// The gradient-boosted classification model.

impl GBTClassificationModel {
    /// The model of the server-side object `ml_ref`, with the params `params`. It has a new uid.
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(GBT_MODEL, OperatorType::Model),
            params,
            cached: ml_ref,
        }
    }

    /// The model that `path` was saved with (`GBTClassificationModel.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(session, GBT_MODEL, path)?;
        Ok(Self {
            operator,
            ..Self::from_ml_ref(object, params)
        })
    }

    /// The loss of the model after each boosting iteration, on `dataset`
    /// (`evaluateEachIteration`).
    pub fn evaluate_each_iteration(&self, dataset: &DataFrame) -> Result<Vec<f64>> {
        let value = fetch_value(
            &self.model_target()?,
            "evaluateEachIteration",
            vec![FetchArg::Input(dataset.clone())],
        )?;
        as_f64s("evaluateEachIteration", value)
    }

    /// The strategies of the features of each tree (`supportedFeatureSubsetStrategies`).
    pub fn supported_feature_subset_strategies(&self) -> &'static [&'static str] {
        FEATURE_SUBSET_STRATEGIES
    }

    /// The impurity that the trees support (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        VARIANCE_IMPURITY
    }

    /// The loss types that the model supports (`supportedLossTypes`).
    pub fn supported_loss_types(&self) -> &'static [&'static str] {
        LOSS_TYPES
    }
}

impl ClassificationModel for GBTClassificationModel {}

impl ProbabilisticClassificationModel for GBTClassificationModel {}

impl PredictionModel for GBTClassificationModel {}
