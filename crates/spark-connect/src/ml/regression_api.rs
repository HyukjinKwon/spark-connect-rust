// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of the regression models, over the server-side
//! objects of [`super::object`].
//!
//! Each estimator fits its typed model (`LinearRegression::fit` a [`LinearRegressionModel`]), whose
//! members are fetches of its server-side object, as PySpark's `_call_java` of each member. Each
//! family has its own file of its model's members, its summary types and its persistence.

use crate::ml::object::{MlTarget, MlValue};
use crate::ml::MlMatrix;
use spark_connect_core::error::{Result, SparkError};

mod aft;
mod fm;
mod generalized_linear;
pub use generalized_linear::*;
mod isotonic;
mod linear;
pub use linear::*;
mod tree;

// The targets of the training summary of a fit and of the summary of an `evaluate`, which keeps
// its model alive (see `crate::ml::object`).
pub(crate) use crate::ml::object::{evaluated_summary, training_summary};

/// The training summary of a model of the class `class`, or the error of PySpark's `summary` of a
/// model that has none (a model that was loaded rather than fitted).
pub(crate) fn summary_or_error(summary: &Option<MlTarget>, class: &str) -> Result<MlTarget> {
    summary.clone().ok_or_else(|| {
        SparkError::value_msg(format!("No training summary available for this {class}"))
    })
}

pub(crate) fn unexpected(method: &str, expected: &str, value: &MlValue) -> SparkError {
    SparkError::value_msg(format!("expected {expected} of `{method}`, got {value:?}"))
}

/// A long value of `method`, as PySpark reads a `Long` (or a narrower) literal.
pub(crate) fn as_i64(method: &str, value: MlValue) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| unexpected(method, "a long", &value))
}

/// An ML matrix value of `method`.
pub(crate) fn as_matrix(method: &str, value: MlValue) -> Result<MlMatrix> {
    match value {
        MlValue::Matrix(m) => Ok(m),
        other => Err(unexpected(method, "a matrix", &other)),
    }
}
