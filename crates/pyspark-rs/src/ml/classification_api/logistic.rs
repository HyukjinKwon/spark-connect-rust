// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of LogisticRegression: LogisticRegressionModel and its four summaries.
use super::*;

/// Register the typed classes of the family.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
