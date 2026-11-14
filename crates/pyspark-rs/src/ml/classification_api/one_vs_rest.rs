// SPDX-License-Identifier: Apache-2.0
//! The typed model of OneVsRest: OneVsRestModel, a container of the models of the classes of a fit,
//! with its classifier and its other params as Python values (as PySpark's `PipelineModel`).
use super::*;
use spark_connect::ml::{OperatorType, ReadResult};

/// OneVsRestModel: the models of the classes of a fit, with the classifier and the other params
/// that PySpark's OneVsRestModel holds. Its transform is the composite model's of the core. Its
/// `models` need the submodels of the composite, which the core keeps private (see `models`).
#[pyclass(name = "OneVsRestModel", module = "pyspark.ml.connect.classification")]
pub struct PyOneVsRestModel {
    core: object::ModelCore,
    /// The Python values of the identity params: the classifier.
    raw: Vec<(&'static str, Py<PyAny>)>,
}

impl PyOneVsRestModel {
    /// The model of a fit of `model`, the core's composite of the classes, with the classifier
    /// `classifier`. As PySpark's model, it has a uid of its own.
    pub(crate) fn fitted(
        py: Python<'_>,
        model: Box<dyn Model>,
        df: &DataFrame,
        classifier: Py<PyAny>,
    ) -> PyResult<Self> {
        let mut core =
            object::ModelCore::fitted(py, model, df, object::SummaryDataset::None, None)?;
        core.model.operator_mut().uid = params_api::random_uid(py, "OneVsRestModel")?;
        Ok(Self {
            core,
            raw: vec![("classifier", classifier)],
        })
    }

    fn __from_read<'py>(py: Python<'py>, read: ReadResult) -> PyResult<Py<PyAny>> {
        let core = object::ModelCore::loaded(
            py,
            "org.apache.spark.ml.classification.OneVsRestModel",
            "OneVsRestModel",
            read,
        )?;
        Ok(Py::new(
            py,
            Self {
                core,
                raw: Vec::new(),
            },
        )?
        .into_any())
    }
}

impl params_api::ParamsHolder for PyOneVsRestModel {
    fn set_value(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        if let Some((_, value)) = self.raw.iter().find(|(n, _)| *n == name) {
            return Ok(Some(value.clone_ref(py)));
        }
        params_api::literal_value(py, self.core.model.params(), name)
    }

    fn put(&mut self, py: Python<'_>, name: &'static str, value: Converted) -> PyResult<()> {
        self.raw.retain(|(n, _)| *n != name);
        let params = self.core.model.params().clone();
        *self.core.model.params_mut() = match value {
            Converted::Raw(value) => {
                self.raw.push((name, value));
                params.clear_param(name)
            }
            value => object::put_model_param(py, params, name, value)?,
        };
        Ok(())
    }

    fn own_default(&self, py: Python<'_>, name: &str) -> Option<Py<PyAny>> {
        params_api::own_default_of(py, &self.core.own_defaults, name)
    }

    fn put_own_default(&mut self, name: &'static str, value: Py<PyAny>) -> bool {
        params_api::put_own_default_in(&mut self.core.own_defaults, name, value);
        true
    }

    fn unset(&mut self, name: &'static str) -> PyResult<()> {
        self.raw.retain(|(n, _)| *n != name);
        let params = self.core.model.params().clone().clear_param(name);
        *self.core.model.params_mut() = params;
        Ok(())
    }
}

impl params_api::NativeInstance for PyOneVsRestModel {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        let raw = self
            .raw
            .iter()
            .map(|(name, value)| (*name, value.clone_ref(py)))
            .collect();
        Py::new(
            py,
            Self {
                core: self.core.duplicate(py),
                raw,
            },
        )
    }

    fn uid(&self) -> String {
        self.core.uid()
    }

    fn set_uid(&mut self, uid: &str) {
        self.core.model.operator_mut().uid = uid.to_string();
    }
}

native_pymethods! {
    PyOneVsRestModel;
    persistence: python;
    params: [
        "classifier" => ("getClassifier", get_classifier, param_classifier),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];

    /// The models of the classes, as PySpark's `models` attribute. The core keeps the submodels
    /// of its composite model private, and the composite has no server-side object of its own to
    /// fetch them from, so this is NOT_IMPLEMENTED until the core exposes them.
    #[getter]
    fn models(slf: &Bound<'_, Self>) -> PyResult<Py<PyList>> {
        Err(crate::column::raise_pyspark(
            slf.py(),
            "PySparkNotImplementedError",
            "NOT_IMPLEMENTED",
            &[("feature", "OneVsRestModel.models")],
        ))
    }

    /// The transform of `dataset`, as `Transformer.transform`: of a copy of the model of a
    /// non-empty param map `params`.
    #[pyo3(signature = (dataset, params=None))]
    fn transform<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let map = params.filter(|p| p.cast::<PyDict>().is_ok_and(|d| !d.is_empty()));
        if let Some(map) = map {
            let copy = params_api::copy(slf, Some(map))?;
            return Ok(copy.call_method1("transform", (dataset,))?.unbind());
        }
        crate::ml::check_params(params)?;
        let input = crate::ml::dataset_of(dataset, "transform")?;
        let df = slf.try_borrow()?.core.transform(slf.py(), &input)?;
        Ok(Py::new(slf.py(), PyDataFrame::new(df))?.into_any())
    }

    /// A copy of the same uid and params, with `extra`, as `Params.copy`.
    #[pyo3(name = "copy", signature = (extra=None))]
    fn __params_copy<'py>(
        slf: &Bound<'py, Self>,
        extra: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        params_api::copy(slf, extra)
    }

    /// The writer of the model, as `JavaMLWritable.write` (`RemoteMLWriter`).
    fn write(slf: &Bound<'_, Self>) -> PyResult<object::PyMLWriter> {
        slf.try_borrow()?.core.writer(slf.py())
    }

    /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    /// The reader of the class, as `JavaMLReadable.read` (`RemoteMLReader`).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> object::PyMLReader {
        let _ = cls;
        object::PyMLReader::new(
            "org.apache.spark.ml.classification.OneVsRestModel",
            OperatorType::Model,
            PyOneVsRestModel::__from_read,
        )
    }

    /// Load from `path`, as `MLReadable.load`: `cls.read().load(path)`.
    #[classmethod]
    fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
        Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_features_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        params_api::set_by_name(&slf, "featuresCol", value)?;
        Ok(slf)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        params_api::set_by_name(&slf, "predictionCol", value)?;
        Ok(slf)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        params_api::set_by_name(&slf, "rawPredictionCol", value)?;
        Ok(slf)
    }
}

/// Register the typed classes of the family.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOneVsRestModel>()?;
    Ok(())
}
