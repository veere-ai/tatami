//! [`Metric`] — a metric requested at each cell of a query.

use serde::{Deserialize, Serialize};

use crate::schema::{Aggregation, Name};

/// A metric requested at each cell of a [`Query`](crate::Query).
///
/// Either an **aggregated** stored measure — a column plus an explicit
/// function that *overrides* the column's declared aggregation — or a
/// **derived** metric referenced by name and evaluated as the schema defines
/// it. `Aggregated` makes the query the sole authority for α: the function is
/// carried in the query, not read from the schema, so a measure's declared
/// aggregation is only an authoring-time default. `Derived` defers to the
/// schema — a formula metric's expression, or (when the name is a stored
/// measure) that measure's own declared aggregation.
///
/// Compare with `schema::Metric` (a stored formula definition) — this is a
/// *request*, that is a *definition*.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Metric {
    /// Aggregate the stored measure `column` by `agg`, ignoring the column's
    /// schema-declared aggregation.
    Aggregated {
        /// The measure column to aggregate.
        column: Name,
        /// The aggregation applied to it for this query.
        agg: Aggregation,
    },
    /// A metric referenced by name, evaluated as the schema defines it — a
    /// formula metric's expression, or a stored measure by its own declared
    /// aggregation.
    Derived {
        /// The metric name.
        name: Name,
    },
}

impl Metric {
    /// An aggregated stored measure — `column` reduced by `agg`.
    #[must_use]
    pub fn aggregated(column: Name, agg: Aggregation) -> Self {
        Self::Aggregated { column, agg }
    }

    /// A metric referenced by `name`, evaluated by the schema's definition (a
    /// formula metric, or a stored measure's declared aggregation).
    #[must_use]
    pub fn derived(name: Name) -> Self {
        Self::Derived { name }
    }

    /// The name this metric references — the measure column for
    /// [`Metric::Aggregated`], the formula name for [`Metric::Derived`].
    /// Used for display and for matching [`Options`](crate::query::Options)
    /// order/rank keys against the requested metrics.
    #[must_use]
    pub fn name(&self) -> &Name {
        match self {
            Self::Aggregated { column, .. } => column,
            Self::Derived { name } => name,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: &str) -> Name {
        Name::parse(s).expect("valid")
    }

    #[test]
    fn aggregated_roundtrips() {
        let m = Metric::aggregated(n("close"), Aggregation::avg());
        let json = serde_json::to_string(&m).expect("serialize");
        let back: Metric = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(m, back);
    }

    #[test]
    fn derived_roundtrips_stable() {
        let m = Metric::derived(n("Revenue"));
        let json = serde_json::to_string(&m).expect("serialize");
        assert_eq!(json, r#"{"kind":"derived","name":"Revenue"}"#);
        let back: Metric = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(m, back);
    }

    #[test]
    fn name_reads_through_both_variants() {
        assert_eq!(
            Metric::aggregated(n("close"), Aggregation::sum())
                .name()
                .as_str(),
            "close"
        );
        assert_eq!(Metric::derived(n("Revenue")).name().as_str(), "Revenue");
    }
}
