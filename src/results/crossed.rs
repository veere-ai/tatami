//! A series grouped by several dimensions on one axis.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::series;
use crate::query::Tuple;
use crate::schema::Name;

/// One coordinate column, in the grouping order requested by the query.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    /// The dimension identifying this coordinate.
    pub dim: Name,
    /// The hierarchy within the dimension.
    pub hierarchy: Name,
}

/// Complete coordinates and their aligned metric vectors. Group descriptors
/// survive slicing down to zero rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Wire")]
pub struct Result {
    groups: Vec<Group>,
    x: Vec<Tuple>,
    rows: Vec<series::Row>,
}

#[derive(Deserialize)]
struct Wire {
    groups: Vec<Group>,
    x: Vec<Tuple>,
    rows: Vec<series::Row>,
}

impl TryFrom<Wire> for Result {
    type Error = Error;

    fn try_from(wire: Wire) -> std::result::Result<Self, Error> {
        Self::new(wire.groups, wire.x, wire.rows)
    }
}

impl Result {
    /// Bind coordinate tuples and metric rows to an ordered grouping.
    pub fn new(
        groups: Vec<Group>,
        x: Vec<Tuple>,
        rows: Vec<series::Row>,
    ) -> std::result::Result<Self, Error> {
        let mut seen = HashSet::new();
        if groups.len() < 2 || !groups.iter().all(|group| seen.insert(&group.dim)) {
            return Err(Error::Groups);
        }
        for (index, tuple) in x.iter().enumerate() {
            if tuple.len() != groups.len()
                || !tuple.members().iter().zip(&groups).all(|(member, group)| {
                    member.dim == group.dim && member.hierarchy == group.hierarchy
                })
            {
                return Err(Error::Coordinate { index });
            }
        }
        for (index, row) in rows.iter().enumerate() {
            if row.values.len() != x.len() {
                return Err(Error::Width {
                    index,
                    expected: x.len(),
                    actual: row.values.len(),
                });
            }
        }
        Ok(Self { groups, x, rows })
    }

    /// Ordered coordinate descriptors, including for an empty domain.
    #[must_use]
    pub fn groups(&self) -> &[Group] {
        &self.groups
    }

    /// The complete coordinate at each position in every metric row.
    #[must_use]
    pub fn x(&self) -> &[Tuple] {
        &self.x
    }

    /// One row per requested metric.
    #[must_use]
    pub fn rows(&self) -> &[series::Row] {
        &self.rows
    }
}

/// Violations of the crossed-series wire contract.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A crossing needs multiple, distinct dimensions.
    #[error("crossed series requires two or more distinct dimensions")]
    Groups,
    /// A tuple does not bind all groups in the declared order and hierarchies.
    #[error("crossed series coordinate {index} does not match the ordered groups")]
    Coordinate {
        /// Position in the domain.
        index: usize,
    },
    /// A metric's cell count differs from its coordinate count.
    #[error("crossed series metric {index} has {actual} cells for {expected} coordinates")]
    Width {
        /// Position in the metric rows.
        index: usize,
        /// Domain length.
        expected: usize,
        /// Length of the received metric vector.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use crate::Results;
    use serde_json::{Value, json};

    fn payload() -> Value {
        json!({"crossed_series":{
            "groups":[{"dim":"year","hierarchy":"Fiscal"},{"dim":"scenario","hierarchy":"Default"}],
            "x":[[{"dim":"year","hierarchy":"Fiscal","path":["2025"]},{"dim":"scenario","hierarchy":"Default","path":["budget"]}]],
            "rows":[{"label":"nep","values":[{"state":"valid","value":0.0,"unit":null,"format":null}]}]
        }})
    }

    #[test]
    fn crossed_coordinates_round_trip_and_an_old_reader_rejects_the_tag() {
        let json = payload();
        let result: Results = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(result).unwrap(), json);
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum Legacy {
            Series(serde::de::IgnoredAny),
        }
        assert!(serde_json::from_value::<Legacy>(json).is_err());

        let ordinary = r#"{"series":{"x":[{"dim":"year","hierarchy":"Fiscal","path":["2025"]}],"rows":[{"label":"nep","values":[{"state":"valid","value":0.0,"unit":null,"format":null}]}]}}"#;
        let result: Results = serde_json::from_str(ordinary).unwrap();
        assert!(matches!(result, Results::Series(_)));
        assert_eq!(serde_json::to_string(&result).unwrap(), ordinary);
    }

    #[test]
    fn invalid_axes_and_misaligned_metric_rows_are_rejected_at_decode() {
        for (pointer, replacement) in [
            ("/crossed_series/groups/1/dim", json!("year")),
            ("/crossed_series/x/0/1/hierarchy", json!("Other")),
            ("/crossed_series/x/0/0/dim", json!("scenario")),
            ("/crossed_series/x/0", json!([])),
            ("/crossed_series/groups", json!([])),
            ("/crossed_series/rows/0/values", json!([])),
        ] {
            let mut json = payload();
            *json.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                serde_json::from_value::<Results>(json).is_err(),
                "{pointer}"
            );
        }
    }
}
