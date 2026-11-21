//! Optional lineage for a cell in a particular authorized version of a cube.
//!
//! A snapshot identifies an authorized view; it is not an access credential.
//! Servers construct the view from their current policy before evaluating any
//! query or inspection. Client-supplied coordinates and cursors never expand it.

use std::collections::BTreeMap;
use std::num::NonZeroU16;

use serde::{Deserialize, Serialize};

use crate::schema::Name;
use crate::{Cell, Cube, MemberRef, Query, Tuple};

/// Authoritative identity for immutable facts, schema and visibility.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Dataset identity, independent of its display label.
    pub source: Name,
    /// Changes whenever source facts, grain, or cube formulas change.
    pub revision: Name,
    /// Host-issued identity for the principal's effective row and field policy.
    pub visibility: Name,
}

/// The returned coordinate selected by the viewer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Selection {
    /// A scalar KPI at the query slicer.
    Scalar,
    /// A point on a single-member series axis.
    Series {
        /// Fully qualified member from the displayed series result.
        member: MemberRef,
    },
    /// The intersection of displayed pivot row and column coordinates.
    Pivot {
        /// Tuple from the displayed row header.
        row: Tuple,
        /// Tuple from the displayed column header.
        column: Tuple,
    },
}

/// A reproducible, serializable selected-cell reference. Rank, limits, axes,
/// and hierarchy pins remain with the original query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    /// Authorized snapshot used to render the cell.
    pub snapshot: Snapshot,
    /// Exact applied query.
    pub query: Query,
    /// Coordinate selected from the query result.
    pub selection: Selection,
    /// Index into the query's requested metrics.
    pub metric: usize,
}

/// Natural key declaration for one fact at the source's actual grain.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Grain {
    /// Source meaning, e.g. monthly close fact.
    pub name: Name,
    /// Ordered columns of the stable, unique source key.
    pub keys: Vec<Name>,
}

/// One contributing source fact. Key components align with the grain
/// declaration. Values include only the selected metric's measure inputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    /// Natural key components in declared order.
    pub key: Vec<String>,
    /// Raw numeric facts; a null is not a zero.
    pub values: BTreeMap<Name, Option<f64>>,
}

/// Exclusive continuation for a single authorized selected cell.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cursor {
    /// Exact reference for which this continuation was issued.
    pub reference: Reference,
    /// Last delivered fact key.
    pub after: Vec<String>,
}

/// A bounded, stable page from the contributing facts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Page {
    /// Versioned cell identity suitable for attaching to a chat or workbook.
    pub reference: Reference,
    /// What one returned record represents.
    pub grain: Grain,
    /// Value calculated by the scoped cube engine for this cell.
    pub value: Cell,
    /// Each component measure aggregated over all matching authorized facts.
    pub totals: BTreeMap<Name, Cell>,
    /// Count of contributing facts across all pages.
    pub total: usize,
    /// Facts sorted by their natural key.
    pub records: Vec<Fact>,
    /// Continuation if more records remain.
    pub next: Option<Cursor>,
}

/// Optional backend capability. Implementors must establish authorization
/// before evaluating axes, aggregates, member discovery, and fact lookup.
pub trait Source: Cube {
    /// Inspect one selected cell in the current authorized snapshot.
    fn inspect(
        &self,
        reference: &Reference,
        after: Option<&Cursor>,
        limit: NonZeroU16,
    ) -> impl std::future::Future<Output = Result<Page, Self::Error>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Axes, Path, query};

    #[test]
    fn saved_cursor_roundtrips_source_visibility_metric_and_member_path() {
        let name = |s: &str| Name::parse(s).expect("valid name");
        let member = MemberRef::new(
            name("Geography"),
            name("World"),
            Path::parse(vec![name("LAC"), name("AR")]).expect("qualified member"),
        );
        let cursor = Cursor {
            reference: Reference {
                snapshot: Snapshot {
                    source: name("close"),
                    revision: name("2023-v3"),
                    visibility: name("finance-v2"),
                },
                query: Query {
                    axes: Axes::Series {
                        rows: crate::Set::members(
                            name("Geography"),
                            name("World"),
                            name("Country"),
                        ),
                    },
                    slicer: Tuple::empty(),
                    metrics: vec![query::Metric::derived(name("CombinedRatio"))],
                    options: query::Options::default(),
                },
                selection: Selection::Series { member },
                metric: 0,
            },
            after: vec!["2023".into(), "AR".into(), "AGRI".into()],
        };
        let wire = serde_json::to_string(&cursor).expect("serialize cursor");
        assert_eq!(
            serde_json::from_str::<Cursor>(&wire).expect("restore cursor"),
            cursor
        );
    }
}
