//! [`Options`] (query-level tuning knobs), [`OrderBy`], [`OrderKey`], [`Rank`],
//! [`Direction`].

use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use crate::schema::Name;

/// Optional query-level tuning knobs. Defaults: no ordering, no rank, no
/// limit, empty tuples kept.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Options {
    /// Display orderings over the result positions, in priority order (first
    /// is primary). Each orders by a dimension member or a metric value.
    #[serde(default)]
    pub order: Vec<OrderBy>,
    /// Top-N rank: keep the `limit` most-extreme positions by this metric
    /// *before* `order` is applied, so a result can be ranked by one metric
    /// yet displayed in another order. `None` + `limit` truncates the display
    /// order directly.
    #[serde(default)]
    pub rank: Option<Rank>,
    /// Cap on the number of result positions returned. `None` means unlimited.
    #[serde(default)]
    pub limit: Option<NonZeroUsize>,
    /// MDX `NON EMPTY` — drop tuples whose cells are all missing.
    #[serde(default)]
    pub non_empty: bool,
}

/// A single display-ordering directive — by a dimension member or a metric
/// value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrderBy {
    /// What to order by.
    pub key: OrderKey,
    /// The direction to sort.
    pub direction: Direction,
}

/// What an [`OrderBy`] orders by: a dimension's member key (e.g. a time axis
/// chronologically) or a metric's value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderKey {
    /// Order by a dimension's member key.
    Dimension(Name),
    /// Order by a metric's value.
    Metric(Name),
}

/// A top-N rank: keep the most-extreme result positions by `metric`,
/// independent of the display [`OrderBy`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rank {
    /// The metric to rank by.
    pub metric: Name,
    /// `Desc` = largest first (the usual "top N"); `Asc` = "bottom N".
    pub direction: Direction,
}

/// Sort direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Direction {
    /// Ascending — smallest first.
    Asc,
    /// Descending — largest first.
    Desc,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: &str) -> Name {
        Name::parse(s).expect("valid")
    }

    #[test]
    fn default_options_roundtrip() {
        let o = Options::default();
        let json = serde_json::to_string(&o).expect("serialize");
        let back: Options = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(o, back);
    }

    #[test]
    fn options_with_order_rank_and_limit_roundtrip() {
        let o = Options {
            order: vec![OrderBy {
                key: OrderKey::Dimension(n("Date")),
                direction: Direction::Asc,
            }],
            rank: Some(Rank {
                metric: n("Revenue"),
                direction: Direction::Desc,
            }),
            limit: Some(NonZeroUsize::new(10).expect("nonzero")),
            non_empty: true,
        };
        let json = serde_json::to_string(&o).expect("serialize");
        let back: Options = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(o, back);
    }

    #[test]
    fn direction_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&Direction::Asc).expect("serialize"),
            r#""asc""#
        );
        assert_eq!(
            serde_json::to_string(&Direction::Desc).expect("serialize"),
            r#""desc""#
        );
    }
}
