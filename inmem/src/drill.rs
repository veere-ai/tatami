//! Immutable row-authorized in-memory cube and its optional fact lineage.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::num::NonZeroU16;

use polars_core::prelude::{AnyValue, BooleanChunked, DataFrame};
use tatami::{Cube, drill, query, schema};

use crate::{InMemoryCube, eval, resolve};

/// Errors returned by the authorized cube or its inspection operation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Underlying query or cube construction error.
    #[error(transparent)]
    Cube(#[from] crate::Error),
    /// The visibility mask must contain exactly one decision per input row.
    #[error("the visibility mask does not match the source frame")]
    InvalidVisibility,
    /// The source could not be processed at its declared grain.
    #[error("source grain must use distinct, non-null, unique integer or string key columns")]
    InvalidGrain,
    /// Source revision or visibility changed since this cell was selected.
    #[error("the selected cube snapshot or visibility has changed")]
    Stale,
    /// The original query did not return the selected cell.
    #[error("the query did not return this cell")]
    NotSelected,
    /// The cursor came from another cell or no longer identifies a fact.
    #[error("the continuation does not belong to this selection")]
    InvalidCursor,
    /// A backend must prove exactly which facts contributed to the cell.
    #[error("fact lineage is unavailable: {0}")]
    Unsupported(&'static str),
    /// Processing failed on the authorized frame.
    #[error(transparent)]
    Frame(#[from] polars_core::error::PolarsError),
}

/// An immutable cube built from only the rows allowed by trusted host policy.
/// The host also supplies the permitted schema and versioned visibility identity.
#[derive(Debug)]
pub struct View {
    cube: InMemoryCube,
    snapshot: drill::Snapshot,
    grain: drill::Grain,
}

impl View {
    /// Scope rows before member discovery or aggregation. Null mask entries deny
    /// access. The fact key must be unique within the resulting authorized view.
    pub fn new(
        frame: DataFrame,
        schema: tatami::Schema,
        allowed: &BooleanChunked,
        snapshot: drill::Snapshot,
        grain: drill::Grain,
    ) -> Result<Self, Error> {
        if allowed.len() != frame.height() {
            return Err(Error::InvalidVisibility);
        }
        let frame = frame.filter(allowed)?;
        if grain.keys.is_empty()
            || grain.keys.iter().collect::<HashSet<_>>().len() != grain.keys.len()
        {
            return Err(Error::InvalidGrain);
        }
        for key in &grain.keys {
            let column = frame
                .column(key.as_str())
                .map_err(|_| Error::InvalidGrain)?;
            if !(column.dtype().is_integer()
                || column.dtype() == &polars_core::prelude::DataType::String)
            {
                return Err(Error::InvalidGrain);
            }
        }
        let mut seen = HashSet::with_capacity(frame.height());
        for row in 0..frame.height() {
            if !seen.insert(key(&frame, row, &grain)?) {
                return Err(Error::InvalidGrain);
            }
        }
        Ok(Self {
            cube: InMemoryCube::new(frame, schema)?,
            snapshot,
            grain,
        })
    }

    /// Identity to attach to a cell rendered by this view.
    #[must_use]
    pub fn snapshot(&self) -> &drill::Snapshot {
        &self.snapshot
    }

    /// Resolve and evaluate the original query, prove that the requested cell
    /// survived its axes and rank/limit, then prepare its ordered fact set once.
    pub fn select(&self, reference: &drill::Reference) -> Result<Selected, Error> {
        if reference.snapshot != self.snapshot {
            return Err(Error::Stale);
        }
        let request = &reference.query;
        let metric = request
            .metrics
            .get(reference.metric)
            .ok_or(Error::NotSelected)?;
        let resolved = self.cube.resolve(request)?;
        let results = eval::query::evaluate(&resolved, &self.cube)?;
        let (coordinates, value) = self.coordinate(reference, results, &resolved)?;
        if !matches!(value, tatami::Cell::Valid { .. }) {
            return Err(Error::Unsupported(
                "the selected metric has no numeric value",
            ));
        }
        let mut at = resolved.slicer;
        for coordinate in coordinates {
            let resolved_axis = self.cube.resolve(&tatami::Query {
                axes: tatami::Axes::Scalar,
                slicer: coordinate,
                metrics: request.metrics.clone(),
                options: query::Options::default(),
            })?;
            at = eval::query::intersect(&at, &resolved_axis.slicer).ok_or(Error::NotSelected)?;
        }
        let columns = components(metric, &self.cube.schema)?;
        let frame = eval::tuple::filter_by_tuple(&at, &self.cube.df)?;
        let mut totals = BTreeMap::new();
        for name in &columns {
            let definition = self
                .cube
                .schema
                .measures
                .iter()
                .find(|m| m.name == *name)
                .ok_or(Error::NotSelected)?;
            totals.insert(
                name.clone(),
                eval::aggregate::evaluate_measure(
                    &definition.with_aggregation(schema::Aggregation::Sum),
                    &at,
                    &self.cube.df,
                    &self.cube.schema,
                )?,
            );
        }
        let mut facts = Vec::new();
        for row in 0..frame.height() {
            let mut values = BTreeMap::new();
            for name in &columns {
                let v = frame.column(name.as_str())?.get(row)?;
                let value = match v {
                    AnyValue::Null => None,
                    v => Some(
                        v.extract::<f64>()
                            .filter(|v| v.is_finite())
                            .ok_or(Error::Unsupported("measure is not a finite number"))?,
                    ),
                };
                values.insert(name.clone(), value);
            }
            if values.values().any(Option::is_some) {
                facts.push(drill::Fact {
                    key: key(&frame, row, &self.grain)?,
                    values,
                });
            }
        }
        facts.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(Selected {
            reference: reference.clone(),
            grain: self.grain.clone(),
            value,
            totals,
            facts,
        })
    }

    fn coordinate(
        &self,
        reference: &drill::Reference,
        results: tatami::Results,
        resolved: &resolve::ResolvedQuery<'_>,
    ) -> Result<(Vec<tatami::Tuple>, tatami::Cell), Error> {
        let metric = reference.metric;
        match (&reference.selection, results) {
            (drill::Selection::Scalar, tatami::Results::Scalar(s)) => Ok((
                Vec::new(),
                s.values().get(metric).ok_or(Error::NotSelected)?.clone(),
            )),
            (drill::Selection::Series { member }, tatami::Results::Series(s)) => {
                let resolve::ResolvedAxes::Series { rows } = &resolved.axes else {
                    return Err(Error::NotSelected);
                };
                // Series results publish a single x member even for cross-joined axes.
                // That lossy projection cannot identify a contributing fact coordinate.
                if eval::set::evaluate(rows, &self.cube)?
                    .iter()
                    .any(|t| t.members.len() != 1)
                {
                    return Err(Error::Unsupported(
                        "series axis includes multi-member tuples",
                    ));
                }
                let pos = unique(
                    s.x()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, m)| (m == member).then_some(i)),
                )?;
                let cell = s
                    .rows()
                    .get(metric)
                    .and_then(|row| row.values.get(pos))
                    .ok_or(Error::NotSelected)?;
                Ok((vec![tatami::Tuple::single(member.clone())], cell.clone()))
            }
            (drill::Selection::Pivot { row, column }, tatami::Results::Pivot(p)) => {
                let r = unique(
                    p.row_headers()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, t)| (t == row).then_some(i)),
                )?;
                let count = reference.query.metrics.len();
                let c = unique(
                    p.col_headers()
                        .chunks_exact(count)
                        .enumerate()
                        .filter_map(|(i, group)| (&group[0] == column).then_some(i)),
                )?;
                let cell = p
                    .cells()
                    .get(r)
                    .and_then(|cells| cells.get(c * count + metric))
                    .ok_or(Error::NotSelected)?;
                Ok((vec![row.clone(), column.clone()], cell.clone()))
            }
            (_, tatami::Results::Rollup(_)) => {
                Err(Error::Unsupported("rollup lineage is not available"))
            }
            _ => Err(Error::NotSelected),
        }
    }
}

fn unique(mut positions: impl Iterator<Item = usize>) -> Result<usize, Error> {
    let first = positions.next().ok_or(Error::NotSelected)?;
    if positions.next().is_some() {
        return Err(Error::Unsupported("the result has duplicate coordinates"));
    }
    Ok(first)
}

fn key(frame: &DataFrame, row: usize, grain: &drill::Grain) -> Result<Vec<String>, Error> {
    grain
        .keys
        .iter()
        .map(|key| match frame.column(key.as_str())?.get(row)? {
            AnyValue::String(s) => Ok(s.to_owned()),
            AnyValue::StringOwned(s) => Ok(s.as_str().to_owned()),
            v if v.dtype().is_integer() => Ok(v.to_string()),
            _ => Err(Error::InvalidGrain),
        })
        .collect()
}

/// A materialized selection within one immutable authorized snapshot.
#[derive(Debug)]
pub struct Selected {
    reference: drill::Reference,
    grain: drill::Grain,
    value: tatami::Cell,
    totals: BTreeMap<schema::Name, tatami::Cell>,
    facts: Vec<drill::Fact>,
}

impl Selected {
    /// Return the next bounded page. Continuations identify the last fact's
    /// natural key, never an offset into a mutable or differently scoped frame.
    pub fn page(
        &self,
        after: Option<&drill::Cursor>,
        limit: NonZeroU16,
    ) -> Result<drill::Page, Error> {
        let start = match after {
            None => 0,
            Some(cursor) if cursor.reference == self.reference => {
                self.facts
                    .binary_search_by(|f| f.key.cmp(&cursor.after))
                    .map_err(|_| Error::InvalidCursor)?
                    + 1
            }
            Some(_) => return Err(Error::InvalidCursor),
        };
        let end = (start + usize::from(limit.get())).min(self.facts.len());
        let next = (end < self.facts.len()).then(|| drill::Cursor {
            reference: self.reference.clone(),
            after: self.facts[end - 1].key.clone(),
        });
        Ok(drill::Page {
            reference: self.reference.clone(),
            grain: self.grain.clone(),
            value: self.value.clone(),
            totals: self.totals.clone(),
            total: self.facts.len(),
            records: self.facts[start..end].to_vec(),
            next,
        })
    }
}

impl Cube for View {
    type Error = Error;
    async fn schema(&self) -> Result<tatami::Schema, Error> {
        Ok(self.cube.schema().await?)
    }
    async fn query(&self, q: &tatami::Query) -> Result<tatami::Results, Error> {
        Ok(self.cube.query(q).await?)
    }
    async fn members(
        &self,
        dim: &schema::Name,
        hierarchy: &schema::Name,
        at: &tatami::MemberRef,
        relation: tatami::MemberRelation,
    ) -> Result<Vec<tatami::MemberRef>, Error> {
        Ok(self.cube.members(dim, hierarchy, at, relation).await?)
    }
}

impl drill::Source for View {
    async fn inspect(
        &self,
        reference: &drill::Reference,
        after: Option<&drill::Cursor>,
        limit: NonZeroU16,
    ) -> Result<drill::Page, Error> {
        self.select(reference)?.page(after, limit)
    }
}

fn components(
    metric: &query::Metric,
    schema: &tatami::Schema,
) -> Result<BTreeSet<schema::Name>, Error> {
    let mut names = BTreeSet::new();
    match metric {
        query::Metric::Aggregated {
            column,
            agg: schema::Aggregation::Sum,
        } => {
            names.insert(column.clone());
        }
        query::Metric::Derived { name } => leaves(
            &tatami::Expr::Ref { name: name.clone() },
            schema,
            &mut HashSet::new(),
            &mut names,
        )?,
        _ => {
            return Err(Error::Unsupported(
                "only summed measures and their arithmetic expressions can be reconciled",
            ));
        }
    }
    if names.is_empty() {
        return Err(Error::Unsupported("the expression has no source measure"));
    }
    Ok(names)
}

fn leaves(
    expr: &tatami::Expr,
    schema: &tatami::Schema,
    visiting: &mut HashSet<schema::Name>,
    names: &mut BTreeSet<schema::Name>,
) -> Result<(), Error> {
    match expr {
        tatami::Expr::Ref { name } => {
            if let Some(m) = schema.measures.iter().find(|m| m.name == *name) {
                if !matches!(m.aggregation, schema::Aggregation::Sum) {
                    return Err(Error::Unsupported("a component measure is not summed"));
                }
                names.insert(name.clone());
            } else {
                let m = schema
                    .metrics
                    .iter()
                    .find(|m| m.name == *name)
                    .ok_or(Error::NotSelected)?;
                if !visiting.insert(name.clone()) {
                    return Err(Error::Unsupported("cyclic metric"));
                }
                leaves(&m.expr, schema, visiting, names)?;
                visiting.remove(name);
            }
        }
        tatami::Expr::Binary { l, r, .. } => {
            leaves(l, schema, visiting, names)?;
            leaves(r, schema, visiting, names)?;
        }
        tatami::Expr::Const { .. } => {}
        _ => return Err(Error::Unsupported("the expression changes its coordinate")),
    }
    Ok(())
}
