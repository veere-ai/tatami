//! Query orchestration — Phase 5g of MAP_PLAN.md §5.
//!
//! Takes a [`ResolvedQuery`] and folds it into a public [`Results`] value,
//! following the `Axes → Results` table from §3.3:
//!
//! | `Axes`                         | `Results`                                                |
//! |---                             |---                                                       |
//! | `Scalar`                       | `Scalar`                                                 |
//! | `Series { rows }`              | `Series` or `CrossedSeries` for a crossed grouping        |
//! | `Pivot { rows, columns }`      | `Pivot` — or `Rollup` when `rows` is a top-level, single-rooted `Descendants` |
//! | `Pages { rows, columns, pages }` | `Pivot` with stacked column headers (page × column)    |
//!
//! "Single-rooted" means the flat `Descendants` output shares one
//! top-level ancestor (same dim, same hierarchy, same `path[0]`). A
//! multi-rooted descendants set — e.g. `Set::range(FY2025..FY2030).descendants_to(Quarter)`
//! spanning six fiscal years — can't fit a single-rooted [`rollup::Tree`]
//! and falls back to a flat Pivot. Collapsing every root under one
//! synthetic ancestor would silently drop subtrees (MAP §8 R3).
//!
//! ### Shape assembly
//!
//! 1. Evaluate each axis's [`ResolvedSet`] into a `Vec<ResolvedTuple>`.
//! 2. For every grid position `(row, column, page, slicer)`, intersect
//!    the tuples by AUTOEXISTS semantics (MDX): for each dim, prefer
//!    the deeper-path member; incompatible same-dim members collapse
//!    the tuple, and that cell becomes `Cell::Missing`. Dims disjoint
//!    across axes pass through verbatim.
//! 3. Evaluate every metric at the intersected tuple via
//!    [`crate::eval::metric::evaluate_expr`].
//! 4. Assemble per the §3.3 table.
//!
//! Multi-metric pivots widen `col_headers` so the grid stays rectangular:
//! for `m` metrics and `c` column tuples, `col_headers.len() == c * m`,
//! grouped by column-then-metric. Callers reading `cells[r][c * m + k]`
//! address row `r`'s column `c` metric `k`.
//!
//! ### Options
//!
//! `options.non_empty`, `options.order`, `options.limit` are applied
//! post-assembly to `Series` and `Pivot` results. `Scalar` and `Rollup`
//! ignore them — a single cell has nothing to sort, and v0.1 rollup
//! ordering is out of scope.
//!
//! [`Results`]: tatami::Results
//! [`ResolvedQuery`]: crate::resolve::ResolvedQuery
//! [`ResolvedSet`]: crate::resolve::ResolvedSet

use std::cmp::Ordering;

use tatami::query::{Direction, MemberRef, OrderKey, Path, Tuple};
use tatami::schema::Name;
use tatami::schema::metric::Expr;
use tatami::{Cell, Results, crossed, missing, pivot, rollup, scalar, series};

use crate::eval::aggregate::evaluate_measure;
use crate::eval::domain::{self, Domain};
use crate::eval::metric::evaluate_expr;
use crate::eval::set;
use crate::resolve::{ResolvedAxes, ResolvedMember, ResolvedMetric, ResolvedQuery, ResolvedTuple};
use crate::{Error, InMemoryCube};

/// Evaluate a resolved query end-to-end, producing a typed [`Results`]
/// value.
///
/// The outer `Result` surfaces evaluation-time invariant violations
/// (unresolved names defensively threaded through metric eval, polars
/// runtime failures). Successful evaluation always returns a
/// shape-appropriate `Results` variant per §3.3.
pub(crate) fn evaluate<'s>(
    resolved: &ResolvedQuery<'s>,
    cube: &'s InMemoryCube,
) -> Result<Results, Error> {
    match &resolved.axes {
        ResolvedAxes::Scalar => evaluate_scalar(resolved, cube),
        ResolvedAxes::Series { rows } => evaluate_series(resolved, rows, cube),
        ResolvedAxes::Pivot { rows, columns } => evaluate_pivot(resolved, rows, columns, cube),
        ResolvedAxes::Pages {
            rows,
            columns,
            pages,
        } => evaluate_pages(resolved, rows, columns, pages, cube),
    }
}

/// `Axes::Scalar` — no axes. One cell per metric, evaluated at the slicer
/// tuple alone.
fn evaluate_scalar<'s>(
    resolved: &ResolvedQuery<'s>,
    cube: &'s InMemoryCube,
) -> Result<Results, Error> {
    let values = metrics_at(&resolved.metrics, &resolved.slicer, cube)?;
    let tuple = tuple_of(&resolved.slicer);
    Ok(Results::Scalar(scalar::Result::new(tuple, values)))
}

/// Build a shared domain from the resolved axis and one vector per metric.
fn evaluate_series<'s>(
    resolved: &ResolvedQuery<'s>,
    rows: &crate::resolve::ResolvedSet<'s>,
    cube: &'s InMemoryCube,
) -> Result<Results, Error> {
    let row_tuples = prune_axis_by_slicer(set::evaluate(rows, cube)?, &resolved.slicer);

    let domain = Domain::of(rows)?;
    // Assemble per-metric rows. For each metric, walk every row tuple
    // (intersected with the slicer) and evaluate. An empty intersection
    // (slicer pins a dim with a path incompatible with the row) yields
    // `Cell::Missing` rather than evaluating.
    let mut series_rows: Vec<series::Row> = Vec::with_capacity(resolved.metrics.len());
    for metric in &resolved.metrics {
        let label = metric.name().as_str().to_owned();
        let mut values: Vec<Cell> = Vec::with_capacity(row_tuples.len());
        for row_tuple in &row_tuples {
            let cell = match intersect(&resolved.slicer, row_tuple) {
                Some(merged) => evaluate_metric(metric, &merged, cube)?,
                None => no_facts_cell(),
            };
            values.push(cell);
        }
        series_rows.push(series::Row { label, values });
    }

    let order = apply_options_series(&row_tuples, &mut series_rows, resolved);
    match domain {
        Domain::Crossed(groups) => {
            let x = order
                .into_iter()
                .map(|i| domain::coordinate(&groups, &row_tuples[i]))
                .collect::<Result<_, _>>()?;
            Ok(Results::CrossedSeries(crossed::Result::new(
                groups,
                x,
                series_rows,
            )?))
        }
        Domain::Members(_) => {
            let x = order
                .into_iter()
                .map(|i| match row_tuples[i].members.as_slice() {
                    [member] => Ok(member_ref_of(member)),
                    _ => Err(Error::EvalSetCompositionIllFormed {
                        reason: "member series requires one member per position",
                    }),
                })
                .collect::<Result<_, _>>()?;
            Ok(Results::Series(series::Result::new(x, series_rows)))
        }
    }
}

/// `Axes::Pivot { rows, columns }` — rows × columns grid, one cell per
/// metric per (row, column). If `rows` is structurally a `Descendants`
/// set **and** its flat output shares a single top-level ancestor, we
/// return a [`Results::Rollup`] tree instead (§3.3 note).
///
/// Multi-root descendants — e.g. `Set::range(FY2025..FY2030).descendants_to(Quarter)`
/// — don't fit a single-rooted [`rollup::Tree`] and fall through to
/// [`Results::Pivot`]. Collapsing them into one synthetic root would
/// silently drop every non-first-root's subtree (MAP §8 R3 category);
/// a flat grid of (quarter × region) is the honest shape.
fn evaluate_pivot<'s>(
    resolved: &ResolvedQuery<'s>,
    rows: &crate::resolve::ResolvedSet<'s>,
    columns: &crate::resolve::ResolvedSet<'s>,
    cube: &'s InMemoryCube,
) -> Result<Results, Error> {
    if is_descendants(rows) {
        let row_tuples = prune_axis_by_slicer(set::evaluate(rows, cube)?, &resolved.slicer);
        if let Some(root) = single_root_member(&row_tuples) {
            return evaluate_rollup(resolved, row_tuples, root, cube);
        }
        // Fall through: multi-root `Descendants` produces a flat pivot
        // with one row per tuple, same as any other set on the rows axis.
        let col_tuples = prune_axis_by_slicer(set::evaluate(columns, cube)?, &resolved.slicer);
        return build_pivot(resolved, row_tuples, col_tuples, cube);
    }
    let row_tuples = prune_axis_by_slicer(set::evaluate(rows, cube)?, &resolved.slicer);
    let col_tuples = prune_axis_by_slicer(set::evaluate(columns, cube)?, &resolved.slicer);
    build_pivot(resolved, row_tuples, col_tuples, cube)
}

/// Whether a resolved set is structurally a top-level `Descendants`. The
/// §3.3 table triggers a `Rollup` return whenever the pivot's rows axis
/// descends a hierarchy — the tree shape is the natural output.
fn is_descendants(set: &crate::resolve::ResolvedSet<'_>) -> bool {
    matches!(set, crate::resolve::ResolvedSet::Descendants { .. })
}

/// If every tuple in `tuples` carries a single member that shares the same
/// `(dim, hierarchy, path[0])` head, synthesize a [`MemberRef`] at that
/// single-segment path and return it — the implicit root of the rollup
/// tree. Returns `None` when the tuples don't share a single top-level
/// ancestor (multi-root `Descendants` output, cross-joined rows, or a
/// degenerate empty set).
fn single_root_member(tuples: &[ResolvedTuple<'_>]) -> Option<MemberRef> {
    let first = tuples.first()?;
    let lead = first.members.first()?;
    let first_segment: &Name = lead.path.segments().next()?;
    let first_dim = &lead.dim.dim.name;
    let first_hierarchy = &lead.hierarchy.hierarchy.name;
    for tuple in tuples {
        // Rollup trees are single-dim; a cross-joined tuple with multiple
        // members doesn't fit the shape and falls through to Pivot.
        if tuple.members.len() != 1 {
            return None;
        }
        let member = &tuple.members[0];
        let segment = member.path.segments().next()?;
        if segment != first_segment
            || &member.dim.dim.name != first_dim
            || &member.hierarchy.hierarchy.name != first_hierarchy
        {
            return None;
        }
    }
    Some(MemberRef::new(
        first_dim.clone(),
        first_hierarchy.clone(),
        tatami::query::Path::of(first_segment.clone()),
    ))
}

/// `Axes::Pages { rows, columns, pages }` — v0.1 collapses pages onto the
/// columns axis by cross-joining every column tuple with every page
/// tuple. This keeps `Pages` as a strict widening of `Pivot`.
fn evaluate_pages<'s>(
    resolved: &ResolvedQuery<'s>,
    rows: &crate::resolve::ResolvedSet<'s>,
    columns: &crate::resolve::ResolvedSet<'s>,
    pages: &crate::resolve::ResolvedSet<'s>,
    cube: &'s InMemoryCube,
) -> Result<Results, Error> {
    let row_tuples = prune_axis_by_slicer(set::evaluate(rows, cube)?, &resolved.slicer);
    let col_tuples = prune_axis_by_slicer(set::evaluate(columns, cube)?, &resolved.slicer);
    let page_tuples = prune_axis_by_slicer(set::evaluate(pages, cube)?, &resolved.slicer);

    // Widen columns by the page axis: for each page, every column tuple
    // paired with it becomes a top-level column header. Order is
    // page-major to match common UX expectations (one page then the next).
    // A col × page that AUTOEXISTS-collapses (same dim, incompatible
    // paths) drops that widened header — build_pivot then never sees it.
    let mut widened: Vec<ResolvedTuple<'s>> =
        Vec::with_capacity(col_tuples.len().saturating_mul(page_tuples.len()));
    for page_tuple in &page_tuples {
        for col_tuple in &col_tuples {
            if let Some(merged) = intersect(col_tuple, page_tuple) {
                widened.push(merged);
            }
        }
    }
    build_pivot(resolved, row_tuples, widened, cube)
}

/// Apply slicer AUTOEXISTS to an axis before public headers are minted.
///
/// A slicer member from the same dimension/hierarchy narrows an axis to
/// compatible members. This is the enterprise dashboard default: drilling a
/// region and showing state-level members yields states in that region, not a
/// full country list with `Missing` cells outside the region. Sparse fact
/// rows on disjoint dimensions still flow through to `options.non_empty`.
fn prune_axis_by_slicer<'s>(
    tuples: Vec<ResolvedTuple<'s>>,
    slicer: &ResolvedTuple<'s>,
) -> Vec<ResolvedTuple<'s>> {
    if slicer.members.is_empty() {
        return tuples;
    }
    tuples
        .into_iter()
        .filter(|tuple| intersect(slicer, tuple).is_some())
        .collect()
}

/// Assemble a pivot result from already-evaluated row and column tuple
/// lists. Handles the multi-metric widening of `col_headers`.
fn build_pivot<'s>(
    resolved: &ResolvedQuery<'s>,
    row_tuples: Vec<ResolvedTuple<'s>>,
    col_tuples: Vec<ResolvedTuple<'s>>,
    cube: &'s InMemoryCube,
) -> Result<Results, Error> {
    let n_metrics = resolved.metrics.len();

    // For every row × column × metric, evaluate one cell. AUTOEXISTS:
    // if slicer × row collapses, the whole row is Missing; if slicer ×
    // row × col collapses at the column step, just that cell is Missing.
    let mut cells: Vec<Vec<Cell>> = Vec::with_capacity(row_tuples.len());
    for row_tuple in &row_tuples {
        let merged_row = intersect(&resolved.slicer, row_tuple);
        let mut row_cells: Vec<Cell> = Vec::with_capacity(col_tuples.len() * n_metrics);
        for col_tuple in &col_tuples {
            let merged = merged_row.as_ref().and_then(|mr| intersect(mr, col_tuple));
            for metric in &resolved.metrics {
                match &merged {
                    Some(m) => row_cells.push(evaluate_metric(metric, m, cube)?),
                    None => row_cells.push(no_facts_cell()),
                }
            }
        }
        cells.push(row_cells);
    }

    // Widen col_headers by metric count — each column tuple repeats once
    // per metric, keeping `cells[r][c * m + k]` indexable. Public
    // `Tuple`s; we drop the handles here.
    let row_headers: Vec<Tuple> = row_tuples.iter().map(tuple_of).collect();
    let mut col_headers: Vec<Tuple> = Vec::with_capacity(col_tuples.len() * n_metrics.max(1));
    for col_tuple in &col_tuples {
        let base = tuple_of(col_tuple);
        if n_metrics <= 1 {
            col_headers.push(base);
        } else {
            for _ in 0..n_metrics {
                col_headers.push(base.clone());
            }
        }
    }

    let (row_headers, cells) = apply_options_pivot(row_headers, cells, resolved);

    Ok(Results::Pivot(pivot::Result::new(
        row_headers,
        col_headers,
        cells,
    )))
}

/// `Axes::Pivot { rows: Descendants, … }` with a single-rooted row set →
/// [`Results::Rollup`]. The flat row tuples from set evaluation carry
/// [`tatami::query::Path`]s that encode the tree; we reassemble it by
/// shared path prefix rooted at the synthesized ancestor member.
///
/// The caller is responsible for guaranteeing the tuples share a single
/// top-level ancestor (see [`single_root_member`]); multi-root descendants
/// fall through to [`build_pivot`] instead so we don't collapse multiple
/// subtrees under one root (a silent-wrong-answer bug, MAP §8 R3).
///
/// Multi-metric rollup is v0.2; v0.1 uses the first metric for every
/// node's `value` cell. `columns` is ignored — a rollup tree has no
/// column axis by shape.
fn evaluate_rollup<'s>(
    resolved: &ResolvedQuery<'s>,
    row_tuples: Vec<ResolvedTuple<'s>>,
    root_ref: MemberRef,
    cube: &'s InMemoryCube,
) -> Result<Results, Error> {
    let primary = resolved
        .metrics
        .first()
        .ok_or(Error::EvalSetCompositionIllFormed {
            reason: "Rollup requires at least one metric",
        })?;

    // Evaluate the primary metric at the root itself. The descendants set
    // doesn't include its own source member, so we compute the root cell
    // explicitly — pinning the slicer × the synthesized root member.
    let root_member = synthesize_root_member(&row_tuples, &root_ref)?;
    let root_tuple = ResolvedTuple::from_members(vec![root_member]);
    let root_cell = match intersect(&resolved.slicer, &root_tuple) {
        Some(merged) => evaluate_metric(primary, &merged, cube)?,
        None => no_facts_cell(),
    };

    // Walk each flat tuple, evaluate the primary metric, collect
    // (member_ref, cell) pairs. These are the descendants that slot under
    // the synthesized root.
    let mut descendants: Vec<(MemberRef, Cell)> = Vec::with_capacity(row_tuples.len());
    for row_tuple in &row_tuples {
        let cell = match intersect(&resolved.slicer, row_tuple) {
            Some(merged) => evaluate_metric(primary, &merged, cube)?,
            None => no_facts_cell(),
        };
        let first_member = row_tuple
            .members
            .first()
            .ok_or(Error::EvalSetCompositionIllFormed {
                reason: "Rollup row tuple has no members",
            })?;
        descendants.push((member_ref_of(first_member), cell));
    }

    let tree = assemble_rollup(root_ref, root_cell, descendants);
    Ok(Results::Rollup(tree))
}

/// Rebuild a [`crate::resolve::ResolvedMember`] at the `root_ref`'s
/// single-segment path, borrowing dim/hierarchy handles from the first row
/// tuple's member. The caller guarantees `row_tuples` is non-empty and
/// its tuples share a single-dim, single-hierarchy head (enforced by
/// [`single_root_member`]).
fn synthesize_root_member<'s>(
    row_tuples: &[ResolvedTuple<'s>],
    root_ref: &MemberRef,
) -> Result<ResolvedMember<'s>, Error> {
    let lead = row_tuples.first().and_then(|t| t.members.first()).ok_or(
        Error::EvalSetCompositionIllFormed {
            reason: "Rollup row set is empty",
        },
    )?;
    Ok(ResolvedMember {
        dim: lead.dim,
        hierarchy: lead.hierarchy,
        path: root_ref.path.clone(),
    })
}

/// Assemble the rollup tree: start at the synthesized root with its
/// pre-computed cell, then slot every descendant under it by path prefix.
fn assemble_rollup(
    root_ref: MemberRef,
    root_cell: Cell,
    descendants: Vec<(MemberRef, Cell)>,
) -> rollup::Tree {
    let root_depth = root_ref.path.len();
    let mut root = rollup::Tree {
        root: root_ref,
        value: root_cell,
        children: Vec::new(),
    };

    // Since we can't keep simultaneous `&mut` pointers into the tree in
    // Rust without unsafe, the classic workaround is "insert by path": for
    // each incoming node, walk from the root down through `tree.children`
    // matching each prefix segment, creating missing intermediates.
    for (mref, cell) in descendants {
        insert_by_path(&mut root, root_depth, &mref, cell);
    }

    root
}

/// Insert `(mref, cell)` into `root`'s subtree, walking the prefix from
/// depth `root_depth + 1`. Creates intermediate placeholder nodes if the
/// pre-order traversal skipped levels (the catalogue always produces
/// full paths, so this is defensive).
fn insert_by_path(root: &mut rollup::Tree, root_depth: usize, mref: &MemberRef, cell: Cell) {
    let segments: Vec<&Name> = mref.path.segments().collect();
    let mut cursor = root;
    // Skip the root_depth segments (they lead to the root itself).
    for i in root_depth..segments.len() {
        let seg = segments[i];
        // Find or create a child matching this segment.
        let existing = cursor
            .children
            .iter()
            .position(|c| c.root.path.segments().nth(i) == Some(seg));
        match existing {
            Some(idx) => {
                cursor = &mut cursor.children[idx];
            }
            None => {
                // Build the intermediate's path by truncating the input
                // path to `i + 1` segments.
                let path_segments: Vec<Name> =
                    segments.iter().take(i + 1).copied().cloned().collect();
                let path =
                    tatami::query::Path::parse(path_segments).expect("non-empty truncated path");
                let partial_mref = MemberRef::new(mref.dim.clone(), mref.hierarchy.clone(), path);
                let is_terminal = i == segments.len() - 1;
                cursor.children.push(rollup::Tree {
                    root: partial_mref,
                    // Terminal → the incoming cell; intermediate → Missing,
                    // because we didn't receive a cell for this path.
                    value: if is_terminal {
                        cell.clone()
                    } else {
                        Cell::Missing {
                            reason: tatami::missing::Reason::NoFacts,
                        }
                    },
                    children: Vec::new(),
                });
                let last = cursor.children.len() - 1;
                cursor = &mut cursor.children[last];
            }
        }
    }
    // If we walked to an existing node (the exact path was already present
    // as an intermediate), overwrite its value with the real cell.
    cursor.value = cell;
}

/// Select and order positions once. The same indices determine every metric
/// vector and the public domain, so a discarded tuple is never materialized.
fn apply_options_series(
    tuples: &[ResolvedTuple<'_>],
    rows: &mut [series::Row],
    resolved: &ResolvedQuery<'_>,
) -> Vec<usize> {
    let opts = &resolved.options;
    let mut idx: Vec<usize> = (0..tuples.len()).collect();
    if opts.non_empty {
        idx.retain(|&i| {
            rows.iter()
                .any(|row| !matches!(row.values[i], Cell::Missing { .. }))
        });
    }
    let mut ranked = false;
    if let Some(rank) = &opts.rank {
        if let Some(mi) = metric_position(resolved, &rank.metric) {
            sort_indices_by_metric(&mut idx, &rows[mi].values, rank.direction);
        }
        if let Some(n) = opts.limit {
            idx.truncate(n.get().min(idx.len()));
            ranked = true;
        }
    }

    for ob in opts.order.iter().rev() {
        match &ob.key {
            OrderKey::Metric(name) => {
                if let Some(mi) = metric_position(resolved, name) {
                    sort_indices_by_metric(&mut idx, &rows[mi].values, ob.direction);
                }
            }
            OrderKey::Dimension(name) => {
                sort_indices_by_member(&mut idx, tuples, name, ob.direction);
            }
        }
    }

    if !ranked && let Some(n) = opts.limit {
        idx.truncate(n.get().min(idx.len()));
    }

    for row in rows {
        row.values = permute(std::mem::take(&mut row.values), &idx);
    }
    idx
}

/// Apply `options.non_empty`, `options.order`, `options.limit` to a pivot.
/// Sorting and limit act on rows. Non-empty drops rows where every cell is
/// `Missing`.
fn apply_options_pivot(
    mut row_headers: Vec<Tuple>,
    mut cells: Vec<Vec<Cell>>,
    resolved: &ResolvedQuery<'_>,
) -> (Vec<Tuple>, Vec<Vec<Cell>>) {
    let opts = &resolved.options;

    if opts.non_empty {
        let keep: Vec<bool> = cells
            .iter()
            .map(|row| row.iter().any(|c| !matches!(c, Cell::Missing { .. })))
            .collect();
        row_headers = filter_by_mask(row_headers, &keep);
        cells = filter_by_mask(cells, &keep);
    }

    // Position order over pivot rows: top-N rank + metric ordering + limit.
    // A metric slots in at `cells[row][metric_idx]` (column 0 of the
    // metric-widened row). Ordering a 2-D grid by a *dimension* (which row
    // level?) is a follow-up; dimension order keys are left in resolution
    // order here.
    let mut idx: Vec<usize> = (0..row_headers.len()).collect();

    let mut ranked = false;
    if let Some(rank) = &opts.rank {
        if let Some(mi) = metric_position(resolved, &rank.metric) {
            sort_pivot_rows_by_metric(&mut idx, &cells, mi, rank.direction);
        }
        if let Some(n) = opts.limit {
            idx.truncate(n.get().min(idx.len()));
            ranked = true;
        }
    }

    for ob in opts.order.iter().rev() {
        if let OrderKey::Metric(name) = &ob.key
            && let Some(mi) = metric_position(resolved, name)
        {
            sort_pivot_rows_by_metric(&mut idx, &cells, mi, ob.direction);
        }
    }

    if !ranked && let Some(n) = opts.limit {
        idx.truncate(n.get().min(idx.len()));
    }

    let row_headers = permute(row_headers, &idx);
    let cells = permute(cells, &idx);

    (row_headers, cells)
}

/// Score a cell for ordering. `Valid` returns its value; `Missing` /
/// `Error` score `f64::NEG_INFINITY` so they sink to the bottom on a
/// descending sort and to the top on ascending — a deterministic choice
/// documented here.
fn cell_score(cell: &Cell) -> f64 {
    match cell {
        Cell::Valid { value, .. } => *value,
        _ => f64::NEG_INFINITY,
    }
}

/// Position of the requested metric named `name` among the resolved metrics.
fn metric_position(resolved: &ResolvedQuery<'_>, name: &Name) -> Option<usize> {
    resolved.metrics.iter().position(|m| m.name() == name)
}

/// Compare two scores under `direction`. `Asc` is natural; `Desc` reverses.
fn direction_cmp(a: f64, b: f64, direction: Direction) -> Ordering {
    let base = a.partial_cmp(&b).unwrap_or(Ordering::Equal);
    match direction {
        Direction::Desc => base.reverse(),
        _ => base,
    }
}

/// Stable-sort `idx` by a series metric's values at each index, in `direction`.
fn sort_indices_by_metric(idx: &mut [usize], values: &[Cell], direction: Direction) {
    idx.sort_by(|&a, &b| direction_cmp(cell_score(&values[a]), cell_score(&values[b]), direction));
}

/// Stable-sort `idx` by the x-axis member's path key at each index, in
/// `direction` (lexical; ISO dates sort chronologically).
fn sort_indices_by_member(
    idx: &mut [usize],
    x: &[ResolvedTuple<'_>],
    dim: &Name,
    direction: Direction,
) {
    idx.sort_by(|&a, &b| {
        let path = |index: usize| match x[index].members.as_slice() {
            [only] => Some(only.path.to_string()),
            members => members
                .iter()
                .find(|member| &member.dim.dim.name == dim)
                .map(|member| member.path.to_string()),
        };
        let base = path(a).cmp(&path(b));
        match direction {
            Direction::Desc => base.reverse(),
            _ => base,
        }
    });
}

/// Stable-sort pivot row indices by a metric's value (column 0) at each row.
fn sort_pivot_rows_by_metric(
    idx: &mut [usize],
    cells: &[Vec<Cell>],
    metric_idx: usize,
    direction: Direction,
) {
    idx.sort_by(|&a, &b| {
        let va = cell_score(cells[a].get(metric_idx).unwrap_or(&no_facts_cell()));
        let vb = cell_score(cells[b].get(metric_idx).unwrap_or(&no_facts_cell()));
        direction_cmp(va, vb, direction)
    });
}

/// Keep entries of `v` whose mask is `true`; drop the rest.
fn filter_by_mask<T>(v: Vec<T>, mask: &[bool]) -> Vec<T> {
    v.into_iter()
        .zip(mask.iter().copied())
        .filter_map(|(item, keep)| if keep { Some(item) } else { None })
        .collect()
}

/// Reorder `v` by the given index permutation.
fn permute<T: Clone>(v: Vec<T>, indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| v[i].clone()).collect()
}

/// Evaluate every metric at a single tuple — the per-cell engine for the
/// `Scalar` variant and the inner loops of `Series` / `Pivot`.
fn metrics_at<'s>(
    metrics: &[ResolvedMetric<'s>],
    tuple: &ResolvedTuple<'s>,
    cube: &'s InMemoryCube,
) -> Result<Vec<Cell>, Error> {
    metrics
        .iter()
        .map(|m| evaluate_metric(m, tuple, cube))
        .collect()
}

/// Evaluate a single query-position metric at a tuple.
///
/// An aggregated measure is re-aggregated by the query's function — the
/// query is the sole authority for α, so the schema's declared rule is
/// overridden. A derived metric walks its stored expression tree (whose
/// nested measure refs keep their schema-declared aggregation).
fn evaluate_metric<'s>(
    metric: &ResolvedMetric<'s>,
    tuple: &ResolvedTuple<'s>,
    cube: &'s InMemoryCube,
) -> Result<Cell, Error> {
    match metric {
        ResolvedMetric::Aggregated { measure, agg } => evaluate_measure(
            &measure.with_aggregation(agg.clone()),
            tuple,
            cube.df(),
            &cube.schema,
        ),
        ResolvedMetric::Derived { metric } => evaluate_expr(
            &Expr::Ref {
                name: metric.name.clone(),
            },
            tuple,
            cube,
        ),
    }
}

/// Intersect two tuples by AUTOEXISTS (MDX, Microsoft Learn):
/// for each dim, if both sides carry a member their paths must be
/// prefix-compatible — one an ancestor of the other in the same
/// hierarchy — and the deeper (finer-grained) member survives.
/// Dims unique to one side pass through. Incompatible same-dim
/// members collapse the whole tuple to `None`, which the caller maps
/// to [`Cell::Missing`].
pub(crate) fn intersect<'s>(
    base: &ResolvedTuple<'s>,
    overlay: &ResolvedTuple<'s>,
) -> Option<ResolvedTuple<'s>> {
    let mut members: Vec<ResolvedMember<'s>> = Vec::with_capacity(base.members.len());
    for b in &base.members {
        match overlay
            .members
            .iter()
            .find(|o| o.dim.dim.name == b.dim.dim.name)
        {
            None => members.push(b.clone()),
            Some(o) => members.push(intersect_members(b, o)?),
        }
    }
    for o in &overlay.members {
        if !base
            .members
            .iter()
            .any(|b| b.dim.dim.name == o.dim.dim.name)
        {
            members.push(o.clone());
        }
    }
    Some(ResolvedTuple::from_members(members))
}

/// Path-prefix intersection of two same-dim members. Equal paths →
/// either side; one path a prefix of the other → the deeper side.
/// Otherwise `None`. Members on different hierarchies of the same dim
/// are also `None` for v0.1 (no attribute-relationship map yet).
fn intersect_members<'s>(
    a: &ResolvedMember<'s>,
    b: &ResolvedMember<'s>,
) -> Option<ResolvedMember<'s>> {
    if a.hierarchy.hierarchy.name != b.hierarchy.hierarchy.name {
        return None;
    }
    if path_starts_with(&a.path, &b.path) {
        Some(a.clone())
    } else if path_starts_with(&b.path, &a.path) {
        Some(b.clone())
    } else {
        None
    }
}

/// Whether `long` begins with every segment of `short`, in order.
/// `short.len() == long.len()` → equality check (prefix of self).
fn path_starts_with(long: &Path, short: &Path) -> bool {
    if short.len() > long.len() {
        return false;
    }
    long.segments().zip(short.segments()).all(|(l, s)| l == s)
}

/// Shorthand: the `NoFacts` Missing cell used whenever an AUTOEXISTS
/// intersection collapses, or a metric evaluator runs out of facts.
fn no_facts_cell() -> Cell {
    Cell::Missing {
        reason: missing::Reason::NoFacts,
    }
}

/// Project a [`ResolvedTuple`] to a public [`Tuple`] for embedding in
/// result shapes.
fn tuple_of(resolved: &ResolvedTuple<'_>) -> Tuple {
    let members: Vec<MemberRef> = resolved.members.iter().map(member_ref_of).collect();
    // Resolve-time dim-distinctness guarantees `Tuple::of` succeeds; the
    // defensive fallback to `Tuple::empty` keeps the function total.
    Tuple::of(members).unwrap_or_else(|_| Tuple::empty())
}

/// Project a [`ResolvedMember`] to a public [`MemberRef`].
pub(super) fn member_ref_of(member: &ResolvedMember<'_>) -> MemberRef {
    MemberRef::new(
        member.dim.dim.name.clone(),
        member.hierarchy.hierarchy.name.clone(),
        member.path.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars_core::df;
    use polars_core::prelude::DataFrame;
    use std::num::NonZeroUsize;
    use tatami::query::{
        Axes, Direction, MemberRef as MRef, OrderBy, OrderKey, Path, Predicate, Set, Tuple,
    };
    use tatami::schema::metric::Expr;
    use tatami::schema::{
        Aggregation, Calendar, Dimension, Hierarchy, Level, Measure, Metric, Schema,
    };
    use tatami::{Cube, Query};

    use crate::InMemoryCube;

    fn n(s: &str) -> Name {
        Name::parse(s).expect("valid name")
    }

    fn mr(dim: &str, hier: &str, segs: &[&str]) -> MRef {
        let names: Vec<Name> = segs.iter().copied().map(n).collect();
        MRef::new(n(dim), n(hier), Path::parse(names).expect("non-empty"))
    }

    /// Two-hierarchy cube: Geography (Region → Country) + Scenario + amount.
    fn fixture_cube() -> InMemoryCube {
        let schema =
            Schema::builder()
                .dimension(
                    Dimension::regular(n("Geography")).hierarchy(
                        Hierarchy::new(n("Default"))
                            .level(Level::new(n("Region"), n("region")))
                            .level(Level::new(n("Country"), n("country"))),
                    ),
                )
                .dimension(Dimension::scenario(n("Scenario")).hierarchy(
                    Hierarchy::new(n("Default")).level(Level::new(n("Plan"), n("plan"))),
                ))
                .measure(Measure::new(n("amount"), Aggregation::sum()))
                .measure(Measure::new(n("units"), Aggregation::sum()))
                .build()
                .expect("schema");
        let df: DataFrame = df! {
            "region"  => ["EMEA", "EMEA", "APAC", "APAC"],
            "country" => ["UK",   "FR",   "JP",   "SG"],
            "plan"    => ["Actual", "Actual", "Actual", "Actual"],
            "amount"  => [100.0_f64, 200.0, 300.0, 400.0],
            "units"   => [1.0_f64, 2.0, 3.0, 4.0],
        }
        .expect("frame");
        InMemoryCube::new(df, schema).expect("cube")
    }

    /// Build a runtime and block on `cube.query(&q)`.
    fn run_query(cube: &InMemoryCube, q: Query) -> Results {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        rt.block_on(cube.query(&q)).expect("query ok")
    }

    #[test]
    fn scalar_query_returns_scalar_result_with_one_cell_per_metric() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Scalar,
            slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
            metrics: vec![
                tatami::query::Metric::aggregated(n("amount"), tatami::schema::Aggregation::sum()),
                tatami::query::Metric::aggregated(n("units"), tatami::schema::Aggregation::sum()),
            ],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Scalar(s) => {
                assert_eq!(s.values().len(), 2, "one cell per metric");
                match &s.values()[0] {
                    Cell::Valid { value, .. } => assert_eq!(*value, 1000.0),
                    other => panic!("amount: expected Valid, got {other:?}"),
                }
                match &s.values()[1] {
                    Cell::Valid { value, .. } => assert_eq!(*value, 10.0),
                    other => panic!("units: expected Valid, got {other:?}"),
                }
            }
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    /// The per-query aggregation overrides the schema's declared rule.
    /// `amount` is declared `Sum`, yet a query asking for `Avg`/`Min`/`Max`/
    /// `Count`/`DistinctCount` gets that function — the function lives in the
    /// query (α authority), not the schema.
    #[test]
    fn per_query_aggregation_overrides_schema_default() {
        let cube = fixture_cube();
        // amount over the four Actual rows: [100, 200, 300, 400].
        let scalar = |agg: Aggregation| -> f64 {
            let q = Query {
                axes: Axes::Scalar,
                slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
                metrics: vec![tatami::query::Metric::aggregated(n("amount"), agg)],
                options: tatami::query::Options::default(),
            };
            match run_query(&cube, q) {
                Results::Scalar(s) => match &s.values()[0] {
                    Cell::Valid { value, .. } => *value,
                    other => panic!("expected Valid, got {other:?}"),
                },
                other => panic!("expected Scalar, got {other:?}"),
            }
        };
        assert_eq!(scalar(Aggregation::sum()), 1000.0);
        assert_eq!(scalar(Aggregation::avg()), 250.0);
        assert_eq!(scalar(Aggregation::min()), 100.0);
        assert_eq!(scalar(Aggregation::max()), 400.0);
        assert_eq!(scalar(Aggregation::count()), 4.0);
        assert_eq!(scalar(Aggregation::distinct_count()), 4.0);
    }

    /// Two functions over the *same* column coexist in one query — the
    /// function is per-metric, not a query-global mode.
    #[test]
    fn same_column_two_functions_in_one_query() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Scalar,
            slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
            metrics: vec![
                tatami::query::Metric::aggregated(n("amount"), Aggregation::min()),
                tatami::query::Metric::aggregated(n("amount"), Aggregation::max()),
            ],
            options: tatami::query::Options::default(),
        };
        match run_query(&cube, q) {
            Results::Scalar(s) => {
                assert_eq!(s.values().len(), 2, "one cell per metric");
                assert!(matches!(&s.values()[0], Cell::Valid { value, .. } if *value == 100.0));
                assert!(matches!(&s.values()[1], Cell::Valid { value, .. } if *value == 400.0));
            }
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    /// A `Derived` reference to a *stored measure* (not a formula) evaluates
    /// it by its schema-declared aggregation — the "use the schema's
    /// definition" path a name-only caller (e.g. `CUBEVALUE`) relies on.
    #[test]
    fn derived_reference_to_measure_uses_declared_aggregation() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Scalar,
            slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
            metrics: vec![tatami::query::Metric::derived(n("amount"))],
            options: tatami::query::Options::default(),
        };
        match run_query(&cube, q) {
            // amount is declared Sum → 100 + 200 + 300 + 400 = 1000.
            Results::Scalar(s) => {
                assert!(matches!(&s.values()[0], Cell::Valid { value, .. } if *value == 1000.0))
            }
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn series_query_returns_series_result() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Series {
                rows: Set::members(n("Geography"), n("Default"), n("Region")),
            },
            slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Series(s) => {
                assert_eq!(s.x().len(), 2, "two regions");
                assert_eq!(s.rows().len(), 1, "one metric");
                assert_eq!(s.rows()[0].values.len(), 2, "one value per x entry");
            }
            other => panic!("expected Series, got {other:?}"),
        }
    }

    #[test]
    fn pivot_query_returns_pivot_result() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Pivot {
                rows: Set::members(n("Geography"), n("Default"), n("Region")),
                columns: Set::members(n("Scenario"), n("Default"), n("Plan")),
            },
            slicer: Tuple::empty(),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Pivot(p) => {
                assert_eq!(p.row_headers().len(), 2);
                assert_eq!(p.col_headers().len(), 1);
                assert_eq!(p.cells().len(), 2);
                assert_eq!(p.cells()[0].len(), 1);
            }
            other => panic!("expected Pivot, got {other:?}"),
        }
    }

    #[test]
    fn pivot_with_descendants_of_single_member_returns_rollup() {
        // 3-level geography so Descendants(World, to=Country) spans
        // Region + Country — enough depth for a real tree. Region-level
        // source with to_level=Country would only produce the leaves, no
        // intermediate structure.
        let schema = Schema::builder()
            .dimension(
                Dimension::regular(n("Geography")).hierarchy(
                    Hierarchy::new(n("Default"))
                        .level(Level::new(n("World"), n("world")))
                        .level(Level::new(n("Region"), n("region")))
                        .level(Level::new(n("Country"), n("country"))),
                ),
            )
            .measure(Measure::new(n("amount"), Aggregation::sum()))
            .build()
            .expect("schema");
        let df: DataFrame = df! {
            "world"   => ["World", "World", "World"],
            "region"  => ["EMEA",  "EMEA",  "APAC"],
            "country" => ["UK",    "FR",    "JP"],
            "amount"  => [100.0_f64, 200.0, 300.0],
        }
        .expect("frame");
        let cube = InMemoryCube::new(df, schema).expect("cube");
        // §3.3 rule: Pivot with `rows = Descendants` of a single-rooted set
        // → Results::Rollup. The columns axis is ignored by the rollup
        // branch.
        let q = Query {
            axes: Axes::Pivot {
                rows: Set::explicit(vec![mr("Geography", "Default", &["World"])])
                    .expect("non-empty")
                    .descendants_to(n("Country")),
                columns: Set::members(n("Geography"), n("Default"), n("World")),
            },
            slicer: Tuple::empty(),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Rollup(tree) => {
                // Root = the synthesized World member at depth 1; two
                // regions sit as direct children (APAC, EMEA — sorted
                // pre-order by the catalogue), each with its country
                // children.
                assert_eq!(
                    tree.root.path.segments().collect::<Vec<_>>().len(),
                    1,
                    "root is the synthesized ancestor, not a descendant leaf"
                );
                assert_eq!(
                    tree.children.len(),
                    2,
                    "two regions hang off the World root, got {tree:?}"
                );
            }
            other => panic!("expected Rollup, got {other:?}"),
        }
    }

    #[test]
    fn pivot_with_descendants_of_range_returns_pivot_not_rollup() {
        // Multi-root `Set::range(Y1..Y2).descendants_to(Quarter)`: the
        // flat output has multiple top-level path[0]s (one per year) and
        // therefore can't fit a single-rooted `rollup::Tree`. Expect a
        // flat Pivot with one row per (year, quarter) tuple. Today's
        // single-rooted assembler would collapse every non-first-year
        // quarter under the first year's subtree — the silent-wrong-answer
        // shape (MAP §8 R3).
        let schema = Schema::builder()
            .dimension(
                Dimension::time(
                    n("Time"),
                    vec![tatami::schema::Calendar::gregorian(n("Gregorian"))],
                )
                .hierarchy(
                    Hierarchy::new(n("Fiscal"))
                        .level(Level::new(n("Year"), n("year")))
                        .level(Level::new(n("Quarter"), n("quarter"))),
                ),
            )
            .measure(Measure::new(n("amount"), Aggregation::sum()))
            .build()
            .expect("schema");
        let df: DataFrame = df! {
            "year"    => ["FY2025", "FY2025", "FY2026", "FY2026"],
            "quarter" => ["Q1",     "Q2",     "Q1",     "Q2"],
            "amount"  => [10.0_f64, 20.0,     30.0,     40.0],
        }
        .expect("frame");
        let cube = InMemoryCube::new(df, schema).expect("cube");
        let q = Query {
            axes: Axes::Pivot {
                rows: Set::range(
                    n("Time"),
                    n("Fiscal"),
                    mr("Time", "Fiscal", &["FY2025"]),
                    mr("Time", "Fiscal", &["FY2026"]),
                )
                .descendants_to(n("Quarter")),
                columns: Set::members(n("Time"), n("Fiscal"), n("Year")),
            },
            slicer: Tuple::empty(),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Pivot(p) => {
                // Two years × two quarters each = four row tuples.
                assert_eq!(
                    p.row_headers().len(),
                    4,
                    "multi-root descendants produce one row per (year, quarter)"
                );
            }
            other => panic!(
                "expected Pivot for multi-root Descendants, got {other:?} (rollup would collapse cross-year quarters)"
            ),
        }
    }

    #[test]
    fn pivot_with_multiple_metrics_widens_col_headers() {
        let cube = fixture_cube();
        // Two metrics + one column tuple → `col_headers.len() == 2` and
        // each row has two cells wide. Scenario on columns (instead of
        // Geography again) to keep the rows/cols dims distinct.
        let q = Query {
            axes: Axes::Pivot {
                rows: Set::members(n("Geography"), n("Default"), n("Region")),
                columns: Set::members(n("Scenario"), n("Default"), n("Plan")),
            },
            slicer: Tuple::empty(),
            metrics: vec![
                tatami::query::Metric::aggregated(n("amount"), tatami::schema::Aggregation::sum()),
                tatami::query::Metric::aggregated(n("units"), tatami::schema::Aggregation::sum()),
            ],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Pivot(p) => {
                // 1 column × 2 metrics = 2 col_headers; cells[r] width matches.
                assert_eq!(p.col_headers().len(), 2, "1 col × 2 metrics");
                assert_eq!(p.cells()[0].len(), 2);
            }
            other => panic!("expected Pivot, got {other:?}"),
        }
    }

    /// AUTOEXISTS (MDX): Region on rows × Country on columns — same
    /// Geography dim at two levels. Compatible cells (country in region)
    /// carry the country's value; incompatible cells (country not in
    /// region) must be `Cell::Missing`, not silently repeat a value.
    #[test]
    fn autoexists_same_dim_pivot_returns_missing_for_incompatible_cells() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Pivot {
                rows: Set::members(n("Geography"), n("Default"), n("Region")),
                columns: Set::members(n("Geography"), n("Default"), n("Country")),
            },
            slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Pivot(p) => {
                let rows: Vec<&str> = p
                    .row_headers()
                    .iter()
                    .map(|h| {
                        h.members()[0]
                            .path
                            .segments()
                            .last()
                            .expect("non-empty")
                            .as_str()
                    })
                    .collect();
                let cols: Vec<&str> = p
                    .col_headers()
                    .iter()
                    .map(|h| {
                        h.members()[0]
                            .path
                            .segments()
                            .last()
                            .expect("non-empty")
                            .as_str()
                    })
                    .collect();

                let cell_at = |region: &str, country: &str| -> &Cell {
                    let r = rows.iter().position(|s| *s == region).expect("region row");
                    let c = cols
                        .iter()
                        .position(|s| *s == country)
                        .expect("country col");
                    &p.cells()[r][c]
                };

                // EMEA × UK/FR are compatible; EMEA × JP/SG are not.
                match cell_at("EMEA", "UK") {
                    Cell::Valid { value, .. } => assert_eq!(*value, 100.0),
                    other => panic!("EMEA×UK expected Valid, got {other:?}"),
                }
                match cell_at("EMEA", "FR") {
                    Cell::Valid { value, .. } => assert_eq!(*value, 200.0),
                    other => panic!("EMEA×FR expected Valid, got {other:?}"),
                }
                assert!(
                    matches!(cell_at("EMEA", "JP"), Cell::Missing { .. }),
                    "EMEA×JP must be Missing, got {:?}",
                    cell_at("EMEA", "JP")
                );
                assert!(
                    matches!(cell_at("EMEA", "SG"), Cell::Missing { .. }),
                    "EMEA×SG must be Missing, got {:?}",
                    cell_at("EMEA", "SG")
                );

                // APAC × JP/SG compatible; APAC × UK/FR not.
                match cell_at("APAC", "JP") {
                    Cell::Valid { value, .. } => assert_eq!(*value, 300.0),
                    other => panic!("APAC×JP expected Valid, got {other:?}"),
                }
                match cell_at("APAC", "SG") {
                    Cell::Valid { value, .. } => assert_eq!(*value, 400.0),
                    other => panic!("APAC×SG expected Valid, got {other:?}"),
                }
                assert!(matches!(cell_at("APAC", "UK"), Cell::Missing { .. }));
                assert!(matches!(cell_at("APAC", "FR"), Cell::Missing { .. }));
            }
            other => panic!("expected Pivot, got {other:?}"),
        }
    }

    /// Slicer pins a region; rows list countries. Countries under the
    /// pinned region survive (deeper path wins the intersection);
    /// countries under a different region are pruned before public
    /// series headers are minted.
    #[test]
    fn autoexists_slicer_and_axis_same_dim_narrow_to_compatible_rows() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Series {
                rows: Set::members(n("Geography"), n("Default"), n("Country")),
            },
            slicer: Tuple::of([
                mr("Scenario", "Default", &["Actual"]),
                mr("Geography", "Default", &["EMEA"]),
            ])
            .expect("disjoint dims"),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Series(s) => {
                let leaves: Vec<&str> = s
                    .x()
                    .iter()
                    .map(|m| m.path.segments().last().expect("non-empty").as_str())
                    .collect();
                assert_eq!(
                    leaves,
                    vec!["FR", "UK"],
                    "axis narrows to the pinned region"
                );
                let values = &s.rows()[0].values;
                let by = |leaf: &str| -> &Cell {
                    let idx = leaves.iter().position(|s| *s == leaf).expect("leaf");
                    &values[idx]
                };
                match by("UK") {
                    Cell::Valid { value, .. } => assert_eq!(*value, 100.0),
                    other => panic!("UK expected Valid, got {other:?}"),
                }
                match by("FR") {
                    Cell::Valid { value, .. } => assert_eq!(*value, 200.0),
                    other => panic!("FR expected Valid, got {other:?}"),
                }
            }
            other => panic!("expected Series, got {other:?}"),
        }
    }

    #[test]
    fn non_empty_drops_rows_where_every_cell_is_missing() {
        // Fixture shaped so EMEA has no matching fact row for the slicer's
        // Time coordinate — its cells will be Missing. APAC has a matching
        // row and survives. `non_empty = true` must drop EMEA.
        let schema2 = Schema::builder()
            .dimension(
                Dimension::regular(n("Geography")).hierarchy(
                    Hierarchy::new(n("Default"))
                        .level(Level::new(n("Region"), n("region")))
                        .level(Level::new(n("Country"), n("country"))),
                ),
            )
            .dimension(
                Dimension::time(n("Time"), vec![Calendar::gregorian(n("Gregorian"))]).hierarchy(
                    Hierarchy::new(n("Default")).level(Level::new(n("Month"), n("month"))),
                ),
            )
            .measure(Measure::new(n("amount"), Aggregation::sum()))
            .build()
            .expect("schema");
        // APAC has a Jan fact; EMEA has a Feb fact; slicer pins Jan →
        // EMEA row all-Missing, APAC row Valid.
        let df2: DataFrame = df! {
            "region"  => ["EMEA", "APAC"],
            "country" => ["UK",   "JP"],
            "month"   => ["2026-02", "2026-01"],
            "amount"  => [10.0_f64, 20.0],
        }
        .expect("frame");
        let cube2 = InMemoryCube::new(df2, schema2).expect("cube");
        let q = Query {
            axes: Axes::Series {
                rows: Set::members(n("Geography"), n("Default"), n("Region")),
            },
            slicer: Tuple::single(mr("Time", "Default", &["2026-01"])),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options {
                non_empty: true,
                ..tatami::query::Options::default()
            },
        };
        let r = run_query(&cube2, q);
        match r {
            Results::Series(s) => {
                // Only APAC survives — EMEA's cell was Missing (no
                // Jan row).
                assert_eq!(s.x().len(), 1, "EMEA dropped by non_empty");
            }
            other => panic!("expected Series, got {other:?}"),
        }
    }

    #[test]
    fn order_by_metric_desc_sorts_rows() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Series {
                rows: Set::members(n("Geography"), n("Default"), n("Country")),
            },
            slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options {
                order: vec![OrderBy {
                    key: OrderKey::Metric(n("amount")),
                    direction: Direction::Desc,
                }],
                ..tatami::query::Options::default()
            },
        };
        let r = run_query(&cube, q);
        match r {
            Results::Series(s) => {
                // Country-level amounts: SG=400, JP=300, FR=200, UK=100.
                // Desc order = SG, JP, FR, UK.
                let leaves: Vec<&str> = s
                    .x()
                    .iter()
                    .map(|m| m.path.segments().last().expect("non-empty").as_str())
                    .collect();
                assert_eq!(leaves, vec!["SG", "JP", "FR", "UK"]);
            }
            other => panic!("expected Series, got {other:?}"),
        }
    }

    #[test]
    fn limit_truncates_rows_post_order() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Series {
                rows: Set::members(n("Geography"), n("Default"), n("Country")),
            },
            slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options {
                order: vec![OrderBy {
                    key: OrderKey::Metric(n("amount")),
                    direction: Direction::Desc,
                }],
                limit: Some(NonZeroUsize::new(2).expect("nonzero")),
                ..tatami::query::Options::default()
            },
        };
        let r = run_query(&cube, q);
        match r {
            Results::Series(s) => {
                assert_eq!(s.x().len(), 2);
                let leaves: Vec<&str> = s
                    .x()
                    .iter()
                    .map(|m| m.path.segments().last().expect("non-empty").as_str())
                    .collect();
                assert_eq!(leaves, vec!["SG", "JP"], "top-2 by amount desc");
            }
            other => panic!("expected Series, got {other:?}"),
        }
    }

    /// Rank is independent of display order: keep the top 2 by `amount`
    /// (SG=400, JP=300), but display them *ascending* by amount → JP, SG.
    #[test]
    fn rank_keeps_top_n_then_displays_in_chosen_order() {
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Series {
                rows: Set::members(n("Geography"), n("Default"), n("Country")),
            },
            slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options {
                rank: Some(tatami::query::Rank {
                    metric: n("amount"),
                    direction: Direction::Desc,
                }),
                limit: Some(NonZeroUsize::new(2).expect("nonzero")),
                order: vec![OrderBy {
                    key: OrderKey::Metric(n("amount")),
                    direction: Direction::Asc,
                }],
                ..tatami::query::Options::default()
            },
        };
        let r = run_query(&cube, q);
        match r {
            Results::Series(s) => {
                let leaves: Vec<&str> = s
                    .x()
                    .iter()
                    .map(|m| m.path.segments().last().expect("non-empty").as_str())
                    .collect();
                assert_eq!(
                    leaves,
                    vec!["JP", "SG"],
                    "top-2 by amount, shown amount-ascending"
                );
            }
            other => panic!("expected Series, got {other:?}"),
        }
    }

    /// Ordering by a dimension key reverses cleanly between asc and desc
    /// (a time axis sorts chronologically / reverse-chronologically).
    #[test]
    fn order_by_dimension_reverses_with_direction() {
        let cube = fixture_cube();
        let leaves = |direction| {
            let q = Query {
                axes: Axes::Series {
                    rows: Set::members(n("Geography"), n("Default"), n("Country")),
                },
                slicer: Tuple::single(mr("Scenario", "Default", &["Actual"])),
                metrics: vec![tatami::query::Metric::aggregated(
                    n("amount"),
                    tatami::schema::Aggregation::sum(),
                )],
                options: tatami::query::Options {
                    order: vec![OrderBy {
                        key: OrderKey::Dimension(n("Geography")),
                        direction,
                    }],
                    ..tatami::query::Options::default()
                },
            };
            match run_query(&cube, q) {
                Results::Series(s) => s.x().iter().map(|m| m.path.to_string()).collect::<Vec<_>>(),
                other => panic!("expected Series, got {other:?}"),
            }
        };
        let asc = leaves(Direction::Asc);
        let mut desc = leaves(Direction::Desc);
        desc.reverse();
        assert_eq!(asc, desc, "asc and desc are mirror orders");
        assert!(
            asc.windows(2).all(|w| w[0] <= w[1]),
            "asc is sorted by member key"
        );
    }

    /// Minimal hewton-shaped schema and one fact row, so end-to-end
    /// `Cube::query(Scalar)` against a Revenue metric (= `Expr::Ref` to
    /// the `amount` measure) returns a non-zero value.
    #[test]
    fn hewton_shaped_scalar_query_returns_non_zero_revenue() {
        let schema = Schema::builder()
            .dimension(Dimension::regular(n("Geography")).hierarchy(
                Hierarchy::new(n("Default")).level(Level::new(n("Region"), n("region"))),
            ))
            .dimension(
                Dimension::scenario(n("Scenario")).hierarchy(
                    Hierarchy::new(n("Default")).level(Level::new(n("Plan"), n("plan"))),
                ),
            )
            .dimension(
                Dimension::time(n("Time"), vec![Calendar::gregorian(n("Gregorian"))]).hierarchy(
                    Hierarchy::new(n("Fiscal")).level(Level::new(n("FiscalYear"), n("fy"))),
                ),
            )
            .measure(Measure::new(n("amount"), Aggregation::sum()))
            .metric(Metric::new(n("Revenue"), Expr::Ref { name: n("amount") }))
            .build()
            .expect("schema");
        let df: DataFrame = df! {
            "region" => ["NA"],
            "plan"   => ["Actual"],
            "fy"     => ["FY2026"],
            "amount" => [1_000_000.0_f64],
        }
        .expect("frame");
        let cube = InMemoryCube::new(df, schema).expect("cube");
        let q = Query {
            axes: Axes::Scalar,
            slicer: Tuple::of([
                mr("Time", "Fiscal", &["FY2026"]),
                MRef::scenario(n("Actual")),
            ])
            .expect("disjoint"),
            metrics: vec![tatami::query::Metric::derived(n("Revenue"))],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Scalar(s) => {
                assert_eq!(s.values().len(), 1);
                match &s.values()[0] {
                    Cell::Valid { value, .. } => {
                        assert!(*value > 0.0, "hewton revenue must be positive");
                        assert_eq!(*value, 1_000_000.0);
                    }
                    other => panic!("expected Valid, got {other:?}"),
                }
            }
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn filter_with_in_predicate_keeps_matching_path_prefix() {
        // Use Filter on a Country-level set with In on Geography prefix
        // = [EMEA]. EMEA's countries (FR, UK) survive; APAC's don't.
        let cube = fixture_cube();
        let q = Query {
            axes: Axes::Series {
                rows: Set::members(n("Geography"), n("Default"), n("Country")).filter(
                    Predicate::In {
                        dim: n("Geography"),
                        path_prefix: Path::of(n("EMEA")),
                    },
                ),
            },
            slicer: Tuple::empty(),
            metrics: vec![tatami::query::Metric::aggregated(
                n("amount"),
                tatami::schema::Aggregation::sum(),
            )],
            options: tatami::query::Options::default(),
        };
        let r = run_query(&cube, q);
        match r {
            Results::Series(s) => {
                let leaves: Vec<&str> = s
                    .x()
                    .iter()
                    .map(|m| m.path.segments().last().expect("non-empty").as_str())
                    .collect();
                assert_eq!(leaves, vec!["FR", "UK"]);
            }
            other => panic!("expected Series, got {other:?}"),
        }
    }
}
