//! Six unchanged example rows from Meridian's monthly_close.csv. Each row
//! aggregates a monthly close; treaty_count is a measure, never a treaty ID.
use std::num::{NonZeroU16, NonZeroUsize};
use std::path::PathBuf;

use meridianre_serve::cube;
use polars_core::prelude::BooleanChunked;
use tatami::{Cube, drill, query, schema};
use tatami_inmem::drill::View;

fn name(s: &str) -> schema::Name {
    schema::Name::parse(s).expect("valid fixture name")
}
fn member(dim: &str, hierarchy: &str, path: &[&str]) -> tatami::MemberRef {
    tatami::MemberRef::new(
        name(dim),
        name(hierarchy),
        tatami::Path::parse(path.iter().map(|s| name(s)).collect()).expect("path"),
    )
}
fn snapshot(scope: &str) -> drill::Snapshot {
    drill::Snapshot {
        source: name("Meridian monthly close"),
        revision: name("close-v1"),
        visibility: name(scope),
    }
}
fn grain() -> drill::Grain {
    drill::Grain {
        name: name("Monthly close"),
        keys: ["year", "month", "country_code", "product_code", "scenario"]
            .into_iter()
            .map(name)
            .collect(),
    }
}
fn view(scope: &str, allowed: &[bool]) -> View {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/monthly_close.csv");
    let (frame, schema) =
        cube::facts(&path).expect("load actual Meridian schema and monthly facts");
    let mask: BooleanChunked = allowed.iter().copied().collect();
    View::new(frame, schema, &mask, snapshot(scope), grain()).expect("authorized cube")
}
fn reference(snapshot: drill::Snapshot) -> drill::Reference {
    let conditions = [
        member("Geography", "World", &["LAC", "AR"]),
        member("Time", "Calendar", &["2023"]),
        member("Scenario", "Plan", &["actual"]),
    ];
    drill::Reference {
        snapshot,
        query: tatami::Query {
            axes: tatami::Axes::Scalar,
            slicer: tatami::Tuple::of(conditions).expect("slicer"),
            metrics: vec![query::Metric::derived(name("CombinedRatio"))],
            options: query::Options::default(),
        },
        selection: drill::Selection::Scalar,
        metric: 0,
    }
}
fn value(cell: &tatami::Cell) -> f64 {
    match cell {
        tatami::Cell::Valid { value, .. } => *value,
        other => panic!("expected valid cell, got {other:?}"),
    }
}
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
}

#[tokio::test]
async fn combined_ratio_reconciles_to_paged_monthly_facts_before_and_after_visibility_changes() {
    let finance = view("finance-v3", &[true; 6]);
    let regional = view("agri-and-ah-v2", &[true, true, false, false, false, true]);
    let limit = NonZeroU16::new(2).expect("page size");
    for (cube, expected_count, losses, expenses, premium) in [
        (&finance, 3, 42517.78, 49654.87, 107851.11),
        (&regional, 2, 10400.77, 24995.49, 51747.32),
    ] {
        let reference = reference(cube.snapshot().clone());
        let tatami::Results::Scalar(rendered) =
            cube.query(&reference.query).await.expect("scoped KPI")
        else {
            panic!("scalar")
        };
        let selected = cube.select(&reference).expect("metric lineage");
        let first = selected.page(None, limit).expect("page");
        assert_eq!(first.grain, grain());
        assert_eq!(first.total, expected_count);
        assert_eq!(first.value, rendered.values()[0]);
        close(value(&first.value), (losses + expenses) / premium);
        close(value(&first.totals[&name("net_losses")]), losses);
        close(value(&first.totals[&name("total_expenses")]), expenses);
        close(value(&first.totals[&name("nep")]), premium);
        let counted = first
            .records
            .iter()
            .find(|fact| fact.key[3] == "AH")
            .expect("AH fact");
        assert_eq!(counted.values[&name("net_losses")], None);
        let mut records = first.records;
        if let Some(cursor) = first.next {
            let second = selected.page(Some(&cursor), limit).expect("next page");
            assert!(second.next.is_none());
            records.extend(second.records);
        }
        assert_eq!(records.len(), expected_count);
        assert!(records.windows(2).all(|pair| pair[0].key < pair[1].key));
        assert!(
            records
                .iter()
                .all(|f| f.key[0] == "2023" && f.key[2] == "AR")
        );
        let sum = |measure: &str| {
            records
                .iter()
                .filter_map(|f| f.values[&name(measure)])
                .sum::<f64>()
        };
        close(
            (sum("net_losses") + sum("total_expenses")) / sum("nep"),
            value(&first.value),
        );
    }

    let old = reference(finance.snapshot().clone());
    assert!(matches!(
        regional.select(&old),
        Err(tatami_inmem::drill::Error::Stale)
    ));
    let mut changed = old.clone();
    changed.snapshot.revision = name("close-v2");
    assert!(matches!(
        finance.select(&changed),
        Err(tatami_inmem::drill::Error::Stale)
    ));
    let cursor = finance
        .select(&old)
        .expect("original")
        .page(None, limit)
        .expect("first")
        .next
        .expect("cursor");
    assert!(matches!(
        regional
            .select(&reference(regional.snapshot().clone()))
            .expect("regional")
            .page(Some(&cursor), limit),
        Err(tatami_inmem::drill::Error::InvalidCursor)
    ));
}

#[tokio::test]
async fn series_and_pivot_only_accept_visible_result_coordinates() {
    let cube = view("finance", &[true; 6]);
    let mut r = reference(cube.snapshot().clone());
    let products =
        tatami::Set::members(name("Product"), name("LineOfBusiness"), name("ProductLine"));
    let prod = |label: &str| member("Product", "LineOfBusiness", &[label]);
    r.query.axes = tatami::Axes::Series {
        rows: products.clone(),
    };
    r.query.options.limit = NonZeroUsize::new(1);
    let tatami::Results::Series(s) = cube.query(&r.query).await.expect("series") else {
        panic!("series")
    };
    r.selection = drill::Selection::Series {
        member: s.x()[0].clone(),
    };
    let one = NonZeroU16::new(1).expect("one");
    let page = cube
        .select(&r)
        .expect("visible member")
        .page(None, one)
        .expect("selected");
    assert_eq!(page.value, s.rows()[0].values[0]);
    assert_eq!(page.total, 1);
    let hidden = if s.x()[0] == prod("COMM") {
        prod("AGRI")
    } else {
        prod("COMM")
    };
    r.selection = drill::Selection::Series { member: hidden };
    assert!(matches!(
        cube.select(&r),
        Err(tatami_inmem::drill::Error::NotSelected)
    ));

    r.query.options = query::Options::default();
    r.query.axes = tatami::Axes::Pivot {
        rows: products,
        columns: tatami::Set::members(name("Time"), name("Calendar"), name("Month")),
    };
    r.query
        .metrics
        .push(query::Metric::derived(name("LossRatio")));
    r.metric = 1;
    let tatami::Results::Pivot(p) = cube.query(&r.query).await.expect("pivot") else {
        panic!("pivot")
    };
    let row = p
        .row_headers()
        .iter()
        .position(|t| t.members() == [prod("AGRI")])
        .expect("AGRI");
    r.selection = drill::Selection::Pivot {
        row: p.row_headers()[row].clone(),
        column: p.col_headers()[1].clone(),
    };
    let page = cube
        .select(&r)
        .expect("visible pivot cell")
        .page(None, one)
        .expect("fact");
    assert_eq!(page.value, p.cells()[row][1]);
    assert_eq!(page.total, 1);
    assert_eq!(page.records[0].key[3], "AGRI");
    assert_eq!(page.totals.len(), 2);
}

#[tokio::test]
async fn hidden_members_temporal_metrics_and_forged_continuations_fail_closed() {
    let cube = view("regional", &[true, true, false, false, false, true]);
    let hidden = member("Geography", "World", &["NAM"]);
    assert!(
        cube.members(
            &hidden.dim,
            &hidden.hierarchy,
            &hidden,
            tatami::MemberRelation::Children
        )
        .await
        .is_err()
    );
    let mut selected = reference(cube.snapshot().clone());
    let limit = NonZeroU16::new(1).expect("one");
    let page = cube.select(&selected).expect("selected");
    let mut cursor = page
        .page(None, limit)
        .expect("first")
        .next
        .expect("second page");
    cursor.after = vec!["unrelated".to_owned()];
    assert!(matches!(
        page.page(Some(&cursor), limit),
        Err(tatami_inmem::drill::Error::InvalidCursor)
    ));
    selected.query.metrics = vec![query::Metric::derived(name("RevenueYoY"))];
    assert!(matches!(
        cube.select(&selected),
        Err(tatami_inmem::drill::Error::Unsupported(_))
    ));
}

#[test]
fn invalid_grain_and_partial_visibility_masks_reject_before_aggregation() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/monthly_close.csv");
    let (frame, schema) = cube::facts(&path).expect("monthly facts");
    let partial: BooleanChunked = [true].into_iter().collect();
    assert!(matches!(
        View::new(
            frame.clone(),
            schema.clone(),
            &partial,
            snapshot("test"),
            grain()
        ),
        Err(tatami_inmem::drill::Error::InvalidVisibility)
    ));
    let all: BooleanChunked = [true; 6].into_iter().collect();
    let mut duplicate = grain();
    duplicate.keys = vec![name("year")];
    assert!(matches!(
        View::new(
            frame.clone(),
            schema.clone(),
            &all,
            snapshot("test"),
            duplicate
        ),
        Err(tatami_inmem::drill::Error::InvalidGrain)
    ));
    let denied: BooleanChunked = [None, Some(true), None, None, None, None]
        .into_iter()
        .collect();
    let view =
        View::new(frame, schema, &denied, snapshot("only-ah"), grain()).expect("sparse mask");
    let only = view
        .select(&reference(view.snapshot().clone()))
        .expect("authorized cell");
    let page = only.page(None, NonZeroU16::MIN).expect("permitted fact");
    assert_eq!(page.total, 1);
    assert_eq!(page.records[0].key[3], "AH");
}
