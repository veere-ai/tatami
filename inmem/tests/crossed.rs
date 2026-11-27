//! The quarterly series keeps named year/scenario coordinates through all axis operations.

use std::num::NonZeroUsize;

use polars_core::df;
use tatami::{Cell, Cube, Query, Results, crossed, query, schema};
use tatami_inmem::InMemoryCube;

fn name(s: &str) -> schema::Name {
    schema::Name::parse(s).unwrap()
}

fn axis(column: &str) -> query::Set {
    query::Set::members(name(column), name("Default"), name(column))
}

fn cube(budget_nep: f64) -> InMemoryCube {
    let dimension = |id| {
        schema::Dimension::regular(name(id)).hierarchy(
            schema::Hierarchy::new(name("Default")).level(schema::Level::new(name(id), name(id))),
        )
    };
    let measure = |id| schema::Measure::new(name(id), schema::Aggregation::sum());
    let schema = schema::Schema::builder()
        .dimension(dimension("year"))
        .dimension(dimension("scenario"))
        .measure(measure("total_expenses"))
        .measure(measure("nep"));
    let facts = df!(
        "year" => ["2024","2025","2025"],
        "scenario" => ["actual","actual","budget"],
        "total_expenses" => [110_965_342.07,169_961_679.92,163_250_000.37],
        "nep" => [326_005_505.67,483_441_370.93,budget_nep],
    )
    .unwrap();
    InMemoryCube::new(facts, schema.build().unwrap()).unwrap()
}

fn grouped(rows: query::Set) -> Query {
    Query {
        axes: query::Axes::Series { rows },
        slicer: query::Tuple::empty(),
        metrics: ["total_expenses", "nep"]
            .map(|id| query::Metric::aggregated(name(id), schema::Aggregation::sum()))
            .into(),
        options: query::Options::default(),
    }
}

async fn run(cube: &InMemoryCube, query: &Query) -> crossed::Result {
    let Results::CrossedSeries(result) = cube.query(query).await.unwrap() else {
        panic!("expected full tuples")
    };
    result
}

fn points(result: &crossed::Result) -> Vec<(String, String, Option<i64>, Option<i64>)> {
    let cents = |cell: &Cell| match cell {
        Cell::Valid { value, .. } => Some((value * 100.).round() as i64),
        Cell::Missing { .. } => None,
        other => panic!("unexpected cell {other:?}"),
    };
    result
        .x()
        .iter()
        .enumerate()
        .map(|(i, coord)| {
            let members = coord.members();
            (
                members[0].path.to_string(),
                members[1].path.to_string(),
                cents(&result.rows()[0].values[i]),
                cents(&result.rows()[1].values[i]),
            )
        })
        .collect()
}

#[tokio::test]
async fn two_metrics_keep_year_and_scenario_through_filter_rank_sort_limit_and_empty_results() {
    let cube = cube(426_500_000.86);
    let rows = axis("year").cross(axis("scenario"));
    let mut q = grouped(rows.clone());
    q.options.order = ["year", "scenario"]
        .map(|id| query::OrderBy {
            key: query::OrderKey::Dimension(name(id)),
            direction: query::Direction::Asc,
        })
        .into();
    let all = run(&cube, &q).await;
    assert_eq!(
        all.groups()
            .iter()
            .map(|group| (group.dim.as_str(), group.hierarchy.as_str()))
            .collect::<Vec<_>>(),
        [("year", "Default"), ("scenario", "Default")]
    );
    assert_eq!(
        all.rows()
            .iter()
            .map(|m| m.label.as_str())
            .collect::<Vec<_>>(),
        ["total_expenses", "nep"]
    );
    let expected = vec![
        (
            "2024".into(),
            "actual".into(),
            Some(11_096_534_207),
            Some(32_600_550_567),
        ),
        ("2024".into(), "budget".into(), None, None),
        (
            "2025".into(),
            "actual".into(),
            Some(16_996_167_992),
            Some(48_344_137_093),
        ),
        (
            "2025".into(),
            "budget".into(),
            Some(16_325_000_037),
            Some(42_650_000_086),
        ),
    ];
    assert_eq!(points(&all), expected);
    q.options.non_empty = true;
    assert_eq!(
        points(&run(&cube, &q).await),
        [
            expected[0].clone(),
            expected[2].clone(),
            expected[3].clone()
        ]
    );
    q.options.order = vec![query::OrderBy {
        key: query::OrderKey::Dimension(name("scenario")),
        direction: query::Direction::Desc,
    }];
    q.options.rank = Some(query::Rank {
        metric: name("nep"),
        direction: query::Direction::Desc,
    });
    q.options.limit = NonZeroUsize::new(2);
    assert_eq!(
        points(&run(&cube, &q).await),
        [expected[3].clone(), expected[2].clone()]
    );
    let top = rows
        .clone()
        .filter(query::Predicate::Gt {
            metric: name("nep"),
            value: 1.0,
        })
        .top(NonZeroUsize::new(1).unwrap(), name("total_expenses"));
    assert_eq!(
        points(&run(&cube, &grouped(top)).await),
        [expected[2].clone()]
    );
    let empty = run(
        &cube,
        &grouped(rows.filter(query::Predicate::Lt {
            metric: name("nep"),
            value: 0.0,
        })),
    )
    .await;
    assert_eq!(empty.groups(), all.groups());
    assert!(empty.x().is_empty());
    assert!(empty.rows().iter().all(|m| m.values.is_empty()));
}

#[tokio::test]
async fn a_sliced_budget_preserves_a_reported_zero_and_omits_excluded_coordinates() {
    let cube = cube(0.0);
    let mut q = grouped(axis("year").cross(axis("scenario")));
    q.slicer = query::Tuple::single(query::MemberRef::new(
        name("scenario"),
        name("Default"),
        query::Path::of(name("budget")),
    ));
    q.options.order = vec![query::OrderBy {
        key: query::OrderKey::Dimension(name("year")),
        direction: query::Direction::Asc,
    }];
    let sliced = points(&run(&cube, &q).await);
    assert_eq!(
        sliced,
        [
            ("2024".into(), "budget".into(), None, None),
            (
                "2025".into(),
                "budget".into(),
                Some(16_325_000_037),
                Some(0)
            )
        ]
    );
}
