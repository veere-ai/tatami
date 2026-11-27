//! [`Dimension`] and its immediate components — [`Kind`] (the dim kind),
//! [`Calendar`],
//! [`Hierarchy`], [`Level`].

use serde::{Deserialize, Serialize};

use crate::schema::{MonthDay, Name};

/// A cube dimension — identified by name, typed by [`Kind`], structured by
/// one or more [`Hierarchy`] entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dimension {
    /// Dimension name; unique within a schema.
    pub name: Name,
    /// One or more hierarchies (e.g., `"Fiscal"`, `"Calendar"`) partitioning
    /// this dimension.
    pub hierarchies: Vec<Hierarchy>,
    /// The structural kind of the dimension — regular, time, or scenario.
    pub kind: Kind,
}

impl Dimension {
    /// Construct a regular dimension with no hierarchies.
    #[must_use]
    pub fn regular(name: Name) -> Self {
        Self {
            name,
            hierarchies: Vec::new(),
            kind: Kind::Regular,
        }
    }

    /// Construct a time dimension with the given calendars.
    #[must_use]
    pub fn time(name: Name, calendars: Vec<Calendar>) -> Self {
        Self {
            name,
            hierarchies: Vec::new(),
            kind: Kind::Time { calendars },
        }
    }

    /// Construct a scenario dimension.
    #[must_use]
    pub fn scenario(name: Name) -> Self {
        Self {
            name,
            hierarchies: Vec::new(),
            kind: Kind::Scenario,
        }
    }

    /// Fluent: append a hierarchy.
    #[must_use]
    pub fn hierarchy(mut self, hierarchy: Hierarchy) -> Self {
        self.hierarchies.push(hierarchy);
        self
    }
}

/// The structural kind of a [`Dimension`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Kind {
    /// A regular dimension — geography, product, account.
    Regular,
    /// A time dimension, carrying one or more calendars (fiscal, Gregorian,
    /// retail, …).
    Time {
        /// Calendars attached to this time dimension.
        calendars: Vec<Calendar>,
    },
    /// A scenario dimension — Actual / Plan / Forecast / What-If.
    Scenario,
}

/// A calendar attached to a [`Kind::Time`] dimension.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Calendar {
    /// Calendar name — `"Fiscal"`, `"Gregorian"`, `"Retail-4-4-5"`.
    pub name: Name,
    /// Fiscal year start as a recurring `(month, day)`. `None` means
    /// Gregorian (January 1).
    pub fiscal_start: Option<MonthDay>,
}

impl Calendar {
    /// A Gregorian calendar — `fiscal_start = None`.
    #[must_use]
    pub fn gregorian(name: Name) -> Self {
        Self {
            name,
            fiscal_start: None,
        }
    }

    /// A fiscal calendar starting on the given recurring `(month, day)`.
    #[must_use]
    pub fn fiscal(name: Name, start: MonthDay) -> Self {
        Self {
            name,
            fiscal_start: Some(start),
        }
    }
}

/// A named hierarchy — an ordered list of [`Level`] entries from root to
/// leaf. `Year → Quarter → Month → Day` is the canonical time example.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hierarchy {
    /// Hierarchy name — `"Fiscal"`, `"Calendar"`, `"Default"`.
    pub name: Name,
    /// Levels, top-down.
    pub levels: Vec<Level>,
}

impl Hierarchy {
    /// Construct a hierarchy with no levels.
    #[must_use]
    pub fn new(name: Name) -> Self {
        Self {
            name,
            levels: Vec::new(),
        }
    }

    /// Fluent: append a level.
    #[must_use]
    pub fn level(mut self, level: Level) -> Self {
        self.levels.push(level);
        self
    }
}

/// A level within a [`Hierarchy`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Level {
    /// Display name — `"Year"`, `"Quarter"`, `"Country"`.
    pub name: Name,
    /// Key column used to identify members at this level in the fact source.
    pub key: Name,
    /// Column that orders this level's members among their siblings: each
    /// member ranks by the smallest value its rows carry, and members with
    /// equal or no rank fall back to their names. A date column gives months
    /// in calendar order where their names would sort `Apr`, `Aug`, `Dec`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Name>,
}

impl Level {
    /// Construct a level whose members sort by name.
    #[must_use]
    pub fn new(name: Name, key: Name) -> Self {
        Self {
            name,
            key,
            order: None,
        }
    }

    /// Fluent: order this level's members by the values of `column`.
    #[must_use]
    pub fn ordered_by(mut self, column: Name) -> Self {
        self.order = Some(column);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::month_day::Month;

    fn n(s: &str) -> Name {
        Name::parse(s).expect("valid")
    }

    #[test]
    fn regular_dimension_roundtrips() {
        let dim = Dimension::regular(n("Geography")).hierarchy(
            Hierarchy::new(n("Default"))
                .level(Level::new(n("Region"), n("region_key")))
                .level(Level::new(n("Country"), n("country_key"))),
        );
        let json = serde_json::to_string(&dim).expect("serialize");
        let back: Dimension = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(dim, back);
    }

    #[test]
    fn time_dimension_with_fiscal_calendar_roundtrips() {
        let dim = Dimension::time(
            n("Time"),
            vec![
                Calendar::gregorian(n("Gregorian")),
                Calendar::fiscal(n("Fiscal"), MonthDay::new(Month::April, 1).expect("valid")),
            ],
        );
        let json = serde_json::to_string(&dim).expect("serialize");
        let back: Dimension = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(dim, back);
    }

    #[test]
    fn an_ordered_level_roundtrips_and_an_unordered_one_keeps_its_old_form() {
        let plain = Level::new(n("Month"), n("month"));
        assert_eq!(
            serde_json::to_string(&plain).expect("serialize"),
            r#"{"name":"Month","key":"month"}"#
        );
        let ordered = plain.ordered_by(n("date"));
        let json = serde_json::to_string(&ordered).expect("serialize");
        let back: Level = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(ordered, back);
    }

    #[test]
    fn scenario_dimension_roundtrips() {
        let dim = Dimension::scenario(n("Scenario"));
        let json = serde_json::to_string(&dim).expect("serialize");
        let back: Dimension = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(dim, back);
    }
}
