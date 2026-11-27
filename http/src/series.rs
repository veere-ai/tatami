//! Reject old servers that flatten a requested grouping to its first member.

use tatami::schema::Name;
use tatami::{Schema, Set};

use crate::Error;

// None means that the set references the schema and must be resolved before
// accepting a legacy single-coordinate result.
pub(crate) fn crossed(set: &Set, schema: Option<&Schema>) -> Result<Option<bool>, Error> {
    fn visit(
        set: &Set,
        schema: Option<&Schema>,
        active: &mut Vec<Name>,
    ) -> Result<Option<bool>, Error> {
        match set {
            Set::CrossJoin { .. } => Ok(Some(true)),
            Set::Children { of } | Set::Descendants { of, .. } => visit(of, schema, active),
            Set::Filter { set, .. } | Set::TopN { set, .. } => visit(set, schema, active),
            Set::Union { left, right } => {
                let left = visit(left, schema, active)?;
                let right = visit(right, schema, active)?;
                Ok(match (left, right) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                })
            }
            Set::Named { name } => {
                let Some(schema) = schema else {
                    return Ok(None);
                };
                let set = schema
                    .named_sets
                    .iter()
                    .find(|set| &set.name == name)
                    .filter(|_| !active.contains(name))
                    .ok_or(Error::UnverifiedGrouping)?;
                active.push(name.clone());
                let result = visit(&set.set, Some(schema), active);
                active.pop();
                result
            }
            Set::Members { .. } | Set::Range { .. } | Set::Explicit { .. } => Ok(Some(false)),
            _ => Err(Error::UnverifiedGrouping),
        }
    }
    visit(set, schema, &mut Vec::new())
}
