//! Series domain derived from the resolved set, even when no facts survive.

use tatami::{MemberRef, Tuple, crossed};

use super::query::member_ref_of;
use crate::Error;
use crate::resolve::{ResolvedSet, ResolvedTuple};

pub(super) enum Domain {
    // Legacy explicit member lists may contain several different hierarchies,
    // though each individual position still contains just one member.
    Members(Option<crossed::Group>),
    Crossed(Vec<crossed::Group>),
}

impl Domain {
    pub(super) fn of(set: &ResolvedSet<'_>) -> Result<Self, Error> {
        match set {
            ResolvedSet::Members { dim, hierarchy, .. }
            | ResolvedSet::Range { dim, hierarchy, .. } => {
                Ok(Self::Members(Some(crossed::Group {
                    dim: dim.dim.name.clone(),
                    hierarchy: hierarchy.hierarchy.name.clone(),
                })))
            }
            ResolvedSet::Explicit { members } => Ok(Self::Members(
                members
                    .first()
                    .filter(|first| {
                        members.iter().all(|member| {
                            first.dim == member.dim && first.hierarchy == member.hierarchy
                        })
                    })
                    .map(|member| crossed::Group {
                        dim: member.dim.dim.name.clone(),
                        hierarchy: member.hierarchy.hierarchy.name.clone(),
                    }),
            )),
            ResolvedSet::Children { of, .. } | ResolvedSet::Descendants { of, .. } => Self::of(of),
            ResolvedSet::Filter { set, .. }
            | ResolvedSet::TopN { set, .. }
            | ResolvedSet::Named { set } => Self::of(set),
            ResolvedSet::CrossJoin { left, right } => {
                let mut groups = Self::of(left)?.into_groups()?;
                groups.extend(Self::of(right)?.into_groups()?);
                Ok(Self::Crossed(groups))
            }
            ResolvedSet::Union { left, right } => match (Self::of(left)?, Self::of(right)?) {
                (Self::Members(a), Self::Members(b)) => {
                    Ok(Self::Members(a.filter(|g| Some(g) == b.as_ref())))
                }
                (Self::Crossed(a), Self::Crossed(b))
                    if a.len() == b.len() && a.iter().all(|g| b.contains(g)) =>
                {
                    Ok(Self::Crossed(a))
                }
                _ => Err(Error::EvalSetCompositionIllFormed {
                    reason: "series union has incompatible coordinate groupings",
                }),
            },
        }
    }

    fn into_groups(self) -> Result<Vec<crossed::Group>, Error> {
        match self {
            Self::Members(Some(group)) => Ok(vec![group]),
            Self::Crossed(groups) => Ok(groups),
            Self::Members(None) => Err(Error::EvalSetCompositionIllFormed {
                reason: "a crossed series needs a single hierarchy per coordinate",
            }),
        }
    }
}

pub(super) fn coordinate(
    groups: &[crossed::Group],
    row: &ResolvedTuple<'_>,
) -> Result<Tuple, Error> {
    let members: Option<Vec<MemberRef>> = groups
        .iter()
        .map(|group| {
            row.members
                .iter()
                .find(|m| {
                    m.dim.dim.name == group.dim && m.hierarchy.hierarchy.name == group.hierarchy
                })
                .map(member_ref_of)
        })
        .collect();
    members
        .filter(|members| members.len() == row.members.len())
        .and_then(|members| Tuple::of(members).ok())
        .ok_or(Error::EvalSetCompositionIllFormed {
            reason: "crossed tuple differs from its resolved grouping",
        })
}
