/*!
Pattern occlusion analysis for leftmost-first match semantics.

With [`MatchKind::LeftmostFirst`], when multiple patterns could match at
the same starting position, the pattern that was registered first always
wins. This means some patterns can never match at all: they are "occluded"
by an earlier pattern. This module provides a structural analysis that
reports such occlusions without running any searches.

Only two relations between non-empty literal patterns are considered:

* **Duplicates**: if pattern `i` and pattern `j` are the same word and
  `i < j`, then `j` can never match because `i` has higher priority.
* **Prefixes**: if pattern `i` is a proper prefix of pattern `j` and
  `i < j`, then whenever `j` could start a match, `i` matches at the same
  position and wins by priority, so `j` can never match.

Note that an arbitrary substring relation is **not** occlusion. For
example, with patterns `bc` and `abcd`, the leftmost match in `abcd`
starts at position `0`, so `abcd` is still reported even though `bc` was
registered first and appears inside it.

This analysis is purely structural. It does not run the matcher and does
not guess from sample text. The matcher itself is unchanged; this module
only inspects the pattern list.

# Example

```
use aho_corasick::{occlusion, MatchKind};

let patterns = ["he", "help", "he", "hello"];
let occlusions = occlusion::analyze(&patterns, MatchKind::LeftmostFirst)?;
assert_eq!(3, occlusions.len());
// "help" is occluded by its earlier prefix "he".
assert_eq!(1, occlusions[0].pattern().as_usize());
assert_eq!(0, occlusions[0].witness().as_usize());
// The duplicate "he" keeps its own pattern ID and is occluded by the
// first registration of "he".
assert_eq!(2, occlusions[1].pattern().as_usize());
assert_eq!(0, occlusions[1].witness().as_usize());
// "hello" is also occluded by "he".
assert_eq!(3, occlusions[2].pattern().as_usize());
assert_eq!(0, occlusions[2].witness().as_usize());
# Ok::<(), aho_corasick::occlusion::OcclusionError>(())
```
*/

use alloc::vec::Vec;

use crate::{MatchKind, PatternID};

/// Analyzes a pattern list for occlusions under leftmost-first semantics.
///
/// Patterns are examined in registration order. A non-empty pattern is
/// reported as occluded when an earlier non-empty pattern is either
/// exactly equal to it ([`OcclusionKind::Duplicate`]) or a proper prefix
/// of it ([`OcclusionKind::Prefix`]). The witness reported is the earliest
/// (and thus highest priority) pattern that occludes it.
///
/// Empty patterns are ignored entirely: they are never reported as
/// occluded and never used as witnesses.
///
/// Every pattern keeps its own pattern ID, including duplicates. The IDs
/// in the returned [`Occlusion`] values always correspond to the positions
/// of the patterns in the given slice.
///
/// # Errors
///
/// This analysis is only defined for [`MatchKind::LeftmostFirst`]. Any
/// other match kind (including [`MatchKind::Standard`] and
/// [`MatchKind::LeftmostLongest`]) returns an [`OcclusionError`].
pub fn analyze<P: AsRef<[u8]>>(
    patterns: &[P],
    kind: MatchKind,
) -> Result<Vec<Occlusion>, OcclusionError> {
    if !kind.is_leftmost_first() {
        return Err(OcclusionError { kind });
    }
    let mut occlusions = Vec::new();
    for (j, later) in patterns.iter().enumerate() {
        let later = later.as_ref();
        if later.is_empty() {
            continue;
        }
        for (i, earlier) in patterns[..j].iter().enumerate() {
            let earlier = earlier.as_ref();
            if earlier.is_empty() {
                continue;
            }
            let kind = if earlier == later {
                OcclusionKind::Duplicate
            } else if later.starts_with(earlier) {
                OcclusionKind::Prefix
            } else {
                continue;
            };
            occlusions.push(Occlusion {
                pattern: PatternID::new(j).unwrap(),
                witness: PatternID::new(i).unwrap(),
                kind,
            });
            // The earliest witness has the highest priority, so no later
            // witness can change the outcome for this pattern.
            break;
        }
    }
    Ok(occlusions)
}

/// A single occlusion: a pattern that can never match under leftmost-first
/// semantics because an earlier registered pattern always wins over it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Occlusion {
    /// The ID of the occluded pattern.
    pattern: PatternID,
    /// The ID of the earlier pattern that occludes it.
    witness: PatternID,
    /// The relation between the witness and the occluded pattern.
    kind: OcclusionKind,
}

impl Occlusion {
    /// Returns the ID of the pattern that can never match.
    #[inline]
    pub fn pattern(&self) -> PatternID {
        self.pattern
    }

    /// Returns the ID of the earlier registered pattern that occludes
    /// [`Occlusion::pattern`].
    ///
    /// The witness is the highest priority (earliest registered) pattern
    /// with an occlusion relation to the occluded pattern.
    #[inline]
    pub fn witness(&self) -> PatternID {
        self.witness
    }

    /// Returns the kind of the occlusion relation.
    #[inline]
    pub fn kind(&self) -> OcclusionKind {
        self.kind
    }
}

/// The kind of relation that causes one pattern to occlude another under
/// leftmost-first semantics.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OcclusionKind {
    /// The witness is the same word as the occluded pattern, but was
    /// registered earlier. Both patterns keep their own pattern IDs.
    Duplicate,
    /// The witness is a proper prefix of the occluded pattern and was
    /// registered earlier.
    Prefix,
}

/// An error returned by [`analyze`] when the requested match kind is not
/// supported by occlusion analysis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OcclusionError {
    kind: MatchKind,
}

impl OcclusionError {
    /// Returns the match kind that occlusion analysis does not support.
    #[inline]
    pub fn unsupported_kind(&self) -> MatchKind {
        self.kind
    }
}

impl core::fmt::Display for OcclusionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "pattern occlusion analysis is only supported for \
             leftmost-first match semantics, not {:?}",
            self.kind,
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OcclusionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn analyze_strs(
        patterns: &[&str],
    ) -> Result<Vec<Occlusion>, OcclusionError> {
        analyze(patterns, MatchKind::LeftmostFirst)
    }

    fn triples(
        occlusions: &[Occlusion],
    ) -> Vec<(usize, usize, OcclusionKind)> {
        occlusions
            .iter()
            .map(|o| {
                (o.pattern().as_usize(), o.witness().as_usize(), o.kind())
            })
            .collect()
    }

    #[test]
    fn prefix_after_shorter_is_occluded() {
        // "a" registered before "ab": "ab" can never match.
        let occlusions = analyze_strs(&["a", "ab"]).unwrap();
        assert_eq!(vec![(1, 0, OcclusionKind::Prefix)], triples(&occlusions));
    }

    #[test]
    fn reverse_order_is_not_occluded() {
        // "ab" registered before "a": "a" still matches, e.g. in "xa".
        let occlusions = analyze_strs(&["ab", "a"]).unwrap();
        assert!(occlusions.is_empty());
    }

    #[test]
    fn duplicates_keep_their_own_ids() {
        // Each duplicate keeps its own pattern ID; the later registration
        // is occluded by the earliest identical one.
        let occlusions = analyze_strs(&["x", "y", "x"]).unwrap();
        assert_eq!(
            vec![(2, 0, OcclusionKind::Duplicate)],
            triples(&occlusions)
        );
    }

    #[test]
    fn substring_containment_is_not_occlusion() {
        // "bc" is a substring of "abcd", but not a prefix: no occlusion
        // in either registration order.
        assert!(analyze_strs(&["bc", "abcd"]).unwrap().is_empty());
        assert!(analyze_strs(&["abcd", "bc"]).unwrap().is_empty());
    }

    #[test]
    fn witness_is_earliest_by_registration_priority() {
        // Both "a" and "ab" are prefixes of "abc", but "a" was registered
        // first and is therefore the witness.
        let occlusions = analyze_strs(&["a", "ab", "abc"]).unwrap();
        assert_eq!(
            vec![(1, 0, OcclusionKind::Prefix), (2, 0, OcclusionKind::Prefix),],
            triples(&occlusions)
        );
    }

    #[test]
    fn empty_patterns_are_ignored() {
        // Empty patterns are neither occluded nor witnesses.
        assert!(analyze_strs(&["", "a", ""]).unwrap().is_empty());
        assert!(analyze_strs(&["", "ab"]).unwrap().is_empty());
        assert!(analyze_strs(&["ab", ""]).unwrap().is_empty());
    }

    #[test]
    fn four_word_example() {
        let occlusions = analyze_strs(&["he", "help", "he", "hello"]).unwrap();
        assert_eq!(
            vec![
                (1, 0, OcclusionKind::Prefix),
                (2, 0, OcclusionKind::Duplicate),
                (3, 0, OcclusionKind::Prefix),
            ],
            triples(&occlusions)
        );
    }

    #[test]
    fn standard_match_kind_is_unsupported() {
        let err = analyze(&["a", "ab"], MatchKind::Standard).unwrap_err();
        assert_eq!(MatchKind::Standard, err.unsupported_kind());
    }

    #[test]
    fn leftmost_longest_is_unsupported() {
        let err =
            analyze(&["a", "ab"], MatchKind::LeftmostLongest).unwrap_err();
        assert_eq!(MatchKind::LeftmostLongest, err.unsupported_kind());
    }

    #[test]
    fn no_patterns_or_single_pattern_has_no_occlusion() {
        let empty: &[&str] = &[];
        assert!(analyze_strs(empty).unwrap().is_empty());
        assert!(analyze_strs(&["only"]).unwrap().is_empty());
    }
}
