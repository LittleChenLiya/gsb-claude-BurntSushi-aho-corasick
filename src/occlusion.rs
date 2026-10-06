//! Static analysis of pattern occlusion under leftmost-first match
//! semantics.
//!
//! Under [`MatchKind::LeftmostFirst`] semantics, when multiple patterns
//! could match at the same starting position, the pattern that was
//! registered earliest wins. This means a pattern can be _occluded_
//! (shadowed) by an earlier pattern, such that it can never be reported in
//! any match. This module provides a static analysis that detects two such
//! cases for non-empty literal patterns:
//!
//! * **Duplicates**: the same literal registered more than once. Every
//!   registration after the first can never match, since the earliest
//!   registration always has strictly higher priority.
//! * **Prefixes**: a pattern that is a proper prefix of a later pattern.
//!   The longer pattern can never match, because whenever it could match
//!   at some position, its shorter prefix also matches at that same
//!   position and has higher priority.
//!
//! Note that this analysis deliberately does _not_ use arbitrary substring
//! containment. For example, `he` occurs inside `she`, but `she` is **not**
//! occluded by `he`: at a position where `she` starts, `he` does not
//! match, so `she` can still be reported. Only exact duplicates and prefix
//! relations imply occlusion under leftmost-first semantics.
//!
//! The analysis is purely static: it inspects only the patterns themselves
//! and never runs the searcher on any (random or otherwise) haystack.
//!
//! # Example: four words
//!
//! ```
//! use aho_corasick::{
//!     occlusion::{find_occlusions, OcclusionKind},
//!     MatchKind, PatternID,
//! };
//!
//! let patterns = &["he", "she", "his", "hers"];
//! let occlusions =
//!     find_occlusions(MatchKind::LeftmostFirst, patterns).unwrap();
//! // `he` is a prefix of `hers` and was registered earlier, so `hers`
//! // can never match. Note that `she` is NOT occluded: `he` occurs
//! // inside it, but not as a prefix.
//! assert_eq!(occlusions.len(), 1);
//! assert_eq!(occlusions[0].occluded(), PatternID::must(3));
//! assert_eq!(occlusions[0].occluder(), PatternID::must(0));
//! assert_eq!(occlusions[0].kind(), OcclusionKind::Prefix);
//! ```

use alloc::vec::Vec;

use crate::{MatchKind, PatternID};

/// The reason why one pattern occludes another under leftmost-first
/// semantics.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OcclusionKind {
    /// The occluding pattern is the exact same literal as the occluded
    /// pattern, but was registered earlier. Since leftmost-first semantics
    /// always prefer the earliest registration, the later duplicate can
    /// never be reported.
    Duplicate,
    /// The occluding pattern is a proper prefix of the occluded pattern
    /// and was registered earlier. Whenever the longer pattern could match
    /// at some position, the shorter prefix matches at the same position
    /// with higher priority, so the longer pattern can never be reported.
    Prefix,
}

/// A witness reporting that one pattern can never match under
/// leftmost-first semantics because an earlier registration takes priority
/// over it.
///
/// A witness consists of the occluded pattern (which can never match), the
/// occluding pattern (the earliest registration that shadows it) and the
/// kind of relation between them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Occlusion {
    occluded: PatternID,
    occluder: PatternID,
    kind: OcclusionKind,
}

impl Occlusion {
    /// Returns the ID of the pattern that can never match.
    #[inline]
    pub fn occluded(&self) -> PatternID {
        self.occluded
    }

    /// Returns the ID of the pattern that occludes it. This is always the
    /// earliest registered pattern (i.e. the one with the highest
    /// leftmost-first priority) that shadows the occluded pattern.
    #[inline]
    pub fn occluder(&self) -> PatternID {
        self.occluder
    }

    /// Returns the kind of relation between the occluder and the occluded
    /// pattern.
    #[inline]
    pub fn kind(&self) -> OcclusionKind {
        self.kind
    }
}

/// An error returned when occlusion analysis is requested for a match kind
/// that this analysis does not support.
///
/// Currently, only [`MatchKind::LeftmostFirst`] is supported. Requesting
/// analysis for any other match kind (including [`MatchKind::Standard`])
/// returns this error instead of attempting to guess occlusion behavior
/// from searches over haystacks.
///
/// When the `std` feature is enabled, this implements the
/// `std::error::Error` trait.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OcclusionError {
    kind: MatchKind,
}

impl OcclusionError {
    /// Returns the match kind for which occlusion analysis is not
    /// supported.
    #[inline]
    pub fn match_kind(&self) -> MatchKind {
        self.kind
    }
}

impl core::fmt::Display for OcclusionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "pattern occlusion analysis is not supported for \
             match kind {:?} (only LeftmostFirst is supported)",
            self.kind,
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OcclusionError {}

/// Analyzes the given patterns for occlusion under the given match kind.
///
/// The patterns are considered in registration order: the pattern at index
/// `0` has the highest leftmost-first priority, the pattern at index `1`
/// the next highest, and so on.
///
/// This analysis only handles non-empty literal patterns and only reports
/// exact duplicates and prefix relations. Empty patterns are ignored
/// entirely: they neither occlude other patterns nor are reported as
/// occluded. Arbitrary substring containment is never treated as
/// occlusion.
///
/// For each occluded pattern, a single witness is returned, naming the
/// earliest registered pattern that occludes it. Duplicate patterns are
/// _not_ deduplicated: each registration keeps its own [`PatternID`], and
/// every duplicate registration after the first is reported as occluded.
///
/// # Errors
///
/// Returns an [`OcclusionError`] when `match_kind` is anything other than
/// [`MatchKind::LeftmostFirst`]. Other match kinds are explicitly not
/// supported by this analysis.
///
/// # Example
///
/// ```
/// use aho_corasick::{
///     occlusion::{find_occlusions, OcclusionKind},
///     MatchKind, PatternID,
/// };
///
/// // `a` is registered before `ab`, so `ab` can never match.
/// let occlusions =
///     find_occlusions(MatchKind::LeftmostFirst, &["a", "ab"]).unwrap();
/// assert_eq!(occlusions.len(), 1);
/// assert_eq!(occlusions[0].occluded(), PatternID::must(1));
/// assert_eq!(occlusions[0].occluder(), PatternID::must(0));
/// assert_eq!(occlusions[0].kind(), OcclusionKind::Prefix);
///
/// // In the reverse registration order, neither pattern is occluded.
/// let occlusions =
///     find_occlusions(MatchKind::LeftmostFirst, &["ab", "a"]).unwrap();
/// assert!(occlusions.is_empty());
/// ```
pub fn find_occlusions<I, P>(
    match_kind: MatchKind,
    patterns: I,
) -> Result<Vec<Occlusion>, OcclusionError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<[u8]>,
{
    if !match_kind.is_leftmost_first() {
        return Err(OcclusionError { kind: match_kind });
    }
    // Collect the patterns along with their registration index. Empty
    // patterns are excluded from the analysis entirely.
    let pats: Vec<P> = patterns.into_iter().collect();
    let mut indexed: Vec<(usize, &[u8])> = Vec::new();
    for (i, pat) in pats.iter().enumerate() {
        let pat = pat.as_ref();
        if !pat.is_empty() {
            indexed.push((i, pat));
        }
    }
    let mut occlusions = Vec::new();
    for &(j, later) in indexed.iter() {
        // Find the earliest registration that occludes `later`. The first
        // hit in registration order is the witness with the highest
        // leftmost-first priority.
        for &(i, earlier) in indexed.iter() {
            if i >= j {
                break;
            }
            let kind = if earlier == later {
                Some(OcclusionKind::Duplicate)
            } else if earlier.len() < later.len() && later.starts_with(earlier)
            {
                Some(OcclusionKind::Prefix)
            } else {
                None
            };
            if let Some(kind) = kind {
                occlusions.push(Occlusion {
                    occluded: PatternID::must(j),
                    occluder: PatternID::must(i),
                    kind,
                });
                break;
            }
        }
    }
    Ok(occlusions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn occlusions(
        match_kind: MatchKind,
        patterns: &[&str],
    ) -> Result<Vec<Occlusion>, OcclusionError> {
        find_occlusions(match_kind, patterns)
    }

    #[test]
    fn prefix_earlier_registration_occludes() {
        // `a` is registered before `ab`, so `ab` is occluded.
        let ocs = occlusions(MatchKind::LeftmostFirst, &["a", "ab"]).unwrap();
        assert_eq!(ocs.len(), 1);
        assert_eq!(ocs[0].occluded(), PatternID::must(1));
        assert_eq!(ocs[0].occluder(), PatternID::must(0));
        assert_eq!(ocs[0].kind(), OcclusionKind::Prefix);
    }

    #[test]
    fn prefix_later_registration_does_not_occlude() {
        // Reverse order: `ab` has priority, so `a` still matches anywhere
        // not followed by `b`. No occlusion may be reported.
        let ocs = occlusions(MatchKind::LeftmostFirst, &["ab", "a"]).unwrap();
        assert!(ocs.is_empty());
    }

    #[test]
    fn duplicates_keep_their_own_ids() {
        // Every duplicate registration keeps its own PatternID and every
        // registration after the first is reported as occluded by the
        // earliest one.
        let ocs = occlusions(MatchKind::LeftmostFirst, &["x", "y", "x", "x"])
            .unwrap();
        assert_eq!(ocs.len(), 2);
        assert_eq!(ocs[0].occluded(), PatternID::must(2));
        assert_eq!(ocs[0].occluder(), PatternID::must(0));
        assert_eq!(ocs[0].kind(), OcclusionKind::Duplicate);
        assert_eq!(ocs[1].occluded(), PatternID::must(3));
        assert_eq!(ocs[1].occluder(), PatternID::must(0));
        assert_eq!(ocs[1].kind(), OcclusionKind::Duplicate);
    }

    #[test]
    fn four_words_example() {
        // The classic four-word set. `he` is a prefix of `hers`, so `hers`
        // is occluded. `he` occurs inside `she` only as a substring, which
        // must NOT be treated as occlusion.
        let ocs = occlusions(
            MatchKind::LeftmostFirst,
            &["he", "she", "his", "hers"],
        )
        .unwrap();
        assert_eq!(ocs.len(), 1);
        assert_eq!(ocs[0].occluded(), PatternID::must(3));
        assert_eq!(ocs[0].occluder(), PatternID::must(0));
        assert_eq!(ocs[0].kind(), OcclusionKind::Prefix);
    }

    #[test]
    fn substring_containment_is_not_occlusion() {
        // `he` is a substring of `she`, but not a prefix, so nothing is
        // occluded.
        let ocs =
            occlusions(MatchKind::LeftmostFirst, &["he", "she"]).unwrap();
        assert!(ocs.is_empty());
    }

    #[test]
    fn empty_patterns_are_ignored() {
        let ocs = occlusions(MatchKind::LeftmostFirst, &["", "a", "", "ab"])
            .unwrap();
        assert_eq!(ocs.len(), 1);
        assert_eq!(ocs[0].occluded(), PatternID::must(3));
        assert_eq!(ocs[0].occluder(), PatternID::must(1));
        assert_eq!(ocs[0].kind(), OcclusionKind::Prefix);
    }

    #[test]
    fn witness_is_earliest_registration() {
        // Both `a` (0) and `ab` (1) occlude `abc` (2); the witness must
        // name the earliest registration.
        let ocs =
            occlusions(MatchKind::LeftmostFirst, &["a", "ab", "abc"]).unwrap();
        assert_eq!(ocs.len(), 2);
        assert_eq!(ocs[0].occluded(), PatternID::must(1));
        assert_eq!(ocs[0].occluder(), PatternID::must(0));
        assert_eq!(ocs[1].occluded(), PatternID::must(2));
        assert_eq!(ocs[1].occluder(), PatternID::must(0));
    }

    #[test]
    fn standard_match_kind_is_unsupported() {
        let err = occlusions(MatchKind::Standard, &["a", "ab"]).unwrap_err();
        assert_eq!(err.match_kind(), MatchKind::Standard);
    }

    #[test]
    fn leftmost_longest_match_kind_is_unsupported() {
        let err =
            occlusions(MatchKind::LeftmostLongest, &["a", "ab"]).unwrap_err();
        assert_eq!(err.match_kind(), MatchKind::LeftmostLongest);
    }
}
