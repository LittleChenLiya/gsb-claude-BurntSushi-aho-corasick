use crate::util::{
    primitives::{PatternID, SmallIndex},
    search::{MatchKind, Span},
};

/// An error that occurred during the construction of an Aho-Corasick
/// automaton.
///
/// Build errors occur when some kind of limit has been exceeded, either in the
/// number of states, the number of patterns of the length of a pattern. These
/// limits aren't part of the public API, but they should generally be large
/// enough to handle most use cases.
///
/// When the `std` feature is enabled, this implements the `std::error::Error`
/// trait.
#[derive(Clone, Debug)]
pub struct BuildError {
    kind: ErrorKind,
}

/// The kind of error that occurred.
#[derive(Clone, Debug)]
enum ErrorKind {
    /// An error that occurs when allocating a new state would result in an
    /// identifier that exceeds the capacity of a `StateID`.
    StateIDOverflow {
        /// The maximum possible id.
        max: u64,
        /// The maximum ID requested.
        requested_max: u64,
    },
    /// An error that occurs when adding a pattern to an Aho-Corasick
    /// automaton would result in an identifier that exceeds the capacity of a
    /// `PatternID`.
    PatternIDOverflow {
        /// The maximum possible id.
        max: u64,
        /// The maximum ID requested.
        requested_max: u64,
    },
    /// Occurs when a pattern string is given to the Aho-Corasick constructor
    /// that is too long.
    PatternTooLong {
        /// The ID of the pattern that was too long.
        pattern: PatternID,
        /// The length that was too long.
        len: usize,
    },
}

impl BuildError {
    pub(crate) fn state_id_overflow(
        max: u64,
        requested_max: u64,
    ) -> BuildError {
        BuildError { kind: ErrorKind::StateIDOverflow { max, requested_max } }
    }

    pub(crate) fn pattern_id_overflow(
        max: u64,
        requested_max: u64,
    ) -> BuildError {
        BuildError {
            kind: ErrorKind::PatternIDOverflow { max, requested_max },
        }
    }

    pub(crate) fn pattern_too_long(
        pattern: PatternID,
        len: usize,
    ) -> BuildError {
        BuildError { kind: ErrorKind::PatternTooLong { pattern, len } }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BuildError {}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind {
            ErrorKind::StateIDOverflow { max, requested_max } => {
                write!(
                    f,
                    "state identifier overflow: failed to create state ID \
                     from {}, which exceeds the max of {}",
                    requested_max, max,
                )
            }
            ErrorKind::PatternIDOverflow { max, requested_max } => {
                write!(
                    f,
                    "pattern identifier overflow: failed to create pattern ID \
                     from {}, which exceeds the max of {}",
                    requested_max, max,
                )
            }
            ErrorKind::PatternTooLong { pattern, len } => {
                write!(
                    f,
                    "pattern {} with length {} exceeds \
                     the maximum pattern length of {}",
                    pattern.as_usize(),
                    len,
                    SmallIndex::MAX.as_usize(),
                )
            }
        }
    }
}

/// An error that occurred during an Aho-Corasick search.
///
/// An error that occurs during a search is limited to some kind of
/// misconfiguration that resulted in an illegal call. Stated differently,
/// whether an error occurs is not dependent on the specific bytes in the
/// haystack.
///
/// Examples of misconfiguration:
///
/// * Executing a stream or overlapping search on a searcher that was built was
/// something other than [`MatchKind::Standard`](crate::MatchKind::Standard)
/// semantics.
/// * Requested an anchored or an unanchored search on a searcher that doesn't
/// support unanchored or anchored searches, respectively.
///
/// When the `std` feature is enabled, this implements the `std::error::Error`
/// trait.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchError(alloc::boxed::Box<MatchErrorKind>);

impl MatchError {
    /// Create a new error value with the given kind.
    ///
    /// This is a more verbose version of the kind-specific constructors, e.g.,
    /// `MatchError::unsupported_stream`.
    pub fn new(kind: MatchErrorKind) -> MatchError {
        MatchError(alloc::boxed::Box::new(kind))
    }

    /// Returns a reference to the underlying error kind.
    pub fn kind(&self) -> &MatchErrorKind {
        &self.0
    }

    /// Create a new "invalid anchored search" error. This occurs when the
    /// caller requests an anchored search but where anchored searches aren't
    /// supported.
    ///
    /// This is the same as calling `MatchError::new` with a
    /// [`MatchErrorKind::InvalidInputAnchored`] kind.
    pub fn invalid_input_anchored() -> MatchError {
        MatchError::new(MatchErrorKind::InvalidInputAnchored)
    }

    /// Create a new "invalid unanchored search" error. This occurs when the
    /// caller requests an unanchored search but where unanchored searches
    /// aren't supported.
    ///
    /// This is the same as calling `MatchError::new` with a
    /// [`MatchErrorKind::InvalidInputUnanchored`] kind.
    pub fn invalid_input_unanchored() -> MatchError {
        MatchError::new(MatchErrorKind::InvalidInputUnanchored)
    }

    /// Create a new "unsupported stream search" error. This occurs when the
    /// caller requests a stream search while using an Aho-Corasick automaton
    /// with a match kind other than [`MatchKind::Standard`].
    ///
    /// The match kind given should be the match kind of the automaton. It
    /// should never be `MatchKind::Standard`.
    pub fn unsupported_stream(got: MatchKind) -> MatchError {
        MatchError::new(MatchErrorKind::UnsupportedStream { got })
    }

    /// Create a new "unsupported overlapping search" error. This occurs when
    /// the caller requests an overlapping search while using an Aho-Corasick
    /// automaton with a match kind other than [`MatchKind::Standard`].
    ///
    /// The match kind given should be the match kind of the automaton. It
    /// should never be `MatchKind::Standard`.
    pub fn unsupported_overlapping(got: MatchKind) -> MatchError {
        MatchError::new(MatchErrorKind::UnsupportedOverlapping { got })
    }

    /// Create a new "unsupported empty pattern" error. This occurs when the
    /// caller requests a search for which matching an automaton that contains
    /// an empty pattern string is not supported.
    pub fn unsupported_empty() -> MatchError {
        MatchError::new(MatchErrorKind::UnsupportedEmpty)
    }
}

/// The underlying kind of a [`MatchError`].
///
/// This is a **non-exhaustive** enum. That means new variants may be added in
/// a semver-compatible release.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MatchErrorKind {
    /// An error indicating that an anchored search was requested, but from a
    /// searcher that was built without anchored support.
    InvalidInputAnchored,
    /// An error indicating that an unanchored search was requested, but from a
    /// searcher that was built without unanchored support.
    InvalidInputUnanchored,
    /// An error indicating that a stream search was attempted on an
    /// Aho-Corasick automaton with an unsupported `MatchKind`.
    UnsupportedStream {
        /// The match semantics for the automaton that was used.
        got: MatchKind,
    },
    /// An error indicating that an overlapping search was attempted on an
    /// Aho-Corasick automaton with an unsupported `MatchKind`.
    UnsupportedOverlapping {
        /// The match semantics for the automaton that was used.
        got: MatchKind,
    },
    /// An error indicating that the operation requested doesn't support
    /// automatons that contain an empty pattern string.
    UnsupportedEmpty,
}

#[cfg(feature = "std")]
impl std::error::Error for MatchError {}

impl core::fmt::Display for MatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match *self.kind() {
            MatchErrorKind::InvalidInputAnchored => {
                write!(f, "anchored searches are not supported or enabled")
            }
            MatchErrorKind::InvalidInputUnanchored => {
                write!(f, "unanchored searches are not supported or enabled")
            }
            MatchErrorKind::UnsupportedStream { got } => {
                write!(
                    f,
                    "match kind {:?} does not support stream searching",
                    got,
                )
            }
            MatchErrorKind::UnsupportedOverlapping { got } => {
                write!(
                    f,
                    "match kind {:?} does not support overlapping searches",
                    got,
                )
            }
            MatchErrorKind::UnsupportedEmpty => {
                write!(
                    f,
                    "matching with an empty pattern string is not \
                     supported for this operation",
                )
            }
        }
    }
}

/// An error that occurred while replacing raw byte matches in a haystack.
///
/// This error is used by replacement routines that operate on an explicit
/// sequence of [`Match`](crate::Match) results, e.g.,
/// [`AhoCorasick::try_replace_matches_bytes`](crate::AhoCorasick::try_replace_matches_bytes).
/// Since callers provide the match spans themselves, the spans are validated
/// before any replacement occurs. In particular, overlapping matches are
/// always reported as an error rather than being resolved by arbitrarily
/// choosing one of them.
///
/// When the `std` feature is enabled, this implements the `std::error::Error`
/// trait.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplaceError(alloc::boxed::Box<ReplaceErrorKind>);

impl ReplaceError {
    /// Create a new error value with the given kind.
    ///
    /// This is a more verbose version of the kind-specific constructors, e.g.,
    /// `ReplaceError::overlapping_matches`.
    pub fn new(kind: ReplaceErrorKind) -> ReplaceError {
        ReplaceError(alloc::boxed::Box::new(kind))
    }

    /// Returns a reference to the underlying error kind.
    pub fn kind(&self) -> &ReplaceErrorKind {
        &self.0
    }

    /// Create a new "overlapping matches" error. This occurs when a match
    /// in the sequence given starts before the end of the previous match.
    ///
    /// This is the same as calling `ReplaceError::new` with a
    /// [`ReplaceErrorKind::OverlappingMatches`] kind.
    pub fn overlapping_matches(first: Span, second: Span) -> ReplaceError {
        ReplaceError::new(ReplaceErrorKind::OverlappingMatches {
            first,
            second,
        })
    }

    /// Create a new "unknown pattern ID" error. This occurs when a match
    /// has a pattern ID that has no corresponding replacement.
    ///
    /// This is the same as calling `ReplaceError::new` with a
    /// [`ReplaceErrorKind::UnknownPatternID`] kind.
    pub fn unknown_pattern_id(pattern: PatternID) -> ReplaceError {
        ReplaceError::new(ReplaceErrorKind::UnknownPatternID { pattern })
    }

    /// Create a new "span out of bounds" error. This occurs when a match
    /// has a span that falls outside of the haystack.
    ///
    /// This is the same as calling `ReplaceError::new` with a
    /// [`ReplaceErrorKind::SpanOutOfBounds`] kind.
    pub fn span_out_of_bounds(
        span: Span,
        haystack_len: usize,
    ) -> ReplaceError {
        ReplaceError::new(ReplaceErrorKind::SpanOutOfBounds {
            span,
            haystack_len,
        })
    }
}

/// The underlying kind of a [`ReplaceError`].
///
/// This is a **non-exhaustive** enum. That means new variants may be added in
/// a semver-compatible release.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplaceErrorKind {
    /// An error indicating that two matches in the sequence given overlap.
    /// Overlapping matches are never resolved by choosing one arbitrarily;
    /// they are always rejected.
    OverlappingMatches {
        /// The span of the earlier match.
        first: Span,
        /// The span of the match that overlaps with `first`.
        second: Span,
    },
    /// An error indicating that a match has a pattern ID for which no
    /// replacement was provided.
    UnknownPatternID {
        /// The pattern ID that has no corresponding replacement.
        pattern: PatternID,
    },
    /// An error indicating that a match has a span that falls outside of
    /// the haystack being searched.
    SpanOutOfBounds {
        /// The span that falls outside of the haystack.
        span: Span,
        /// The length of the haystack.
        haystack_len: usize,
    },
}

#[cfg(feature = "std")]
impl std::error::Error for ReplaceError {}

impl core::fmt::Display for ReplaceError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match *self.kind() {
            ReplaceErrorKind::OverlappingMatches { first, second } => {
                write!(
                    f,
                    "overlapping matches are not supported when replacing: \
                     match at {}..{} overlaps with match at {}..{}",
                    first.start, first.end, second.start, second.end,
                )
            }
            ReplaceErrorKind::UnknownPatternID { pattern } => {
                write!(
                    f,
                    "pattern ID {} has no corresponding replacement",
                    pattern.as_usize(),
                )
            }
            ReplaceErrorKind::SpanOutOfBounds { span, haystack_len } => {
                write!(
                    f,
                    "match span {}..{} is out of bounds for haystack \
                     of length {}",
                    span.start, span.end, haystack_len,
                )
            }
        }
    }
}
