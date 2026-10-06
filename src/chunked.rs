/*!
Provides an explicit chunked overlapping matcher.

The [`ChunkedMatcher`] in this module implements *overlapping* Aho-Corasick
search over a haystack that arrives as a sequence of explicitly delimited
chunks (for example, network packets, file blocks or records pulled from a
channel). Unlike [`AhoCorasick::find_overlapping_iter`](crate::AhoCorasick),
which requires the entire haystack to be contiguous in memory, a
`ChunkedMatcher` is fed one chunk at a time via [`ChunkedMatcher::update`]
and reports every match---including matches that span a chunk boundary---with
its pattern ID and its *absolute* byte range in the logical haystack. Each
match is reported exactly once, and matches that overlap a chunk boundary are
never dropped.

The matcher works by carrying exactly one automaton state (plus an absolute
offset) between chunks. It never rescans previously seen bytes: the
Aho-Corasick state alone encodes every partial match suffix that is still
"open," so a match whose start lies in an earlier chunk is completed purely
by advancing the state machine. The start offset of a match is recovered
arithmetically from the absolute end offset and the pattern length, so no
history buffer is retained at all.

# Match semantics

Only [`MatchKind::Standard`] semantics are supported, since standard
semantics are the only semantics for which overlapping matches are defined.
Automata built with [`MatchKind::LeftmostFirst`] or
[`MatchKind::LeftmostLongest`] are explicitly rejected by
[`ChunkedMatcher::new`] and [`ChunkedMatcher::restore`] with
[`ChunkedError::UnsupportedMatchKind`], rather than silently producing
non-overlapping results.

# Instance isolation and recovery points

All working state lives inside the `ChunkedMatcher` value itself, so any
number of matchers may run concurrently (and independently) over the same
automaton. The state can be exported at any chunk boundary as a
[`RecoveryPoint`] via [`ChunkedMatcher::recovery_point`]. A recovery point
carries a *fingerprint* of the automaton it was derived from;
[`ChunkedMatcher::restore`] refuses to resume from a recovery point whose
fingerprint does not match the automaton given to it
([`ChunkedError::FingerprintMismatch`]). This makes it safe to persist
recovery points (e.g. for crash recovery or migration) without risking
resuming a stream against the wrong automaton.

# EOF handling

[`ChunkedMatcher::finish`] must be called when the end of the haystack is
reached. It flushes any trailing state (for example, matches of the empty
pattern at the final position) and consumes the matcher, making it
impossible to accidentally continue feeding chunks after EOF.

# Example: a match spanning three chunks

Patterns `ab` and `bab` matched against the haystack `bab`, delivered as the
three chunks `b`, `a` and `b`. Both overlapping matches---including the two
matches that only complete when the final chunk arrives---are reported with
their absolute offsets:

```
use aho_corasick::{
    chunked::ChunkedMatcher,
    nfa::contiguous::NFA,
};

let aut = NFA::new(["ab", "bab"]).unwrap();
let mut matcher = ChunkedMatcher::new(&aut).unwrap();

let mut matches = vec![];
for chunk in [&b"b"[..], &b"a"[..], &b"b"[..]] {
    matches.extend(matcher.update(chunk));
}
matches.extend(matcher.finish()); // EOF: flush trailing state.

let got: Vec<(usize, usize, usize)> = matches
    .iter()
    .map(|m| (m.pattern().as_usize(), m.start(), m.end()))
    .collect();
assert_eq!(got, vec![(1, 0, 3), (0, 1, 3)]);
```

# Example: crash recovery bound to an automaton fingerprint

```
use aho_corasick::{
    chunked::{ChunkedError, ChunkedMatcher},
    nfa::contiguous::NFA,
};

let aut = NFA::new(["ab", "bab"]).unwrap();
let mut matcher = ChunkedMatcher::new(&aut).unwrap();
assert!(matcher.update(b"b").is_empty());

// Export a recovery point at a chunk boundary.
let point = matcher.recovery_point();

// Resuming with the same automaton works and continues exactly where
// the original matcher left off.
let mut restored = ChunkedMatcher::restore(&aut, point).unwrap();
restored.update(b"a");
let matches = restored.update(b"b");
assert_eq!(matches.len(), 2); // "bab" at 0..3 and "ab" at 1..3

// Resuming with a *different* automaton is rejected.
let other = NFA::new(["xy"]).unwrap();
assert_eq!(
    ChunkedMatcher::restore(&other, point).unwrap_err(),
    ChunkedError::FingerprintMismatch {
        expected: point.fingerprint(),
        actual: ChunkedMatcher::fingerprint_of(&other),
    },
);
```

# Testing

This module ships with unit tests, including a differential test that
compares chunked matching against whole-haystack overlapping search over
randomly generated patterns, haystacks and chunk splits. Run them with:

```text
cargo test chunked
cargo test --doc chunked
```
*/

use alloc::vec::Vec;

use core::fmt;

use crate::{
    automaton::Automaton,
    util::{
        primitives::{PatternID, StateID},
        search::{Anchored, Match, MatchKind},
    },
};

/// The version of the recovery point format. This is mixed into every
/// automaton fingerprint so that recovery points produced by different
/// versions of this module never compare equal.
const RECOVERY_FORMAT_VERSION: u8 = 1;

/// An explicit chunked overlapping matcher.
///
/// A `ChunkedMatcher` borrows an [`Automaton`] and consumes a haystack one
/// chunk at a time, reporting every overlapping match with its pattern ID
/// and absolute byte range. See the [module documentation](self) for details
/// and examples.
///
/// All mutable working state is stored in this value, so distinct matcher
/// instances are fully isolated from one another even when they share the
/// same automaton.
#[derive(Debug)]
pub struct ChunkedMatcher<'a, A: Automaton + ?Sized> {
    /// The automaton being searched with.
    aut: &'a A,
    /// The current automaton state. This, together with `offset`, is the
    /// *entire* working state carried between chunks.
    sid: StateID,
    /// The absolute offset in the logical haystack of the next byte to be
    /// consumed. Equivalently, the number of bytes consumed so far.
    offset: usize,
    /// Whether the very first position (offset 0) has been processed yet.
    /// This is tracked separately so that matches ending at offset 0 (i.e.
    /// empty matches at the start of the haystack) are reported exactly
    /// once, and so that a recovery point exported before any `update` call
    /// still reproduces them upon restore.
    started: bool,
}

impl<'a, A: Automaton + ?Sized> ChunkedMatcher<'a, A> {
    /// Create a new chunked overlapping matcher over `aut`, positioned at
    /// the start of a fresh logical haystack.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkedError::UnsupportedMatchKind`] if `aut` uses leftmost
    /// match semantics. Overlapping search is only defined for
    /// [`MatchKind::Standard`] semantics, and this matcher refuses to
    /// silently degrade to non-overlapping behavior.
    ///
    /// # Panics
    ///
    /// Panics if the automaton fails to return a start state for unanchored
    /// search. This cannot happen for any automaton implementation provided
    /// by this crate.
    pub fn new(aut: &'a A) -> Result<ChunkedMatcher<'a, A>, ChunkedError> {
        if aut.match_kind() != MatchKind::Standard {
            return Err(ChunkedError::UnsupportedMatchKind {
                kind: aut.match_kind(),
            });
        }
        let sid = match aut.start_state(Anchored::No) {
            Ok(sid) => sid,
            Err(err) => {
                panic!("start_state(Anchored::No) failed: {:?}", err)
            }
        };
        Ok(ChunkedMatcher { aut, sid, offset: 0, started: false })
    }

    /// Feed the next chunk of the haystack and return every match that
    /// completed while consuming it.
    ///
    /// Matches are returned with absolute byte ranges (i.e. offsets into
    /// the logical haystack formed by concatenating all chunks), not
    /// chunk-relative offsets. Matches that begin in an earlier chunk and
    /// end in this one are reported here, exactly once. An empty chunk is
    /// permitted and is a no-op (aside from reporting matches at offset 0
    /// on the very first call).
    ///
    /// This routine walks the automaton state machine byte by byte and
    /// never looks at---let alone rescans---any previously consumed chunk.
    pub fn update(&mut self, chunk: &[u8]) -> Vec<Match> {
        let mut matches = Vec::new();
        self.ensure_started(&mut matches);
        for &byte in chunk {
            self.sid = self.aut.next_state(Anchored::No, self.sid, byte);
            self.offset += 1;
            self.emit_matches_at(&mut matches, self.offset);
        }
        matches
    }

    /// Signal end-of-file and flush any trailing state.
    ///
    /// Under standard semantics every non-empty match is reported eagerly
    /// as soon as its final byte is consumed, so the only matches that can
    /// remain at EOF are those that end at the final position without any
    /// byte being consumed after them---concretely, matches of the empty
    /// pattern at offset 0 when the haystack was empty (or when `update`
    /// was never called). Those are reported here.
    ///
    /// This consumes the matcher, so no further chunks can be fed after
    /// EOF. A recovery point for the final position may still be exported
    /// via [`ChunkedMatcher::recovery_point`] *before* calling `finish`.
    pub fn finish(mut self) -> Vec<Match> {
        let mut matches = Vec::new();
        self.ensure_started(&mut matches);
        matches
    }

    /// Export the current working state as a [`RecoveryPoint`].
    ///
    /// The returned value is a small `Copy`able token that can be persisted
    /// or transferred elsewhere, and later resumed with
    /// [`ChunkedMatcher::restore`]. It is bound to this matcher's automaton
    /// by a fingerprint; restoring it against a different automaton fails.
    ///
    /// For a resumed matcher to produce identical results, the recovery
    /// point must be exported at a chunk boundary (i.e. between `update`
    /// calls), and the remaining chunks must be fed in the same order.
    pub fn recovery_point(&self) -> RecoveryPoint {
        RecoveryPoint {
            fingerprint: self.fingerprint(),
            state_id: self.sid.as_u64(),
            offset: self.offset as u64,
            started: self.started,
        }
    }

    /// Resume a chunked search from a previously exported [`RecoveryPoint`].
    ///
    /// # Errors
    ///
    /// * [`ChunkedError::UnsupportedMatchKind`] if `aut` uses leftmost
    ///   match semantics.
    /// * [`ChunkedError::FingerprintMismatch`] if `point` was exported from
    ///   a matcher over a different automaton. The fingerprint covers the
    ///   match kind, the number and lengths of the patterns, the
    ///   automaton's memory usage and its full transition graph, so
    ///   automata built from different pattern sets (or with different
    ///   configurations) are reliably told apart.
    /// * [`ChunkedError::InvalidStateID`] if the state ID recorded in
    ///   `point` is not representable.
    /// * [`ChunkedError::OffsetOverflow`] if the absolute offset recorded
    ///   in `point` does not fit in a `usize` on this target.
    ///
    /// # Panics
    ///
    /// A recovery point is only meaningful for the automaton it was
    /// exported from. The fingerprint check guards against accidental
    /// misuse, but fingerprints are 64-bit hashes: deliberately forging a
    /// recovery point (or a hash collision) can lead to panics or silently
    /// incorrect results later, exactly as documented for invalid state IDs
    /// on the [`Automaton`] trait.
    pub fn restore(
        aut: &'a A,
        point: RecoveryPoint,
    ) -> Result<ChunkedMatcher<'a, A>, ChunkedError> {
        if aut.match_kind() != MatchKind::Standard {
            return Err(ChunkedError::UnsupportedMatchKind {
                kind: aut.match_kind(),
            });
        }
        let actual = ChunkedMatcher::fingerprint_of(aut);
        if point.fingerprint != actual {
            return Err(ChunkedError::FingerprintMismatch {
                expected: point.fingerprint,
                actual,
            });
        }
        let raw = usize::try_from(point.state_id).map_err(|_| {
            ChunkedError::InvalidStateID { id: point.state_id }
        })?;
        let sid = StateID::new(raw).map_err(|_| {
            ChunkedError::InvalidStateID { id: point.state_id }
        })?;
        let offset = usize::try_from(point.offset).map_err(|_| {
            ChunkedError::OffsetOverflow { offset: point.offset }
        })?;
        Ok(ChunkedMatcher { aut, sid, offset, started: point.started })
    }

    /// Return the fingerprint that binds recovery points to this matcher's
    /// automaton.
    pub fn fingerprint(&self) -> u64 {
        ChunkedMatcher::fingerprint_of(self.aut)
    }

    /// Compute the fingerprint of an automaton.
    ///
    /// The fingerprint is a 64-bit FNV-1a hash over the automaton's
    /// externally visible identity: the recovery format version, the match
    /// kind, the pattern count, the length of every pattern (in pattern ID
    /// order), the heap memory usage and---most importantly---the complete
    /// unanchored transition graph of the automaton (reached via a
    /// breadth-first walk from the start state, hashing every state's ID,
    /// its dead/match/start flags, the pattern IDs it matches and all 256
    /// of its outgoing transitions). Since the transition graph is fully
    /// determined by the patterns and the build configuration, two automata
    /// built from different pattern sets---even ones with identical pattern
    /// lengths, like `ab`/`bab` versus `xy`/`yxy`---have (with overwhelming
    /// probability) distinct fingerprints.
    ///
    /// This is an `O(states * alphabet)` computation and is only performed
    /// when exporting or restoring a recovery point, never while feeding
    /// chunks.
    pub fn fingerprint_of(aut: &A) -> u64 {
        use alloc::collections::BTreeSet;

        // FNV-1a over a canonical byte encoding of the automaton's
        // externally visible identity.
        fn fnv1a(mut h: u64, byte: u8) -> u64 {
            h ^= u64::from(byte);
            h.wrapping_mul(0x0000_0100_0000_01B3)
        }
        fn fnv1a_u64(mut h: u64, v: u64) -> u64 {
            for b in v.to_le_bytes() {
                h = fnv1a(h, b);
            }
            h
        }
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        h = fnv1a(h, RECOVERY_FORMAT_VERSION);
        h = fnv1a(
            h,
            match aut.match_kind() {
                MatchKind::Standard => 0,
                MatchKind::LeftmostFirst => 1,
                MatchKind::LeftmostLongest => 2,
            },
        );
        h = fnv1a_u64(h, aut.patterns_len() as u64);
        for i in 0..aut.patterns_len() {
            let pid = PatternID::must(i);
            h = fnv1a_u64(h, aut.pattern_len(pid) as u64);
        }
        h = fnv1a_u64(h, aut.min_pattern_len() as u64);
        h = fnv1a_u64(h, aut.max_pattern_len() as u64);
        h = fnv1a_u64(h, aut.memory_usage() as u64);
        // Hash the complete reachable transition graph. This is what ties
        // the fingerprint to the actual pattern *bytes*, which are not
        // otherwise observable through the `Automaton` trait.
        let start = match aut.start_state(Anchored::No) {
            Ok(sid) => sid,
            Err(err) => {
                panic!("start_state(Anchored::No) failed: {:?}", err)
            }
        };
        let mut seen = BTreeSet::new();
        let mut stack = alloc::vec![start];
        while let Some(sid) = stack.pop() {
            if !seen.insert(sid) {
                continue;
            }
            h = fnv1a_u64(h, sid.as_u64());
            h = fnv1a(h, aut.is_dead(sid) as u8);
            h = fnv1a(h, aut.is_match(sid) as u8);
            h = fnv1a(h, aut.is_start(sid) as u8);
            // Calling `match_len` or `match_pattern` on a non-match state
            // has unspecified behavior, so only inspect the patterns of
            // states that are actually match states.
            if aut.is_match(sid) {
                h = fnv1a_u64(h, aut.match_len(sid) as u64);
                for i in 0..aut.match_len(sid) {
                    h = fnv1a_u64(h, aut.match_pattern(sid, i).as_u64());
                }
            } else {
                h = fnv1a_u64(h, 0);
            }
            for byte in 0..=255u8 {
                let next = aut.next_state(Anchored::No, sid, byte);
                h = fnv1a_u64(h, next.as_u64());
                if !seen.contains(&next) {
                    stack.push(next);
                }
            }
        }
        h
    }

    /// Return the absolute offset of the next byte to be consumed, i.e. the
    /// total number of haystack bytes consumed so far.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Return the matcher's current automaton state.
    pub fn state_id(&self) -> StateID {
        self.sid
    }

    /// Process the initial position (offset 0) if it hasn't been yet.
    ///
    /// This is where matches that end at offset 0---i.e. empty-pattern
    /// matches at the very start of the haystack---are reported, exactly
    /// once per logical haystack.
    fn ensure_started(&mut self, matches: &mut Vec<Match>) {
        if self.started {
            return;
        }
        self.started = true;
        self.emit_matches_at(matches, self.offset);
    }

    /// If the current state is a match state, append one [`Match`] per
    /// pattern matched by it, each ending at absolute offset `end`.
    fn emit_matches_at(&self, matches: &mut Vec<Match>, end: usize) {
        if !self.aut.is_match(self.sid) {
            return;
        }
        for i in 0..self.aut.match_len(self.sid) {
            let pid = self.aut.match_pattern(self.sid, i);
            let len = self.aut.pattern_len(pid);
            matches.push(Match::new(pid, (end - len)..end));
        }
    }
}

/// An exported recovery point for a [`ChunkedMatcher`].
///
/// A recovery point captures the complete working state of a chunked
/// matcher at a chunk boundary: the automaton state, the absolute offset
/// and whether the initial position has been processed. It is bound to the
/// automaton it was exported from via a fingerprint (see
/// [`ChunkedMatcher::restore`]).
///
/// The value is `Copy` and its representation is fully accessible via its
/// accessors, so it can be serialized by callers (e.g. as three integers
/// and a flag) and reconstructed with [`RecoveryPoint::from_parts`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RecoveryPoint {
    /// Fingerprint of the automaton this recovery point belongs to.
    fingerprint: u64,
    /// The automaton state, as a `u64`.
    state_id: u64,
    /// The absolute offset of the next byte to consume.
    offset: u64,
    /// Whether the initial position (offset 0) has been processed.
    started: bool,
}

impl RecoveryPoint {
    /// Rebuild a recovery point from its raw parts.
    ///
    /// This is the inverse of the accessors [`RecoveryPoint::fingerprint`],
    /// [`RecoveryPoint::state_id`], [`RecoveryPoint::offset`] and
    /// [`RecoveryPoint::is_started`], and exists so that recovery points
    /// can be round-tripped through caller-chosen serialization formats.
    /// Prefer [`ChunkedMatcher::recovery_point`] over hand-constructing
    /// values; [`ChunkedMatcher::restore`] validates what it can, but a
    /// semantically invalid recovery point can still cause panics or
    /// incorrect results.
    pub fn from_parts(
        fingerprint: u64,
        state_id: u64,
        offset: u64,
        started: bool,
    ) -> RecoveryPoint {
        RecoveryPoint { fingerprint, state_id, offset, started }
    }

    /// The fingerprint of the automaton this recovery point is bound to.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The automaton state at the point of export.
    pub fn state_id(&self) -> u64 {
        self.state_id
    }

    /// The absolute haystack offset of the next byte to consume.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Whether the initial position had been processed at export time.
    pub fn is_started(&self) -> bool {
        self.started
    }
}

/// An error that can occur when creating or restoring a [`ChunkedMatcher`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ChunkedError {
    /// The automaton uses leftmost match semantics, which do not support
    /// overlapping matches. Only [`MatchKind::Standard`] is accepted.
    UnsupportedMatchKind {
        /// The offending match kind.
        kind: MatchKind,
    },
    /// A recovery point was presented for resumption together with an
    /// automaton whose fingerprint does not match the one recorded in the
    /// recovery point.
    FingerprintMismatch {
        /// The fingerprint recorded in the recovery point.
        expected: u64,
        /// The fingerprint of the automaton given to
        /// [`ChunkedMatcher::restore`].
        actual: u64,
    },
    /// A recovery point contained a state ID that is not representable.
    InvalidStateID {
        /// The offending state ID.
        id: u64,
    },
    /// A recovery point contained an absolute offset that does not fit in
    /// a `usize` on this target.
    OffsetOverflow {
        /// The offending offset.
        offset: u64,
    },
}

impl fmt::Display for ChunkedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ChunkedError::UnsupportedMatchKind { kind } => write!(
                f,
                "chunked overlapping search requires MatchKind::Standard, \
                 but automaton uses {:?}",
                kind,
            ),
            ChunkedError::FingerprintMismatch { expected, actual } => write!(
                f,
                "recovery point belongs to a different automaton \
                 (expected fingerprint {:#018x}, got {:#018x})",
                expected, actual,
            ),
            ChunkedError::InvalidStateID { id } => {
                write!(f, "recovery point has invalid state ID: {}", id)
            }
            ChunkedError::OffsetOverflow { offset } => write!(
                f,
                "recovery point offset {} overflows usize on this target",
                offset,
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ChunkedError {}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;
    use crate::{nfa::contiguous::NFA, Input};

    /// Collect every overlapping match of `aut` on the whole `haystack`
    /// using the crate's own (contiguous-haystack) search. This is the
    /// oracle that chunked matching is compared against.
    fn oracle<A: Automaton>(aut: &A, haystack: &[u8]) -> Vec<Match> {
        aut.try_find_overlapping_iter(Input::new(haystack)).unwrap().collect()
    }

    /// Run `chunks` through a chunked matcher, one chunk at a time, and
    /// collect every reported match (including EOF flushing).
    fn chunked<A: Automaton>(aut: &A, chunks: &[&[u8]]) -> Vec<Match> {
        let mut matcher = ChunkedMatcher::new(aut).unwrap();
        let mut all = vec![];
        for chunk in chunks {
            all.extend(matcher.update(chunk));
        }
        all.extend(matcher.finish());
        all
    }

    fn as_tuples(matches: &[Match]) -> Vec<(usize, usize, usize)> {
        matches
            .iter()
            .map(|m| (m.pattern().as_usize(), m.start(), m.end()))
            .collect()
    }

    #[test]
    fn cross_chunk_overlap_three_chunks() {
        // The acceptance case: patterns "ab" and "bab", haystack "bab"
        // delivered as b|a|b. Both overlapping matches complete only when
        // the final chunk arrives, and both must be reported exactly once
        // with absolute offsets.
        let aut = NFA::new(["ab", "bab"]).unwrap();
        let mut matcher = ChunkedMatcher::new(&aut).unwrap();
        assert_eq!(matcher.update(b"b"), vec![]);
        assert_eq!(matcher.update(b"a"), vec![]);
        let matches = matcher.update(b"b");
        assert_eq!(as_tuples(&matches), vec![(1, 0, 3), (0, 1, 3)]);
        assert_eq!(matcher.finish(), vec![]);
        // And it must agree with the whole-haystack oracle.
        assert_eq!(as_tuples(&matches), as_tuples(&oracle(&aut, b"bab")));
    }

    #[test]
    fn overlap_not_dropped_at_chunk_end() {
        // A match ending exactly at a chunk boundary must be reported by
        // that chunk's `update` call, not dropped or deferred.
        let aut = NFA::new(["ab", "bab"]).unwrap();
        let matches = chunked(&aut, &[b"ba", b"b"]);
        assert_eq!(as_tuples(&matches), vec![(1, 0, 3), (0, 1, 3)]);
        assert_eq!(as_tuples(&matches), as_tuples(&oracle(&aut, b"bab")));

        // Same, but with the match ending at the end of the first of two
        // chunks, followed by more non-matching input.
        let aut = NFA::new(["ab"]).unwrap();
        let matches = chunked(&aut, &[b"xxab", b"zz"]);
        assert_eq!(as_tuples(&matches), vec![(0, 2, 4)]);
    }

    #[test]
    fn absolute_offsets_across_chunks() {
        let aut = NFA::new(["ab"]).unwrap();
        let matches = chunked(&aut, &[b"xx", b"a", b"bxx", b"ab"]);
        assert_eq!(as_tuples(&matches), vec![(0, 2, 4), (0, 6, 8)]);
    }

    #[test]
    fn leftmost_match_kinds_rejected() {
        for kind in [MatchKind::LeftmostFirst, MatchKind::LeftmostLongest] {
            let aut =
                NFA::builder().match_kind(kind).build(["ab", "bab"]).unwrap();
            assert_eq!(
                ChunkedMatcher::new(&aut).unwrap_err(),
                ChunkedError::UnsupportedMatchKind { kind },
            );
            // Restore is rejected too, even with a fingerprint-matching
            // recovery point from a standard automaton.
            let standard = NFA::new(["ab", "bab"]).unwrap();
            let point =
                ChunkedMatcher::new(&standard).unwrap().recovery_point();
            assert_eq!(
                ChunkedMatcher::restore(&aut, point).unwrap_err(),
                ChunkedError::UnsupportedMatchKind { kind },
            );
        }
    }

    #[test]
    fn restore_with_wrong_automaton_rejected() {
        let aut = NFA::new(["ab", "bab"]).unwrap();
        let mut matcher = ChunkedMatcher::new(&aut).unwrap();
        matcher.update(b"b");
        let point = matcher.recovery_point();

        // Same automaton: accepted.
        assert!(ChunkedMatcher::restore(&aut, point).is_ok());

        // Different patterns: rejected.
        let other = NFA::new(["xy"]).unwrap();
        assert_eq!(
            ChunkedMatcher::restore(&other, point).unwrap_err(),
            ChunkedError::FingerprintMismatch {
                expected: point.fingerprint(),
                actual: ChunkedMatcher::fingerprint_of(&other),
            },
        );

        // Same pattern lengths but different pattern bytes: still rejected
        // (the automaton's memory usage differs because its transitions
        // differ).
        let other = NFA::new(["xy", "yxy"]).unwrap();
        assert!(matches!(
            ChunkedMatcher::restore(&other, point),
            Err(ChunkedError::FingerprintMismatch { .. }),
        ));
    }

    #[test]
    fn restore_roundtrip_continues_exactly() {
        let aut = NFA::new(["ab", "bab"]).unwrap();
        let haystack = b"bbababbab";

        // Reference: one uninterrupted chunked run, one byte per chunk.
        let reference: Vec<Match> = {
            let mut m = ChunkedMatcher::new(&aut).unwrap();
            let mut all = vec![];
            for &b in haystack {
                all.extend(m.update(&[b]));
            }
            all.extend(m.finish());
            all
        };
        assert_eq!(as_tuples(&reference), as_tuples(&oracle(&aut, haystack)));

        // Now: run halfway, export a recovery point, restore into a fresh
        // matcher and finish the run. The concatenation must be identical.
        let split = 4;
        let mut first = ChunkedMatcher::new(&aut).unwrap();
        let mut all = vec![];
        for &b in &haystack[..split] {
            all.extend(first.update(&[b]));
        }
        let point = first.recovery_point();
        assert_eq!(point.offset(), split as u64);
        let mut second = ChunkedMatcher::restore(&aut, point).unwrap();
        for &b in &haystack[split..] {
            all.extend(second.update(&[b]));
        }
        all.extend(second.finish());
        assert_eq!(all, reference);
    }

    #[test]
    fn instances_are_isolated() {
        // Two matchers over the same automaton advance independently.
        let aut = NFA::new(["ab", "bab"]).unwrap();
        let mut m1 = ChunkedMatcher::new(&aut).unwrap();
        let mut m2 = ChunkedMatcher::new(&aut).unwrap();

        assert_eq!(m1.update(b"b"), vec![]);
        // m2 sees a full match in one chunk while m1 is mid-match.
        assert_eq!(as_tuples(&m2.update(b"ab")), vec![(0, 0, 2)]);
        // m1 is unaffected and still completes its own cross-chunk match.
        assert_eq!(as_tuples(&m1.update(b"ab")), vec![(1, 0, 3), (0, 1, 3)]);
        // m2 is unaffected by m1's progress. Its logical haystack is now
        // "abab", which contains "bab" at 1..4 and "ab" at 2..4.
        assert_eq!(as_tuples(&m2.update(b"ab")), vec![(1, 1, 4), (0, 2, 4)]);
    }

    #[test]
    fn eof_flushes_empty_pattern_at_start() {
        // With an empty pattern, the empty haystack still yields one match
        // at 0..0, reported by `finish` even if `update` was never called.
        let aut = NFA::new([""]).unwrap();
        let matcher = ChunkedMatcher::new(&aut).unwrap();
        let matches = matcher.finish();
        assert_eq!(as_tuples(&matches), vec![(0, 0, 0)]);
        assert_eq!(matches, oracle(&aut, b""));

        // And empty-pattern matches at every position agree with the
        // oracle when other patterns are in the mix too.
        let aut = NFA::new(["", "ab"]).unwrap();
        let matches = chunked(&aut, &[b"a", b"b", b"x"]);
        assert_eq!(matches, oracle(&aut, b"abx"));
    }

    #[test]
    fn empty_chunks_are_harmless() {
        let aut = NFA::new(["ab", "bab"]).unwrap();
        let matches = chunked(&aut, &[b"", b"b", b"", b"a", b"b", b""]);
        assert_eq!(as_tuples(&matches), vec![(1, 0, 3), (0, 1, 3)]);
    }

    #[test]
    fn works_with_every_automaton_kind() {
        // The matcher is generic over the `Automaton` trait: the acceptance
        // scenario must behave identically for the noncontiguous NFA, the
        // contiguous NFA and the DFA, including recovery/restore.
        use crate::{dfa, nfa::noncontiguous};

        let chunks: &[&[u8]] = &[b"b", b"a", b"b"];
        let expected = vec![(1, 0, 3), (0, 1, 3)];

        let noncontig = noncontiguous::NFA::new(["ab", "bab"]).unwrap();
        assert_eq!(as_tuples(&chunked(&noncontig, chunks)), expected);

        let contig = NFA::new(["ab", "bab"]).unwrap();
        assert_eq!(as_tuples(&chunked(&contig, chunks)), expected);

        let dfa = dfa::DFA::new(["ab", "bab"]).unwrap();
        assert_eq!(as_tuples(&chunked(&dfa, chunks)), expected);

        // Recovery points are automaton-specific: a point exported from
        // the NFA must not restore against the DFA even though both were
        // built from the same patterns.
        let mut m = ChunkedMatcher::new(&contig).unwrap();
        m.update(b"b");
        let point = m.recovery_point();
        assert!(matches!(
            ChunkedMatcher::restore(&dfa, point),
            Err(ChunkedError::FingerprintMismatch { .. }),
        ));
        // But it restores fine against an identically built NFA.
        let contig2 = NFA::new(["ab", "bab"]).unwrap();
        let mut m = ChunkedMatcher::restore(&contig2, point).unwrap();
        m.update(b"a");
        assert_eq!(as_tuples(&m.update(b"b")), expected);
    }

    /// A tiny deterministic xorshift RNG so the differential test needs no
    /// external dependencies and is fully reproducible.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    fn differential_against_whole_haystack() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for case in 0..200 {
            // Random patterns over a tiny alphabet (lots of overlaps).
            let npatterns = 1 + rng.below(4);
            let mut patterns: Vec<Vec<u8>> = vec![];
            for _ in 0..npatterns {
                let len = 1 + rng.below(3);
                let mut p = vec![];
                for _ in 0..len {
                    p.push(b"ab"[rng.below(2)]);
                }
                patterns.push(p);
            }
            // Random haystack.
            let haystack: Vec<u8> =
                (0..rng.below(40)).map(|_| b"ab"[rng.below(2)]).collect();
            // Random chunk split.
            let mut chunks: Vec<&[u8]> = vec![];
            let mut rest = &haystack[..];
            while !rest.is_empty() {
                let n = 1 + rng.below(rest.len());
                chunks.push(&rest[..n]);
                rest = &rest[n..];
            }

            let aut = NFA::new(&patterns).unwrap();
            let expected = oracle(&aut, &haystack);

            // Plain chunked run.
            let got = chunked(&aut, &chunks);
            assert_eq!(
                got, expected,
                "case {}: patterns={:?} haystack={:?} chunks={:?}",
                case, patterns, haystack, chunks,
            );

            // Chunked run with a recovery/restore cycle at every chunk
            // boundary. This must not change the results at all.
            if !chunks.is_empty() {
                let mut all = vec![];
                let mut matcher = ChunkedMatcher::new(&aut).unwrap();
                for chunk in &chunks {
                    all.extend(matcher.update(chunk));
                    let point = matcher.recovery_point();
                    matcher = ChunkedMatcher::restore(&aut, point).unwrap();
                }
                all.extend(matcher.finish());
                assert_eq!(
                    all, expected,
                    "case {} (with restore): patterns={:?} haystack={:?}",
                    case, patterns, haystack,
                );
            }
        }
    }
}
