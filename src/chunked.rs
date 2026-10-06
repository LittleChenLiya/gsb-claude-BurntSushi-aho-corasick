/*!
An explicit chunked overlapping matcher.

This module provides [`ChunkedOverlappingMatcher`], a small stateful scanner
built on top of [`AhoCorasick`] that finds **all overlapping** matches while
the haystack is delivered as a sequence of explicit chunks (for example,
packets, lines or buffered reads). Its key properties are:

* Matches that span a chunk boundary are reported exactly once, with their
  pattern ID and **absolute** byte offsets (relative to the start of the
  stream, not the start of a chunk).
* Only a bounded suffix of the history is retained between chunks (at most
  `2 * (max_pattern_len - 1)` bytes). The history is never rescanned in
  full.
* Only [`MatchKind::Standard`] is supported. Leftmost match semantics
  cannot be combined with this incremental overlapping scan, so they are
  rejected explicitly at construction time.
* End-of-stream is signalled explicitly via
  [`ChunkedCursor::finish`], which flushes matches that end in the final
  `max_pattern_len - 1` bytes (those are deliberately deferred while more
  chunks might still arrive, so that an overlapping match is never lost at
  a chunk boundary).
* Working state is isolated per [`ChunkedCursor`] instance. Any number of
  cursors may scan independent chunk sequences with the same matcher.
* A cursor can export a [`ChunkCheckpoint`], an opaque recovery point bound
  to the automaton fingerprint. Resuming a checkpoint with a matcher
  built from different patterns is rejected.

# Example: three one-byte chunks

Patterns `ab` and `bab` over the stream `b|a|b` (fed as three chunks)
yield two cross-chunk hits, `bab` at `0..3` and `ab` at `1..3`:

```
use aho_corasick::chunked::ChunkedOverlappingMatcher;

let matcher = ChunkedOverlappingMatcher::new(["ab", "bab"]).unwrap();
let mut cur = matcher.start();
let mut hits = vec![];
for chunk in [b"b".as_slice(), b"a", b"b"] {
    hits.extend(cur.feed(chunk).unwrap());
}
hits.extend(cur.finish().unwrap());
hits.sort_by_key(|m| (m.start(), m.end()));
assert_eq!(hits.len(), 2);
let first = (hits[0].pattern().as_u32(), hits[0].start(), hits[0].end());
let second = (hits[1].pattern().as_u32(), hits[1].start(), hits[1].end());
assert_eq!(first, (1, 0, 3));
assert_eq!(second, (0, 1, 3));
```
*/

use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use crate::{
    ahocorasick::AhoCorasick, util::primitives::PatternID, MatchKind,
};

/// A match reported by a chunked cursor, with absolute stream offsets.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChunkMatch {
    pattern: PatternID,
    start: u64,
    end: u64,
}

impl ChunkMatch {
    /// The ID of the pattern that matched.
    #[inline]
    pub fn pattern(&self) -> PatternID {
        self.pattern
    }

    /// The absolute start byte offset of the match in the stream.
    #[inline]
    pub fn start(&self) -> u64 {
        self.start
    }

    /// The absolute end byte offset of the match in the stream.
    #[inline]
    pub fn end(&self) -> u64 {
        self.end
    }
}

/// Errors returned by the chunked overlapping matcher.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ChunkedError {
    /// The automaton could not be built from the given patterns.
    Build {
        /// The underlying build error message.
        message: String,
    },
    /// The match semantics are not supported by this algorithm.
    ///
    /// Only [`MatchKind::Standard`] may be used: leftmost semantics cannot
    /// be combined with an incremental overlapping scan, so they are
    /// rejected explicitly instead of silently misbehaving.
    UnsupportedMatchKind {
        /// The match kind that was rejected.
        kind: MatchKind,
    },
    /// A checkpoint was resumed with a matcher whose automaton does not
    /// match the fingerprint recorded in the checkpoint.
    AutomatonMismatch {
        /// Fingerprint recorded in the checkpoint.
        expected: u64,
        /// Fingerprint of the matcher used to resume.
        actual: u64,
    },
    /// A checkpoint is malformed or internally inconsistent.
    CorruptCheckpoint {
        /// A human readable description of the problem.
        message: String,
    },
    /// A chunk was fed after the cursor was finished.
    AlreadyFinished,
}

impl core::fmt::Display for ChunkedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            ChunkedError::Build { ref message } => {
                write!(f, "failed to build automaton: {}", message)
            }
            ChunkedError::UnsupportedMatchKind { kind } => write!(
                f,
                "match kind {:?} is not supported by the chunked \
                 overlapping matcher; only MatchKind::Standard is allowed",
                kind,
            ),
            ChunkedError::AutomatonMismatch { expected, actual } => write!(
                f,
                "checkpoint belongs to a different automaton \
                 (checkpoint fingerprint {:#018x}, matcher fingerprint \
                 {:#018x})",
                expected, actual,
            ),
            ChunkedError::CorruptCheckpoint { ref message } => {
                write!(f, "corrupt checkpoint: {}", message)
            }
            ChunkedError::AlreadyFinished => {
                write!(f, "cannot feed a chunk after finish")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ChunkedError {}

/// Compute a fingerprint binding a match kind and an ordered pattern list.
///
/// This is FNV-1a (64-bit) over the match kind, the pattern count and each
/// pattern length and bytes. It is deterministic across processes.
fn fingerprint(patterns: &[&[u8]], kind: MatchKind) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    let kind_tag = match kind {
        MatchKind::Standard => 0u8,
        MatchKind::LeftmostFirst => 1,
        MatchKind::LeftmostLongest => 2,
    };
    mix(kind_tag);
    mix(b'C'); // domain separator: chunked matcher fingerprint v1
    for &b in (patterns.len() as u64).to_le_bytes().iter() {
        mix(b);
    }
    for pat in patterns {
        for &b in (pat.len() as u64).to_le_bytes().iter() {
            mix(b);
        }
        for &b in pat.iter() {
            mix(b);
        }
    }
    hash
}

/// An explicit chunked overlapping matcher.
///
/// This owns the automaton and the immutable configuration shared by all
/// cursors. Create per-stream working state with
/// [`ChunkedOverlappingMatcher::start`] or restore it from a checkpoint with
/// [`ChunkedOverlappingMatcher::resume`].
#[derive(Clone, Debug)]
pub struct ChunkedOverlappingMatcher {
    ac: AhoCorasick,
    fingerprint: u64,
    max_pattern_len: usize,
}

impl ChunkedOverlappingMatcher {
    /// Build a matcher from patterns, forcing [`MatchKind::Standard`].
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying automaton cannot be built.
    pub fn new<I, P>(
        patterns: I,
    ) -> Result<ChunkedOverlappingMatcher, ChunkedError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let owned: Vec<P> = patterns.into_iter().collect();
        let pats: Vec<&[u8]> = owned.iter().map(|p| p.as_ref()).collect();
        let ac = crate::AhoCorasickBuilder::new()
            .match_kind(MatchKind::Standard)
            .build(&pats)
            .map_err(|e| ChunkedError::Build {
                message: e.to_string(),
            })?;
        ChunkedOverlappingMatcher::from_parts(ac, &pats)
    }

    /// Wrap an existing automaton together with the exact ordered pattern
    /// list it was built from.
    ///
    /// The patterns are required because the automaton does not expose
    /// them, and they are needed to compute the fingerprint that binds
    /// checkpoints to this automaton.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkedError::UnsupportedMatchKind`] unless the automaton
    /// uses [`MatchKind::Standard`]. Leftmost semantics cannot be combined
    /// with this incremental overlapping scan and are rejected explicitly.
    pub fn from_aho_corasick<I, P>(
        ac: AhoCorasick,
        patterns: I,
    ) -> Result<ChunkedOverlappingMatcher, ChunkedError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let owned: Vec<P> = patterns.into_iter().collect();
        let pats: Vec<&[u8]> = owned.iter().map(|p| p.as_ref()).collect();
        if pats.len() != ac.patterns_len() {
            return Err(ChunkedError::CorruptCheckpoint {
                message: alloc::format!(
                    "pattern list length {} does not match automaton \
                     pattern count {}",
                    pats.len(),
                    ac.patterns_len(),
                ),
            });
        }
        ChunkedOverlappingMatcher::from_parts(ac, &pats)
    }

    fn from_parts(
        ac: AhoCorasick,
        patterns: &[&[u8]],
    ) -> Result<ChunkedOverlappingMatcher, ChunkedError> {
        let kind = ac.match_kind();
        if kind != MatchKind::Standard {
            return Err(ChunkedError::UnsupportedMatchKind { kind });
        }
        Ok(ChunkedOverlappingMatcher {
            fingerprint: fingerprint(patterns, kind),
            max_pattern_len: ac.max_pattern_len(),
            ac,
        })
    }

    /// Return the fingerprint binding checkpoints to this automaton.
    #[inline]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Start a fresh cursor with its own isolated working state.
    #[inline]
    pub fn start(&self) -> ChunkedCursor<'_> {
        ChunkedCursor {
            matcher: self,
            offset: 0,
            reported: 0,
            tail: Vec::new(),
            finished: false,
        }
    }

    /// Resume a cursor from a checkpoint previously exported by
    /// [`ChunkedCursor::checkpoint`].
    ///
    /// # Errors
    ///
    /// Returns [`ChunkedError::AutomatonMismatch`] if the checkpoint was
    /// produced by a matcher with a different automaton fingerprint, and
    /// [`ChunkedError::CorruptCheckpoint`] if the checkpoint is internally
    /// inconsistent.
    pub fn resume(
        &self,
        cp: &ChunkCheckpoint,
    ) -> Result<ChunkedCursor<'_>, ChunkedError> {
        if cp.fingerprint != self.fingerprint {
            return Err(ChunkedError::AutomatonMismatch {
                expected: cp.fingerprint,
                actual: self.fingerprint,
            });
        }
        cp.validate(self.max_pattern_len)?;
        Ok(ChunkedCursor {
            matcher: self,
            offset: cp.offset,
            reported: cp.reported,
            tail: cp.tail.clone(),
            finished: false,
        })
    }
}

/// Per-instance working state of a chunked scan.
///
/// A cursor remembers only the absolute stream offset, a reported-up-to
/// watermark and a bounded tail of at most `2 * (max_pattern_len - 1)`
/// bytes, so the history is never rescanned in full. The tail covers both
/// the deferred reporting zone (the last `max_pattern_len - 1` bytes) and
/// the context those deferred matches may start in.
#[derive(Clone, Debug)]
pub struct ChunkedCursor<'m> {
    matcher: &'m ChunkedOverlappingMatcher,
    /// Absolute offset just past the last byte fed so far.
    offset: u64,
    /// All matches with absolute end at or below this have been reported.
    reported: u64,
    /// A bounded suffix of the bytes fed so far.
    tail: Vec<u8>,
    finished: bool,
}

impl<'m> ChunkedCursor<'m> {
    /// The deferral zone: matches ending in the last `keep` bytes are not
    /// finalized yet.
    fn keep(&self) -> usize {
        self.matcher.max_pattern_len.saturating_sub(1)
    }

    /// The maximum number of tail bytes retained between chunks.
    fn tail_capacity(&self) -> usize {
        2 * self.keep()
    }
    /// Return the absolute offset just past the last byte fed so far.
    #[inline]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Feed one chunk and return the matches that became final.
    ///
    /// A match is reported exactly once, when its absolute end offset is
    /// known to be final: matches ending in the last `max_pattern_len - 1`
    /// bytes seen so far are deferred, because they could also be produced
    /// from the retained tail when the next chunk arrives. Anything still
    /// deferred is flushed by [`ChunkedCursor::finish`].
    ///
    /// # Errors
    ///
    /// Returns [`ChunkedError::AlreadyFinished`] if called after
    /// [`ChunkedCursor::finish`].
    pub fn feed(
        &mut self,
        chunk: &[u8],
    ) -> Result<Vec<ChunkMatch>, ChunkedError> {
        if self.finished {
            return Err(ChunkedError::AlreadyFinished);
        }
        let keep = self.keep();
        let mut buf = Vec::with_capacity(self.tail.len() + chunk.len());
        buf.extend_from_slice(&self.tail);
        buf.extend_from_slice(chunk);
        let base = self.offset - self.tail.len() as u64;
        self.offset += chunk.len() as u64;
        // Matches ending at or before `safe` are fully contained in `buf`
        // and can never be produced again by a future scan (future scans
        // only see the retained tail), so it is safe to finalize them now.
        let safe = self.offset.saturating_sub(keep as u64);
        let out = self.scan(&buf, base, safe);
        self.reported = safe;
        let tail_start = buf.len().saturating_sub(self.tail_capacity());
        self.tail.clear();
        self.tail.extend_from_slice(&buf[tail_start..]);
        Ok(out)
    }

    /// Signal end-of-stream and flush matches ending in the final tail.
    ///
    /// After this call the cursor has reported every remaining match whose
    /// end lies in the last `max_pattern_len - 1` bytes of the stream, and
    /// it refuses any further chunks.
    pub fn finish(&mut self) -> Result<Vec<ChunkMatch>, ChunkedError> {
        if self.finished {
            return Err(ChunkedError::AlreadyFinished);
        }
        self.finished = true;
        let tail = self.tail.clone();
        let base = self.offset - tail.len() as u64;
        let out = self.scan(&tail, base, self.offset);
        self.reported = self.offset;
        Ok(out)
    }

    /// Export a recovery point bound to this matcher fingerprint.
    ///
    /// The checkpoint captures the absolute offset, the reported watermark
    /// and the retained tail, so a cursor resumed via
    /// [`ChunkedOverlappingMatcher::resume`] continues exactly where this
    /// one left off without rescanning any history.
    pub fn checkpoint(&self) -> ChunkCheckpoint {
        ChunkCheckpoint {
            fingerprint: self.matcher.fingerprint,
            offset: self.offset,
            reported: self.reported,
            tail: self.tail.clone(),
        }
    }

    /// Report matches in `buf` (whose first byte sits at absolute offset
    /// `base`) with absolute end in `(self.reported, safe]`.
    fn scan(&self, buf: &[u8], base: u64, safe: u64) -> Vec<ChunkMatch> {
        let mut out = Vec::new();
        for m in self.matcher.ac.find_overlapping_iter(buf) {
            let end = base + m.end() as u64;
            if end <= self.reported || end > safe {
                continue;
            }
            out.push(ChunkMatch {
                pattern: m.pattern(),
                start: base + m.start() as u64,
                end,
            });
        }
        out
    }
}

/// An exported recovery point for a chunked scan.
///
/// A checkpoint is bound to the fingerprint of the automaton that produced
/// it; resuming it with a matcher built from different patterns fails with
/// [`ChunkedError::AutomatonMismatch`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkCheckpoint {
    fingerprint: u64,
    offset: u64,
    reported: u64,
    tail: Vec<u8>,
}

impl ChunkCheckpoint {
    /// The fingerprint of the automaton this checkpoint is bound to.
    #[inline]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The absolute stream offset just past the last byte fed.
    #[inline]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The number of retained tail bytes (bounded by
    /// `2 * (max_pattern_len - 1)`).
    #[inline]
    pub fn tail_len(&self) -> usize {
        self.tail.len()
    }

    /// Serialize the checkpoint to a self-describing byte string.
    ///
    /// Format: magic `b"ACCK"`, version byte `1`, then little-endian
    /// `u64` fingerprint, offset and reported watermark, a `u32` tail
    /// length and the raw tail bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(4 + 1 + 8 * 3 + 4 + self.tail.len());
        out.extend_from_slice(b"ACCK");
        out.push(1);
        out.extend_from_slice(&self.fingerprint.to_le_bytes());
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.reported.to_le_bytes());
        out.extend_from_slice(&(self.tail.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.tail);
        out
    }

    /// Deserialize a checkpoint produced by [`ChunkCheckpoint::to_bytes`].
    ///
    /// Note this only checks structural integrity. Binding to an automaton
    /// is verified by [`ChunkedOverlappingMatcher::resume`].
    pub fn from_bytes(
        bytes: &[u8],
    ) -> Result<ChunkCheckpoint, ChunkedError> {
        let corrupt = |message: &str| ChunkedError::CorruptCheckpoint {
            message: message.to_string(),
        };
        if bytes.len() < 4 + 1 + 8 * 3 + 4 {
            return Err(corrupt("checkpoint too short"));
        }
        if &bytes[..4] != b"ACCK" {
            return Err(corrupt("bad magic"));
        }
        if bytes[4] != 1 {
            return Err(corrupt("unsupported version"));
        }
        let u64_at = |i: usize| {
            u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap())
        };
        let fingerprint = u64_at(5);
        let offset = u64_at(13);
        let reported = u64_at(21);
        let tail_len =
            u32::from_le_bytes(bytes[29..33].try_into().unwrap()) as usize;
        if bytes.len() != 33 + tail_len {
            return Err(corrupt("tail length mismatch"));
        }
        Ok(ChunkCheckpoint {
            fingerprint,
            offset,
            reported,
            tail: bytes[33..].to_vec(),
        })
    }

    fn validate(&self, max_pattern_len: usize) -> Result<(), ChunkedError> {
        let corrupt = |message: String| {
            ChunkedError::CorruptCheckpoint { message }
        };
        if self.reported > self.offset {
            return Err(corrupt(alloc::format!(
                "reported watermark {} exceeds offset {}",
                self.reported,
                self.offset,
            )));
        }
        let max_tail = 2 * max_pattern_len.saturating_sub(1);
        if self.tail.len() > max_tail {
            return Err(corrupt(alloc::format!(
                "tail of {} bytes exceeds maximum {}",
                self.tail.len(),
                max_tail,
            )));
        }
        if self.tail.len() as u64 > self.offset {
            return Err(corrupt(alloc::format!(
                "tail of {} bytes exceeds offset {}",
                self.tail.len(),
                self.offset,
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn collect(
        matcher: &ChunkedOverlappingMatcher,
        chunks: &[&[u8]],
    ) -> Vec<ChunkMatch> {
        let mut cur = matcher.start();
        let mut hits = vec![];
        for c in chunks {
            hits.extend(cur.feed(c).unwrap());
        }
        hits.extend(cur.finish().unwrap());
        hits.sort_by_key(|m| (m.start(), m.end(), m.pattern().as_u32()));
        hits
    }

    fn collect_once(
        patterns: &[&[u8]],
        chunks: &[&[u8]],
    ) -> Vec<ChunkMatch> {
        let matcher = ChunkedOverlappingMatcher::new(patterns).unwrap();
        collect(&matcher, chunks)
    }

    fn oneshot(patterns: &[&[u8]], haystack: &[u8]) -> Vec<ChunkMatch> {
        let ac = crate::AhoCorasickBuilder::new()
            .match_kind(MatchKind::Standard)
            .build(patterns)
            .unwrap();
        let mut hits: Vec<ChunkMatch> = ac
            .find_overlapping_iter(haystack)
            .map(|m| ChunkMatch {
                pattern: m.pattern(),
                start: m.start() as u64,
                end: m.end() as u64,
            })
            .collect();
        hits.sort_by_key(|m| (m.start(), m.end(), m.pattern().as_u32()));
        hits
    }

    #[test]
    fn three_single_byte_chunks_cross_boundary() {
        // Acceptance: patterns ab and bab, input b|a|b gives two
        // cross-chunk hits with accurate absolute offsets.
        let matcher = ChunkedOverlappingMatcher::new(["ab", "bab"]).unwrap();
        let hits = collect(&matcher, &[b"b", b"a", b"b"]);
        assert_eq!(hits.len(), 2);
        let first =
            (hits[0].pattern().as_u32(), hits[0].start(), hits[0].end());
        let second =
            (hits[1].pattern().as_u32(), hits[1].start(), hits[1].end());
        assert_eq!(first, (1, 0, 3), "bab at 0..3");
        assert_eq!(second, (0, 1, 3), "ab at 1..3");
    }

    #[test]
    fn overlap_at_chunk_end_is_not_dropped() {
        // "aa" overlapping itself across the chunk boundary: three hits.
        let matcher = ChunkedOverlappingMatcher::new(["aa"]).unwrap();
        let hits = collect(&matcher, &[b"aa", b"aa"]);
        let spans: Vec<(u64, u64)> =
            hits.iter().map(|m| (m.start(), m.end())).collect();
        assert_eq!(spans, vec![(0, 2), (1, 3), (2, 4)]);
    }

    #[test]
    fn matches_oneshot_for_many_chunkings() {
        let patterns: &[&[u8]] = &[b"ab", b"bab", b"a", b"aba"];
        let haystack: &[u8] = b"ababbabababba";
        let want = oneshot(patterns, haystack);
        // Every single split point, plus uneven multi-chunk splits.
        for split in 0..=haystack.len() {
            let got = collect_once(
                patterns,
                &[&haystack[..split], &haystack[split..]],
            );
            assert_eq!(got, want, "split at {}", split);
        }
        let got =
            collect_once(patterns, &[b"ab", b"abb", b"ab", b"ababba"]);
        assert_eq!(got, want);
        // Byte-at-a-time.
        let chunks: Vec<&[u8]> = haystack.chunks(1).collect();
        let got = collect_once(patterns, &chunks);
        assert_eq!(got, want);
    }

    #[test]
    fn leftmost_match_kinds_are_rejected() {
        for kind in [MatchKind::LeftmostFirst, MatchKind::LeftmostLongest]
        {
            let ac = crate::AhoCorasickBuilder::new()
                .match_kind(kind)
                .build(["ab", "bab"])
                .unwrap();
            let err = ChunkedOverlappingMatcher::from_aho_corasick(
                ac,
                [&b"ab"[..], &b"bab"[..]],
            )
            .unwrap_err();
            assert_eq!(err, ChunkedError::UnsupportedMatchKind { kind });
        }
    }

    #[test]
    fn wrong_automaton_resume_is_rejected() {
        let m1 = ChunkedOverlappingMatcher::new(["ab", "bab"]).unwrap();
        let m2 = ChunkedOverlappingMatcher::new(["xy", "bab"]).unwrap();
        let mut cur = m1.start();
        cur.feed(b"ba").unwrap();
        let cp = cur.checkpoint();
        // Same automaton: resumes fine.
        assert!(m1.resume(&cp).is_ok());
        // Different automaton: rejected.
        let err = m2.resume(&cp).unwrap_err();
        assert_eq!(
            err,
            ChunkedError::AutomatonMismatch {
                expected: m1.fingerprint(),
                actual: m2.fingerprint(),
            }
        );
    }

    #[test]
    fn checkpoint_resume_continues_exactly() {
        let patterns: &[&[u8]] = &[b"ab", b"bab"];
        let haystack: &[u8] = b"xbabbab";
        let matcher = ChunkedOverlappingMatcher::new(patterns).unwrap();
        let want = collect(&matcher, &[haystack]);

        let mut cur = matcher.start();
        let mut got = cur.feed(b"xba").unwrap();
        let cp = cur.checkpoint();
        // Round-trip through the serialized form too.
        let cp = ChunkCheckpoint::from_bytes(&cp.to_bytes()).unwrap();
        let mut resumed = matcher.resume(&cp).unwrap();
        got.extend(resumed.feed(b"bbab").unwrap());
        got.extend(resumed.finish().unwrap());
        got.sort_by_key(|m| (m.start(), m.end(), m.pattern().as_u32()));
        assert_eq!(got, want);
    }

    #[test]
    fn tail_stays_bounded() {
        let matcher = ChunkedOverlappingMatcher::new(["ab", "bab"]).unwrap();
        let mut cur = matcher.start();
        for _ in 0..10 {
            cur.feed(b"bbbbbbbbbbbbbbbb").unwrap();
            let cp = cur.checkpoint();
            assert!(
                cp.tail_len() <= 4,
                "tail never exceeds 2 * (max_len - 1)"
            );
        }
    }

    #[test]
    fn cursors_are_isolated_per_instance() {
        let matcher = ChunkedOverlappingMatcher::new(["ab"]).unwrap();
        let mut a = matcher.start();
        let b = matcher.start();
        a.feed(b"xxab").unwrap();
        assert_eq!(a.offset(), 4);
        assert_eq!(b.offset(), 0);
        let hits_b = collect(&matcher, &[b"ab"]);
        assert_eq!(hits_b.len(), 1);
        assert_eq!((hits_b[0].start(), hits_b[0].end()), (0, 2));
    }

    #[test]
    fn feed_after_finish_is_rejected() {
        let matcher = ChunkedOverlappingMatcher::new(["ab"]).unwrap();
        let mut cur = matcher.start();
        cur.feed(b"ab").unwrap();
        cur.finish().unwrap();
        assert_eq!(
            cur.feed(b"ab").unwrap_err(),
            ChunkedError::AlreadyFinished
        );
        assert_eq!(
            cur.finish().unwrap_err(),
            ChunkedError::AlreadyFinished
        );
    }
}
