/*!
Traceable normalized matching.

This module provides a layer on top of [`AhoCorasick`] that performs
*normalized* multi-pattern search while preserving provenance: every match
that is reported carries the byte range of the **original** (unnormalized)
haystack, even though the search itself runs over the normalized form of
both the patterns and the haystack.

# How it works

1. A per-codepoint normalization function is injected via [`Normalizer`].
2. At build time, every pattern is normalized (the function is applied
   exactly once per codepoint) and the normalized patterns are compiled
   into an [`AhoCorasick`] automaton.
3. At search time, the haystack is normalized codepoint-by-codepoint
   (again, the function is applied exactly once per codepoint) while
   recording, for every original codepoint, the byte span it occupies in
   the normalized buffer and the byte span it occupies in the original
   haystack.
4. The automaton runs over the normalized buffer. A candidate match is
   only reported when it **fully covers original codepoint boundaries**:
   its start must coincide with the start of some original codepoint's
   expansion and its end must coincide with the end of some (possibly
   different) original codepoint's expansion. A match that would cover
   only part of an expansion (for example, one `s` of the `ss` expansion
   of `ß`) is discarded, so a half-expansion is never misreported as a
   hit. Non-overlapping matches are then resolved with leftmost-longest
   semantics over the surviving candidates, so a discarded
   partial-expansion candidate never shadows a valid longer match.
5. Reported matches use original haystack byte offsets. The original
   haystack is only ever read, never modified.

# Empty expansions are rejected

The one and only policy for a codepoint that normalizes to the empty
string is **rejection**: [`NormalizeErrorKind::EmptyExpansion`] is
returned. An empty expansion has no bytes in the normalized buffer, so
there is no well-defined original byte range to attribute a match
boundary to. Rather than silently guessing, this layer refuses to build
a matcher (when a pattern contains such a codepoint) or to search (when
the haystack contains such a codepoint).

# Matcher cache

Matchers built from a *known* normalization configuration
([`Normalizer::sharp_s`], [`Normalizer::lowercase`]) may share a global
matcher cache (available with the `std` feature). Matchers built from an
*unknown* configuration ([`Normalizer::custom`]) **never** share the
cache: they are always compiled fresh and are never inserted into the
shared cache, so a custom normalization function can never observe or
pollute another configuration's automaton.

# Three-codepoint example

The haystack `aßb` has exactly three codepoints: `a`, `ß` and `b`. With
`ß` normalized to `ss`, the normalized haystack is `assb`. The pattern
`ss` matches and is reported over the original byte range of `ß` (`1..3`,
since `ß` is two bytes in UTF-8). The pattern `s` does **not** match,
because it would only cover half of the `ss` expansion of `ß`. The
adjacent codepoints keep their original offsets: `a` matches at `0..1`
and `b` matches at `3..4`.

```
use aho_corasick::normalized::{NormalizedAhoCorasick, Normalizer};

let ac = NormalizedAhoCorasick::builder()
    .normalizer(Normalizer::sharp_s())
    .build(["ss", "s", "a", "b"])
    .unwrap();
let got: Vec<_> = ac.find_iter("aßb").unwrap().collect();
// Only "ss" (over the original ß), "a" and "b" match. "s" does not.
assert_eq!(
    got.iter().map(|m| (m.pattern().as_usize(), m.start(), m.end())).collect::<Vec<_>>(),
    vec![(2, 0, 1), (0, 1, 3), (3, 3, 4)],
);
```

# Running the tests for this module

```text
cargo test --features std normalized
```
*/

use alloc::{string::String, sync::Arc, vec, vec::Vec};

use crate::{
    ahocorasick::{AhoCorasick, AhoCorasickBuilder},
    util::{
        error::BuildError,
        primitives::PatternID,
        search::{Match, Span},
    },
};

/// A per-codepoint normalization function.
///
/// A normalizer maps a single `char` to zero or more `char`s by pushing
/// the expansion onto a caller-provided buffer. It is applied **exactly
/// once per codepoint**, both when patterns are compiled and when a
/// haystack is searched.
///
/// A normalizer is either a *known* configuration
/// ([`Normalizer::sharp_s`], [`Normalizer::lowercase`]) or an *unknown*
/// (custom) configuration ([`Normalizer::custom`]). Only known
/// configurations are eligible for the shared matcher cache; unknown
/// configurations always compile their own automaton.
#[derive(Clone)]
pub struct Normalizer {
    kind: NormalizerKind,
}

#[derive(Clone)]
enum NormalizerKind {
    /// Maps `ß` (and `ẞ`) to `ss`. Everything else is left as is.
    SharpS,
    /// Unicode simple lowercase via `char::to_lowercase`.
    Lowercase,
    /// A user-provided function. Never shares the matcher cache.
    Custom(Arc<dyn Fn(char, &mut String) + Send + Sync + 'static>),
}

impl Normalizer {
    /// Returns a normalizer that maps `ß` and `ẞ` to `ss` and leaves
    /// every other codepoint unchanged.
    ///
    /// This is a known configuration and may share the matcher cache.
    pub fn sharp_s() -> Normalizer {
        Normalizer { kind: NormalizerKind::SharpS }
    }

    /// Returns a normalizer that maps every codepoint to its Unicode
    /// lowercase form via `char::to_lowercase`.
    ///
    /// This is a known configuration and may share the matcher cache.
    pub fn lowercase() -> Normalizer {
        Normalizer { kind: NormalizerKind::Lowercase }
    }

    /// Returns a normalizer from a user-provided per-codepoint function.
    ///
    /// This is an *unknown* configuration: matchers built with it never
    /// share the matcher cache and are always compiled fresh.
    pub fn custom<F>(f: F) -> Normalizer
    where
        F: Fn(char, &mut String) + Send + Sync + 'static,
    {
        Normalizer { kind: NormalizerKind::Custom(Arc::new(f)) }
    }

    /// Applies the normalization function to one codepoint, appending
    /// the expansion to `buf`.
    fn normalize_char(&self, ch: char, buf: &mut String) {
        match self.kind {
            NormalizerKind::SharpS => match ch {
                'ß' | 'ẞ' => buf.push_str("ss"),
                _ => buf.push(ch),
            },
            NormalizerKind::Lowercase => buf.extend(ch.to_lowercase()),
            NormalizerKind::Custom(ref f) => f(ch, buf),
        }
    }

    /// Returns a stable cache key for known configurations, or `None`
    /// for unknown (custom) configurations, which must never share the
    /// cache.
    fn cache_key(&self) -> Option<&'static str> {
        match self.kind {
            NormalizerKind::SharpS => {
                Some("aho-corasick::normalized::sharp-s-v1")
            }
            NormalizerKind::Lowercase => {
                Some("aho-corasick::normalized::lowercase-v1")
            }
            NormalizerKind::Custom(_) => None,
        }
    }

    /// Normalizes `s` codepoint-by-codepoint, applying the function
    /// once per codepoint. Returns an error if any codepoint expands to
    /// the empty string (the explicit rejection policy).
    fn normalize_str(&self, s: &str) -> Result<String, NormalizeError> {
        let mut out = String::with_capacity(s.len());
        for ch in s.chars() {
            let before = out.len();
            self.normalize_char(ch, &mut out);
            if out.len() == before {
                return Err(NormalizeError {
                    kind: NormalizeErrorKind::EmptyExpansion { ch },
                });
            }
        }
        Ok(out)
    }
}

impl core::fmt::Debug for Normalizer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self.kind {
            NormalizerKind::SharpS => "SharpS",
            NormalizerKind::Lowercase => "Lowercase",
            NormalizerKind::Custom(_) => "Custom",
        };
        f.debug_tuple("Normalizer").field(&name).finish()
    }
}

/// An error that occurred while building or searching with a
/// [`NormalizedAhoCorasick`].
#[derive(Debug)]
pub struct NormalizeError {
    kind: NormalizeErrorKind,
}

impl NormalizeError {
    /// Returns the kind of this error.
    pub fn kind(&self) -> &NormalizeErrorKind {
        &self.kind
    }
}

impl core::fmt::Display for NormalizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind {
            NormalizeErrorKind::EmptyExpansion { ch } => write!(
                f,
                "codepoint {:?} normalizes to the empty string, \
                 which is rejected by policy",
                ch,
            ),
            NormalizeErrorKind::EmptyPattern => {
                write!(f, "pattern normalizes to the empty string")
            }
            NormalizeErrorKind::InvalidUtf8Pattern => {
                write!(f, "pattern is not valid UTF-8")
            }
            NormalizeErrorKind::Build(ref err) => err.fmt(f),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NormalizeError {}

/// The kind of a [`NormalizeError`].
#[derive(Debug)]
#[non_exhaustive]
pub enum NormalizeErrorKind {
    /// A codepoint normalized to the empty string. Empty expansions are
    /// rejected by explicit policy, since they have no bytes in the
    /// normalized buffer and therefore no well-defined original byte
    /// range for match provenance.
    EmptyExpansion {
        /// The offending codepoint.
        ch: char,
    },
    /// A pattern normalized to the empty string.
    EmptyPattern,
    /// A pattern is not valid UTF-8. Per-codepoint normalization
    /// requires patterns to be valid UTF-8.
    InvalidUtf8Pattern,
    /// Compiling the underlying automaton failed.
    Build(BuildError),
}

/// A provenance-preserving normalized multi-pattern matcher.
///
/// Matches reported by this type always carry byte offsets into the
/// original (unnormalized) haystack. See the [module
/// documentation](self) for details.
#[derive(Clone, Debug)]
pub struct NormalizedAhoCorasick {
    ac: AhoCorasick,
    normalizer: Normalizer,
}

impl NormalizedAhoCorasick {
    /// Returns a builder for a provenance-preserving normalized matcher.
    pub fn builder() -> NormalizedAhoCorasickBuilder {
        NormalizedAhoCorasickBuilder::new()
    }

    /// Builds a matcher from the given normalizer and patterns.
    ///
    /// This is a convenience wrapper around
    /// [`NormalizedAhoCorasickBuilder`].
    pub fn new<I, P>(
        normalizer: Normalizer,
        patterns: I,
    ) -> Result<NormalizedAhoCorasick, NormalizeError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        NormalizedAhoCorasickBuilder::new()
            .normalizer(normalizer)
            .build(patterns)
    }

    /// Returns the normalizer used by this matcher.
    pub fn normalizer(&self) -> &Normalizer {
        &self.normalizer
    }

    /// Returns true if the haystack contains at least one match whose
    /// normalized form fully covers original codepoint boundaries.
    ///
    /// # Errors
    ///
    /// Returns [`NormalizeErrorKind::EmptyExpansion`] if any codepoint
    /// in the haystack normalizes to the empty string.
    pub fn is_match(&self, haystack: &str) -> Result<bool, NormalizeError> {
        Ok(self.find(haystack)?.is_some())
    }

    /// Returns the first match, if any, with offsets into the original
    /// haystack.
    ///
    /// # Errors
    ///
    /// Returns [`NormalizeErrorKind::EmptyExpansion`] if any codepoint
    /// in the haystack normalizes to the empty string.
    pub fn find(
        &self,
        haystack: &str,
    ) -> Result<Option<Match>, NormalizeError> {
        Ok(self.find_iter(haystack)?.next())
    }

    /// Returns an iterator over all non-overlapping matches, each
    /// carrying byte offsets into the original (unnormalized) haystack.
    ///
    /// Only matches that fully cover original codepoint boundaries are
    /// reported; a candidate that would split a multi-codepoint
    /// expansion (such as one `s` of the `ss` expansion of `ß`) is
    /// discarded. Non-overlapping matches are resolved with
    /// leftmost-longest semantics over the boundary-aligned candidates
    /// (ties are broken by pattern order), so a discarded
    /// partial-expansion candidate never shadows a valid longer match.
    ///
    /// # Errors
    ///
    /// Returns [`NormalizeErrorKind::EmptyExpansion`] if any codepoint
    /// in the haystack normalizes to the empty string.
    pub fn find_iter(
        &self,
        haystack: &str,
    ) -> Result<NormalizedFindIter, NormalizeError> {
        let normalized =
            NormalizedHaystack::build(&self.normalizer, haystack)?;
        // Collect every boundary-aligned candidate. Overlapping search
        // is used so that an unaligned candidate (which is discarded)
        // can never shadow a longer aligned candidate.
        let mut candidates: Vec<(usize, usize, usize, usize, PatternID)> =
            vec![];
        for mat in self.ac.find_overlapping_iter(&normalized.buf) {
            if let Some((orig_start, orig_end)) =
                normalized.original_span(mat.start(), mat.end())
            {
                candidates.push((
                    mat.start(),
                    mat.end(),
                    orig_start,
                    orig_end,
                    mat.pattern(),
                ));
            }
        }
        // Leftmost-longest over aligned candidates; ties broken by
        // pattern order.
        candidates
            .sort_by_key(|c| (c.0, core::cmp::Reverse(c.1), c.4.as_usize()));
        let mut matches = Vec::new();
        let mut pos = 0;
        for c in candidates {
            if c.0 < pos {
                continue;
            }
            matches.push(Match::new(c.4, Span { start: c.2, end: c.3 }));
            pos = c.1;
        }
        Ok(NormalizedFindIter { matches, pos: 0 })
    }
}

/// A builder for a [`NormalizedAhoCorasick`].
#[derive(Clone, Debug, Default)]
pub struct NormalizedAhoCorasickBuilder {
    normalizer: Option<Normalizer>,
    ac: AhoCorasickBuilder,
}

impl NormalizedAhoCorasickBuilder {
    /// Creates a new builder with the default configuration.
    pub fn new() -> NormalizedAhoCorasickBuilder {
        NormalizedAhoCorasickBuilder::default()
    }

    /// Sets the per-codepoint normalization function.
    pub fn normalizer(&mut self, normalizer: Normalizer) -> &mut Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Builds the matcher by normalizing every pattern (applying the
    /// normalization function exactly once per codepoint) and compiling
    /// the normalized patterns into an [`AhoCorasick`] automaton.
    ///
    /// Matchers built from a known configuration may share a global
    /// matcher cache (with the `std` feature). Matchers built from an
    /// unknown (custom) configuration never share the cache.
    ///
    /// # Errors
    ///
    /// * [`NormalizeErrorKind::EmptyExpansion`] if any codepoint in any
    ///   pattern normalizes to the empty string (rejected by policy).
    /// * [`NormalizeErrorKind::EmptyPattern`] if a pattern normalizes
    ///   to the empty string.
    /// * [`NormalizeErrorKind::Build`] if compiling the automaton
    ///   fails.
    pub fn build<I, P>(
        &self,
        patterns: I,
    ) -> Result<NormalizedAhoCorasick, NormalizeError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let normalizer =
            self.normalizer.clone().unwrap_or_else(Normalizer::sharp_s);
        let mut normalized_patterns: Vec<Vec<u8>> = vec![];
        for pat in patterns {
            let pat = core::str::from_utf8(pat.as_ref()).map_err(|_| {
                NormalizeError { kind: NormalizeErrorKind::InvalidUtf8Pattern }
            })?;
            let normalized = normalizer.normalize_str(pat)?;
            if normalized.is_empty() {
                return Err(NormalizeError {
                    kind: NormalizeErrorKind::EmptyPattern,
                });
            }
            normalized_patterns.push(normalized.into_bytes());
        }
        let ac = match normalizer.cache_key() {
            Some(key) => {
                cached_build(key, &normalized_patterns).map_err(|err| {
                    NormalizeError { kind: NormalizeErrorKind::Build(err) }
                })?
            }
            // Unknown configuration: never share the matcher cache.
            None => self.ac.build(&normalized_patterns).map_err(|err| {
                NormalizeError { kind: NormalizeErrorKind::Build(err) }
            })?,
        };
        Ok(NormalizedAhoCorasick { ac, normalizer })
    }
}

/// An iterator over non-overlapping normalized matches.
///
/// Each yielded [`Match`] carries byte offsets into the original
/// (unnormalized) haystack.
#[derive(Debug)]
pub struct NormalizedFindIter {
    matches: Vec<Match>,
    pos: usize,
}

impl Iterator for NormalizedFindIter {
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        let mat = self.matches.get(self.pos)?;
        self.pos += 1;
        Some(*mat)
    }
}

/// A normalized haystack plus the provenance map back to original byte
/// offsets.
#[derive(Debug)]
struct NormalizedHaystack {
    /// The normalized haystack.
    buf: String,
    /// One entry per original codepoint, in order, recording the byte
    /// span of its expansion in `buf` and its byte span in the original
    /// haystack.
    segments: Vec<Segment>,
}

#[derive(Debug)]
struct Segment {
    /// Byte start of this codepoint's expansion in the normalized
    /// buffer.
    norm_start: usize,
    /// Byte end of this codepoint's expansion in the normalized buffer.
    norm_end: usize,
    /// Byte start of this codepoint in the original haystack.
    orig_start: usize,
    /// Byte end of this codepoint in the original haystack.
    orig_end: usize,
}

impl NormalizedHaystack {
    /// Normalizes `haystack`, applying the normalization function
    /// exactly once per codepoint and recording provenance segments.
    fn build(
        normalizer: &Normalizer,
        haystack: &str,
    ) -> Result<NormalizedHaystack, NormalizeError> {
        let mut buf = String::with_capacity(haystack.len());
        let mut segments = Vec::new();
        for (orig_start, ch) in haystack.char_indices() {
            let norm_start = buf.len();
            normalizer.normalize_char(ch, &mut buf);
            let norm_end = buf.len();
            if norm_end == norm_start {
                // Explicit policy: empty expansions are rejected.
                return Err(NormalizeError {
                    kind: NormalizeErrorKind::EmptyExpansion { ch },
                });
            }
            segments.push(Segment {
                norm_start,
                norm_end,
                orig_start,
                orig_end: orig_start + ch.len_utf8(),
            });
        }
        Ok(NormalizedHaystack { buf, segments })
    }

    /// Maps a match span in the normalized buffer back to a byte span
    /// in the original haystack.
    ///
    /// Returns `None` when the span does not fully cover original
    /// codepoint boundaries, i.e. when `norm_start` or `norm_end`
    /// falls strictly inside a codepoint's expansion. Such a candidate
    /// is a partial-expansion match and must not be reported.
    fn original_span(
        &self,
        norm_start: usize,
        norm_end: usize,
    ) -> Option<(usize, usize)> {
        if norm_start >= norm_end {
            return None;
        }
        let first = self.segment_at(norm_start)?;
        if first.norm_start != norm_start {
            // Starts in the middle of an expansion.
            return None;
        }
        let last = self.segment_at(norm_end - 1)?;
        if last.norm_end != norm_end {
            // Ends in the middle of an expansion.
            return None;
        }
        Some((first.orig_start, last.orig_end))
    }

    /// Finds the segment containing normalized byte offset `at`.
    fn segment_at(&self, at: usize) -> Option<&Segment> {
        let i = self.segments.partition_point(|seg| seg.norm_end <= at);
        self.segments.get(i)
    }
}

/// Builds (or reuses) an automaton for a known configuration.
///
/// With `std`, known configurations share a global cache keyed by the
/// configuration identity and the normalized patterns. Without `std`,
/// a fresh automaton is always compiled.
#[cfg(feature = "std")]
fn cached_build(
    key: &'static str,
    patterns: &[Vec<u8>],
) -> Result<AhoCorasick, BuildError> {
    use std::{collections::HashMap, sync::OnceLock};

    type Key = (&'static str, Vec<Vec<u8>>);
    static CACHE: OnceLock<std::sync::Mutex<HashMap<Key, AhoCorasick>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let key = (key, patterns.to_vec());
    if let Some(ac) = cache.lock().unwrap().get(&key) {
        return Ok(ac.clone());
    }
    let ac = AhoCorasick::new(patterns)?;
    cache.lock().unwrap().insert(key, ac.clone());
    Ok(ac)
}

/// Without `std`, there is no shared cache; compile fresh every time.
#[cfg(not(feature = "std"))]
fn cached_build(
    _key: &'static str,
    patterns: &[Vec<u8>],
) -> Result<AhoCorasick, BuildError> {
    AhoCorasick::new(patterns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn matches(
        ac: &NormalizedAhoCorasick,
        haystack: &str,
    ) -> Vec<(usize, usize, usize)> {
        ac.find_iter(haystack)
            .unwrap()
            .map(|m| (m.pattern().as_usize(), m.start(), m.end()))
            .collect()
    }

    #[test]
    fn sharp_s_full_expansion_matches_original_range() {
        // Three-codepoint example: "aßb" normalizes to "assb".
        let ac =
            NormalizedAhoCorasick::new(Normalizer::sharp_s(), ["ss"]).unwrap();
        // "ss" hits the expansion of ß and is reported over the
        // original byte range of ß (ß is 2 bytes in UTF-8, at 1..3).
        assert_eq!(matches(&ac, "aßb"), vec![(0, 1, 3)]);
        assert_eq!(matches(&ac, "ß"), vec![(0, 0, 2)]);
        // A literal "ss" in the haystack still matches with its own
        // original offsets.
        assert_eq!(matches(&ac, "assb"), vec![(0, 1, 3)]);
    }

    #[test]
    fn half_expansion_is_not_a_match() {
        // Pattern "s" must not match half of the ß -> ss expansion.
        let ac =
            NormalizedAhoCorasick::new(Normalizer::sharp_s(), ["s"]).unwrap();
        assert_eq!(matches(&ac, "ß"), vec![]);
        assert_eq!(matches(&ac, "aßb"), vec![]);
        // But a real "s" still matches.
        assert_eq!(matches(&ac, "s"), vec![(0, 0, 1)]);
        assert_eq!(matches(&ac, "ss"), vec![(0, 0, 1), (0, 1, 2)]);
    }

    #[test]
    fn adjacent_text_offsets_are_correct() {
        let ac = NormalizedAhoCorasick::new(
            Normalizer::sharp_s(),
            ["a", "b", "ss"],
        )
        .unwrap();
        assert_eq!(matches(&ac, "aßb"), vec![(0, 0, 1), (2, 1, 3), (1, 3, 4)]);
    }

    #[test]
    fn empty_expansion_is_rejected_at_build_time() {
        // Explicit policy: a codepoint that expands to nothing is an
        // error.
        let ac = NormalizedAhoCorasick::new(
            Normalizer::custom(|ch, buf| {
                if ch != 'x' {
                    buf.push(ch);
                }
            }),
            ["axb"],
        );
        match ac {
            Err(ref err) => {
                assert!(matches!(
                    err.kind(),
                    NormalizeErrorKind::EmptyExpansion { ch: 'x' }
                ));
            }
            Ok(_) => panic!("expected empty expansion error"),
        }
    }

    #[test]
    fn empty_expansion_is_rejected_at_search_time() {
        let ac = NormalizedAhoCorasick::new(
            Normalizer::custom(|ch, buf| {
                if ch != 'x' {
                    buf.push(ch);
                }
            }),
            ["ab"],
        )
        .unwrap();
        match ac.find_iter("axb") {
            Err(ref err) => {
                assert!(matches!(
                    err.kind(),
                    NormalizeErrorKind::EmptyExpansion { ch: 'x' }
                ));
            }
            Ok(_) => panic!("expected empty expansion error"),
        }
    }

    #[test]
    fn empty_normalized_pattern_is_rejected() {
        let ac = NormalizedAhoCorasick::new(
            Normalizer::custom(|_ch, _buf| {}),
            [""],
        );
        // The empty pattern contains no codepoints, so it normalizes
        // to the empty string and is rejected.
        match ac {
            Err(ref err) => {
                assert!(matches!(
                    err.kind(),
                    NormalizeErrorKind::EmptyPattern
                ));
            }
            Ok(_) => panic!("expected empty pattern error"),
        }
    }

    #[test]
    fn unknown_configs_do_not_share_matcher_cache() {
        // Two custom normalizers with opposite behavior. If they
        // shared a cache keyed only by patterns, the second build
        // would wrongly reuse the first one's automaton.
        let expand = NormalizedAhoCorasick::new(
            Normalizer::custom(|ch, buf| {
                if ch == 'ß' {
                    buf.push_str("ss");
                } else {
                    buf.push(ch);
                }
            }),
            ["ss"],
        )
        .unwrap();
        let identity = NormalizedAhoCorasick::new(
            Normalizer::custom(|ch, buf| buf.push(ch)),
            ["ss"],
        )
        .unwrap();
        assert_eq!(matches(&expand, "ß"), vec![(0, 0, 2)]);
        assert_eq!(matches(&identity, "ß"), vec![]);
        assert_eq!(matches(&identity, "ss"), vec![(0, 0, 2)]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn known_configs_share_matcher_cache() {
        let pats = vec![b"ss".to_vec()];
        let key = Normalizer::sharp_s().cache_key().unwrap();
        let a = cached_build(key, &pats).unwrap();
        let b = cached_build(key, &pats).unwrap();
        // Same cache entry: both automatons find the same match.
        assert_eq!(
            a.find("assb").map(|m| (m.start(), m.end())),
            b.find("assb").map(|m| (m.start(), m.end())),
        );
        assert_eq!(a.find("assb").map(|m| (m.start(), m.end())), Some((1, 3)));
    }

    #[test]
    fn lowercase_normalization() {
        let ac = NormalizedAhoCorasick::new(Normalizer::lowercase(), ["abc"])
            .unwrap();
        assert_eq!(matches(&ac, "xxABCxx"), vec![(0, 2, 5)]);
    }

    #[test]
    fn unaligned_candidate_does_not_shadow_aligned_one() {
        // "sb" starts inside the ß -> ss expansion and is discarded;
        // the equally long but aligned "ss" must still be reported.
        let ac =
            NormalizedAhoCorasick::new(Normalizer::sharp_s(), ["sb", "ss"])
                .unwrap();
        assert_eq!(matches(&ac, "aßb"), vec![(1, 1, 3)]);
        // A shorter unaligned candidate must not shadow a longer
        // aligned one either.
        let ac =
            NormalizedAhoCorasick::new(Normalizer::sharp_s(), ["s", "ss"])
                .unwrap();
        assert_eq!(matches(&ac, "aßb"), vec![(1, 1, 3)]);
    }

    #[test]
    fn multi_codepoint_expansion_boundaries() {
        // 'ß' -> "ss": patterns spanning the expansion plus neighbors
        // must cover whole codepoints. "aßb" normalizes to "assb".
        let ac = NormalizedAhoCorasick::new(Normalizer::sharp_s(), ["ass"])
            .unwrap();
        // "ass" covers "a" and all of ß: original range 0..3.
        assert_eq!(matches(&ac, "aßb"), vec![(0, 0, 3)]);
        let ac = NormalizedAhoCorasick::new(Normalizer::sharp_s(), ["ssb"])
            .unwrap();
        // "ssb" covers all of ß and "b": original range 1..4.
        assert_eq!(matches(&ac, "aßb"), vec![(0, 1, 4)]);
        let ac = NormalizedAhoCorasick::new(Normalizer::sharp_s(), ["assb"])
            .unwrap();
        // "assb" covers the whole haystack: original range 0..4.
        assert_eq!(matches(&ac, "aßb"), vec![(0, 0, 4)]);
        // "as" and "sb" end/start in the middle of the ß expansion
        // and must not match.
        let ac =
            NormalizedAhoCorasick::new(Normalizer::sharp_s(), ["as", "sb"])
                .unwrap();
        assert_eq!(matches(&ac, "aßb"), vec![]);
    }
}
