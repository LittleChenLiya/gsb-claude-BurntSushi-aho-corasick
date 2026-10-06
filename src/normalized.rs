/*!
Traceable normalized matching.

This module provides a small layer on top of [`AhoCorasick`] for searching
text through a per-codepoint normalization function while preserving the
byte offsets of the _original_ (unnormalized) text.

The workflow is:

1. Inject a normalization function via [`Normalizer`]. The function maps a
   single `char` to its replacement (`Some`) or to itself (`None`). A single
   character may expand to multiple characters (e.g. `ß` to `ss`).
2. Build a [`NormalizedMatcher`] from a set of patterns. Each pattern is
   normalized with the same function before being compiled into the
   automaton.
3. Normalize each haystack with [`NormalizedMatcher::normalize_haystack`].
   This produces a [`NormalizedHaystack`] that remembers, for every fragment
   of normalized text, the byte range of the original character it came
   from. The original text is only ever read, never modified.
4. Search with [`NormalizedMatcher::find_iter`]. Every reported
   [`NormalizedMatch`] carries the byte range in the _original_ text.

# Boundary guarantee

Because one original character may expand to several normalized characters,
a match in normalized space is only reported when it covers a whole number
of original characters. A match that starts or ends in the middle of an
expansion is silently dropped. For example, with `ß` normalized to `ss`,
the pattern `ss` matches the original `ß` (its full byte range), but the
pattern `s` never matches "half" of a `ß`.

# Normalization is applied once per character

The normalization function is memoized: each distinct `char` is passed
through the function at most once per [`Normalizer`], no matter how many
times it occurs in the patterns or in the haystacks.

# Empty expansions are rejected

A normalization function that maps a character to the empty string would
make the original character vanish, which cannot be represented faithfully
in the offset mapping. This is rejected explicitly: building a matcher or
normalizing a haystack fails with [`NormalizeErrorKind::EmptyExpansion`].

# Matcher caching

Callers that build many matchers may route construction through a
[`MatcherCache`] (requires the `std` feature). A builder is only allowed to
share its matcher through the cache when it has been given an explicit
[`NormalizedMatcherBuilder::cache_key`]. A builder with an unknown
configuration (no cache key) never shares and never populates the cache.

# Example

This example uses the three character string `aßb`. The `ß` (two bytes in
UTF-8) expands to `ss`, so the pattern `ss` matches the byte range `1..3`
of the original text, while the neighboring characters keep their original
offsets.

```
use aho_corasick::normalized::{NormalizedMatcherBuilder, Normalizer};

let normalizer = Normalizer::new(|ch| match ch {
    'ß' => Some("ss".to_string()),
    _ => None,
});
let matcher = NormalizedMatcherBuilder::new()
    .normalizer(normalizer)
    .build(["ss", "a", "b"])
    .unwrap();

// The original text is only read, never modified.
let hay = matcher.normalize_haystack("aßb").unwrap();
let mut got = vec![];
for m in matcher.find_iter(&hay) {
    got.push((m.pattern().as_usize(), m.span().start, m.span().end));
}
assert_eq!(got, vec![(1, 0, 1), (0, 1, 3), (2, 3, 4)]);

// But the pattern "s" must not match half of the expanded "ß".
let matcher = NormalizedMatcherBuilder::new()
    .normalizer(Normalizer::new(|ch| match ch {
        'ß' => Some("ss".to_string()),
        _ => None,
    }))
    .build(["s"])
    .unwrap();
let hay = matcher.normalize_haystack("aßb").unwrap();
assert_eq!(None, matcher.find(&hay));
```
*/

use alloc::{
    boxed::Box, collections::BTreeMap, string::String, sync::Arc, vec::Vec,
};
use core::cell::RefCell;

use crate::util::error::BuildError;
use crate::{AhoCorasick, AhoCorasickBuilder, FindIter, Match, PatternID, Span};

/// A per-codepoint normalization function with memoization.
///
/// The wrapped function maps a `char` to `Some(replacement)` to rewrite it
/// (possibly to multiple characters) or to `None` to keep it as-is.
/// Returning `Some("")` is an error; see
/// [`NormalizeErrorKind::EmptyExpansion`].
///
/// Each distinct character is passed to the wrapped function at most once;
/// results are memoized for the lifetime of the `Normalizer`.
///
/// Note that because of the memoization, a `Normalizer` (and thus a
/// [`NormalizedMatcher`]) is not `Sync`.
pub struct Normalizer {
    f: Box<dyn Fn(char) -> Option<String> + Send + Sync>,
    memo: RefCell<BTreeMap<char, Option<String>>>,
}

impl core::fmt::Debug for Normalizer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Normalizer").finish_non_exhaustive()
    }
}

impl Normalizer {
    /// Creates a new normalizer from the given per-codepoint function.
    pub fn new<F>(f: F) -> Normalizer
    where
        F: Fn(char) -> Option<String> + Send + Sync + 'static,
    {
        Normalizer { f: Box::new(f), memo: RefCell::new(BTreeMap::new()) }
    }

    /// Returns the expansion for `ch`, calling the underlying function at
    /// most once per distinct character.
    fn expand(&self, ch: char) -> Option<String> {
        if let Some(hit) = self.memo.borrow().get(&ch) {
            return hit.clone();
        }
        let out = (self.f)(ch);
        self.memo.borrow_mut().insert(ch, out.clone());
        out
    }

    /// Normalizes `text`, returning the normalized string plus one fragment
    /// record per original character. The original text is only read.
    fn normalize_text(
        &self,
        text: &str,
    ) -> Result<(String, Vec<Fragment>), NormalizeError> {
        let mut normalized = String::with_capacity(text.len());
        let mut frags = Vec::new();
        for (offset, ch) in text.char_indices() {
            match self.expand(ch) {
                None => normalized.push(ch),
                Some(replacement) => {
                    if replacement.is_empty() {
                        return Err(NormalizeError {
                            kind: NormalizeErrorKind::EmptyExpansion {
                                ch,
                                offset,
                            },
                        });
                    }
                    normalized.push_str(&replacement);
                }
            }
            frags.push(Fragment {
                norm_end: normalized.len(),
                orig_start: offset,
                orig_end: offset + ch.len_utf8(),
            });
        }
        Ok((normalized, frags))
    }
}

/// A builder for [`NormalizedMatcher`].
#[derive(Debug, Default)]
pub struct NormalizedMatcherBuilder {
    normalizer: Option<Arc<Normalizer>>,
    #[cfg(feature = "std")]
    cache_key: Option<String>,
    ac: AhoCorasickBuilder,
}

impl NormalizedMatcherBuilder {
    /// Creates a new builder with the default (identity) normalization and
    /// no cache key.
    pub fn new() -> NormalizedMatcherBuilder {
        NormalizedMatcherBuilder::default()
    }

    /// Sets the normalization function applied to both the patterns and the
    /// haystacks.
    pub fn normalizer(&mut self, normalizer: Normalizer) -> &mut Self {
        self.normalizer = Some(Arc::new(normalizer));
        self
    }

    /// Sets the cache key identifying this configuration.
    ///
    /// When a key is set, [`NormalizedMatcherBuilder::build_cached`] may
    /// share the resulting matcher through a [`MatcherCache`]. When no key
    /// is set (the configuration is "unknown"), the builder never shares
    /// its matcher and never populates the cache.
    #[cfg(feature = "std")]
    pub fn cache_key<S: Into<String>>(&mut self, key: S) -> &mut Self {
        self.cache_key = Some(key.into());
        self
    }

    /// Gives access to the underlying [`AhoCorasickBuilder`] for tuning the
    /// automaton itself.
    pub fn aho_corasick_builder(&mut self) -> &mut AhoCorasickBuilder {
        &mut self.ac
    }

    /// Normalizes the patterns and compiles them into a new matcher.
    ///
    /// Every pattern is normalized with the configured function before
    /// compilation, so patterns live in the same normalized space as the
    /// haystacks produced by the matcher.
    pub fn build<I, P>(
        &self,
        patterns: I,
    ) -> Result<NormalizedMatcher, NormalizeError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        let normalizer = match self.normalizer {
            Some(ref n) => Arc::clone(n),
            None => Arc::new(Normalizer::new(|_| None)),
        };
        let mut normalized_patterns = Vec::new();
        for pat in patterns {
            let (normalized, _) = normalizer.normalize_text(pat.as_ref())?;
            normalized_patterns.push(normalized);
        }
        let ac = self
            .ac
            .build(normalized_patterns)
            .map_err(NormalizeError::from_build)?;
        Ok(NormalizedMatcher { ac, normalizer })
    }

    /// Builds a matcher, sharing it through `cache` when this builder has a
    /// cache key.
    ///
    /// If [`NormalizedMatcherBuilder::cache_key`] was set and a matcher was
    /// already built for the same key, the cached matcher is returned and
    /// `patterns` is ignored. Otherwise a new matcher is built and, only
    /// when a key is set, inserted into the cache.
    ///
    /// A builder without a cache key (an unknown configuration) always
    /// builds a fresh matcher and never reads from or writes to the cache.
    #[cfg(feature = "std")]
    pub fn build_cached<I, P>(
        &self,
        patterns: I,
        cache: &MatcherCache,
    ) -> Result<Arc<NormalizedMatcher>, NormalizeError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        let key = match self.cache_key {
            Some(ref key) => key.clone(),
            // Unknown configuration: never share the matcher cache.
            None => return Ok(Arc::new(self.build(patterns)?)),
        };
        if let Some(hit) = cache.get(&key) {
            return Ok(hit);
        }
        let matcher = Arc::new(self.build(patterns)?);
        cache.insert(key, Arc::clone(&matcher));
        Ok(matcher)
    }
}

/// An Aho-Corasick matcher over normalized text that reports byte ranges in
/// the original text.
///
/// Created by [`NormalizedMatcherBuilder::build`]. Note that a
/// `NormalizedMatcher` is not `Sync` because the normalizer memoizes.
#[derive(Debug)]
pub struct NormalizedMatcher {
    ac: AhoCorasick,
    normalizer: Arc<Normalizer>,
}

impl NormalizedMatcher {
    /// Normalizes `text` and records the mapping from every fragment of
    /// normalized text back to the byte range of the original character it
    /// came from.
    ///
    /// The original text is only read, never modified, and does not need to
    /// outlive the returned value.
    ///
    /// # Errors
    ///
    /// Returns [`NormalizeErrorKind::EmptyExpansion`] if the normalization
    /// function maps any character of `text` to the empty string.
    pub fn normalize_haystack(
        &self,
        text: &str,
    ) -> Result<NormalizedHaystack, NormalizeError> {
        let (normalized, frags) = self.normalizer.normalize_text(text)?;
        Ok(NormalizedHaystack { normalized, orig_len: text.len(), frags })
    }

    /// Returns an iterator over all matches in `hay`, in normalized space
    /// order, with each match carrying its byte range in the original text.
    ///
    /// Matches that would start or end in the middle of a multi-character
    /// expansion are not reported.
    pub fn find_iter<'m, 'h>(
        &'m self,
        hay: &'h NormalizedHaystack,
    ) -> NormalizedFindIter<'m, 'h> {
        NormalizedFindIter { it: self.ac.find_iter(&hay.normalized), hay }
    }

    /// Returns the first match in `hay`, if any, with its byte range in the
    /// original text.
    pub fn find(&self, hay: &NormalizedHaystack) -> Option<NormalizedMatch> {
        self.find_iter(hay).next()
    }
}

/// A normalized haystack plus the mapping back to original byte offsets.
///
/// Produced by [`NormalizedMatcher::normalize_haystack`].
#[derive(Debug)]
pub struct NormalizedHaystack {
    normalized: String,
    orig_len: usize,
    frags: Vec<Fragment>,
}

/// Maps one original character to its fragment of normalized text.
/// Fragments partition the normalized string in order, so only the end of
/// each fragment is stored.
#[derive(Debug)]
struct Fragment {
    /// End offset (exclusive) of this fragment in the normalized string.
    norm_end: usize,
    /// Start of the original character, in original byte offsets.
    orig_start: usize,
    /// End of the original character, in original byte offsets.
    orig_end: usize,
}

impl NormalizedHaystack {
    /// Returns the normalized form of the haystack.
    pub fn normalized(&self) -> &str {
        &self.normalized
    }

    /// Returns true when `offset` is a fragment boundary in normalized
    /// space, i.e. a position at which a match may start or end without
    /// splitting an expanded character.
    fn is_boundary(&self, offset: usize) -> bool {
        offset == 0
            || self
                .frags
                .binary_search_by_key(&offset, |f| f.norm_end)
                .is_ok()
    }

    /// Maps a match in normalized space to original byte offsets. Returns
    /// `None` when the match does not align with whole original characters.
    fn map_match(&self, mat: Match) -> Option<NormalizedMatch> {
        let (start, end) = (mat.start(), mat.end());
        if !self.is_boundary(start) || !self.is_boundary(end) {
            return None;
        }
        let orig_start = if start == self.normalized.len() {
            self.orig_len
        } else {
            let i = self.frags.partition_point(|f| f.norm_end <= start);
            self.frags[i].orig_start
        };
        let orig_end = if end == 0 {
            0
        } else {
            let i = self.frags.partition_point(|f| f.norm_end <= end - 1);
            self.frags[i].orig_end
        };
        Some(NormalizedMatch {
            mat,
            orig: Span { start: orig_start, end: orig_end },
        })
    }
}

/// A single match in a normalized haystack.
///
/// Carries both the span in normalized space and the corresponding span in
/// the original text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizedMatch {
    mat: Match,
    orig: Span,
}

impl NormalizedMatch {
    /// Returns the ID of the pattern that matched.
    pub fn pattern(&self) -> PatternID {
        self.mat.pattern()
    }

    /// Returns the byte range of this match in the original (unnormalized)
    /// text.
    pub fn span(&self) -> Span {
        self.orig
    }

    /// Returns the byte range of this match in the normalized text.
    pub fn normalized_span(&self) -> Span {
        Span { start: self.mat.start(), end: self.mat.end() }
    }
}

/// An iterator over normalized matches in a [`NormalizedHaystack`].
///
/// Created by [`NormalizedMatcher::find_iter`].
#[derive(Debug)]
pub struct NormalizedFindIter<'m, 'h> {
    it: FindIter<'m, 'h>,
    hay: &'h NormalizedHaystack,
}

impl<'m, 'h> Iterator for NormalizedFindIter<'m, 'h> {
    type Item = NormalizedMatch;

    fn next(&mut self) -> Option<NormalizedMatch> {
        loop {
            let mat = self.it.next()?;
            if let Some(m) = self.hay.map_match(mat) {
                return Some(m);
            }
        }
    }
}

/// An error that occurred while normalizing patterns or a haystack, or
/// while compiling the underlying automaton.
#[derive(Debug)]
pub struct NormalizeError {
    kind: NormalizeErrorKind,
}

/// The kind of a [`NormalizeError`].
#[derive(Debug)]
#[non_exhaustive]
pub enum NormalizeErrorKind {
    /// The normalization function mapped a character to the empty string.
    /// Empty expansions are rejected because a vanishing character cannot
    /// be represented faithfully in the offset mapping.
    EmptyExpansion {
        /// The character that expanded to the empty string.
        ch: char,
        /// The byte offset of the character in the text being normalized.
        offset: usize,
    },
    /// Compiling the normalized patterns into an automaton failed.
    Build(BuildError),
}

impl NormalizeError {
    fn from_build(err: BuildError) -> NormalizeError {
        NormalizeError { kind: NormalizeErrorKind::Build(err) }
    }

    /// Returns the kind of this error.
    pub fn kind(&self) -> &NormalizeErrorKind {
        &self.kind
    }
}

impl core::fmt::Display for NormalizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind {
            NormalizeErrorKind::EmptyExpansion { ch, offset } => write!(
                f,
                "normalization of {:?} at byte offset {} expands to the empty string, which is not supported",
                ch, offset,
            ),
            NormalizeErrorKind::Build(ref err) => {
                write!(f, "failed to build automaton: {}", err)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NormalizeError {}

/// A cache of matchers keyed by an explicit configuration key.
///
/// Only builders that carry a [`NormalizedMatcherBuilder::cache_key`]
/// participate in the cache. Builders with an unknown configuration (no
/// key) never share matchers through a cache.
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub struct MatcherCache {
    map: std::sync::Mutex<
        std::collections::HashMap<String, Arc<NormalizedMatcher>>,
    >,
}

#[cfg(feature = "std")]
impl MatcherCache {
    /// Creates a new, empty cache.
    pub fn new() -> MatcherCache {
        MatcherCache::default()
    }

    /// Returns the number of cached matchers.
    pub fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }

    /// Returns true when the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, key: &str) -> Option<Arc<NormalizedMatcher>> {
        self.map.lock().unwrap().get(key).cloned()
    }

    fn insert(&self, key: String, matcher: Arc<NormalizedMatcher>) {
        self.map.lock().unwrap().insert(key, matcher);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sharp_s(ch: char) -> Option<String> {
        match ch {
            'ß' => Some("ss".to_string()),
            _ => None,
        }
    }

    fn matcher(patterns: &[&str]) -> NormalizedMatcher {
        NormalizedMatcherBuilder::new()
            .normalizer(Normalizer::new(sharp_s))
            .build(patterns.iter().copied())
            .unwrap()
    }

    fn spans(
        m: &NormalizedMatcher,
        hay: &NormalizedHaystack,
    ) -> Vec<(usize, usize, usize)> {
        m.find_iter(hay)
            .map(|m| (m.pattern().as_usize(), m.span().start, m.span().end))
            .collect()
    }

    #[test]
    fn sharp_s_expands_and_pattern_ss_covers_whole_char() {
        // Three character example: "aßb". 'ß' is two bytes (0xC3 0x9F),
        // so it occupies the original byte range 1..3.
        let m = matcher(&["ss"]);
        let hay = m.normalize_haystack("aßb").unwrap();
        assert_eq!("assb", hay.normalized());
        assert_eq!(vec![(0, 1, 3)], spans(&m, &hay));
        // The normalized-space span is reported too.
        let mat = m.find(&hay).unwrap();
        assert_eq!(Span { start: 1, end: 3 }, mat.normalized_span());
        assert_eq!(Span { start: 1, end: 3 }, mat.span());
    }

    #[test]
    fn single_s_cannot_match_half_of_expansion() {
        let m = matcher(&["s"]);
        // "ß" normalizes to "ss", but neither half may match on its own.
        let hay = m.normalize_haystack("aßb").unwrap();
        assert_eq!(None, m.find(&hay));
        // A plain "s" in the original text still matches, with correct
        // offsets for the adjacent text.
        let hay = m.normalize_haystack("asb").unwrap();
        assert_eq!(vec![(0, 1, 2)], spans(&m, &hay));
    }

    #[test]
    fn adjacent_text_offsets_are_correct() {
        let m = matcher(&["a", "b", "ss", "x"]);
        let hay = m.normalize_haystack("aßb").unwrap();
        assert_eq!(vec![(0, 0, 1), (2, 1, 3), (1, 3, 4)], spans(&m, &hay));
        // And with the expansion in the middle of a longer haystack.
        let hay = m.normalize_haystack("zzßzz").unwrap();
        assert_eq!(vec![(2, 2, 4)], spans(&m, &hay));
    }

    #[test]
    fn patterns_are_normalized_too() {
        // A pattern containing 'ß' is normalized to "ss" and therefore
        // matches a literal "ss" in the haystack.
        let m = matcher(&["ß"]);
        let hay = m.normalize_haystack("assb").unwrap();
        assert_eq!(vec![(0, 1, 3)], spans(&m, &hay));
    }

    #[test]
    fn empty_expansion_is_rejected() {
        let empty_x = || {
            Normalizer::new(|ch| match ch {
                'x' => Some("".to_string()),
                _ => None,
            })
        };
        // Rejected while compiling patterns.
        let err = NormalizedMatcherBuilder::new()
            .normalizer(empty_x())
            .build(["axb"])
            .unwrap_err();
        match err.kind() {
            &NormalizeErrorKind::EmptyExpansion { ch, offset } => {
                assert_eq!('x', ch);
                assert_eq!(1, offset);
            }
            _ => panic!("expected EmptyExpansion, got {:?}", err),
        }

        // Rejected while normalizing a haystack.
        let m = NormalizedMatcherBuilder::new()
            .normalizer(empty_x())
            .build(["a"])
            .unwrap();
        let err = m.normalize_haystack("axb").unwrap_err();
        match err.kind() {
            &NormalizeErrorKind::EmptyExpansion { ch, offset } => {
                assert_eq!('x', ch);
                assert_eq!(1, offset);
            }
            _ => panic!("expected EmptyExpansion, got {:?}", err),
        }
    }

    #[test]
    fn normalizer_is_called_once_per_distinct_char() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let m = NormalizedMatcherBuilder::new()
            .normalizer(Normalizer::new(move |ch| {
                counter.fetch_add(1, Ordering::SeqCst);
                sharp_s(ch)
            }))
            .build(["ss"])
            .unwrap();
        // 's' seen once while compiling the pattern.
        assert_eq!(1, calls.load(Ordering::SeqCst));
        // The haystack has two distinct chars, one of which repeats.
        // Each 'ß' is two bytes; ' ' is one byte at offset 6.
        let hay = m.normalize_haystack("ßßß ß").unwrap();
        assert_eq!(3, calls.load(Ordering::SeqCst));
        assert_eq!(
            vec![(0, 0, 2), (0, 2, 4), (0, 4, 6), (0, 7, 9)],
            spans(&m, &hay),
        );
        // Re-normalizing another haystack hits the memo, not the function.
        let _hay2 = m.normalize_haystack("ß ß").unwrap();
        assert_eq!(3, calls.load(Ordering::SeqCst));
    }

    #[test]
    fn unknown_config_never_shares_matcher_cache() {
        let cache = MatcherCache::new();

        // Unknown configuration (no cache key): always fresh, never cached.
        let b1 = NormalizedMatcherBuilder::new()
            .normalizer(Normalizer::new(sharp_s))
            .build_cached(["ss"], &cache)
            .unwrap();
        let b2 = NormalizedMatcherBuilder::new()
            .normalizer(Normalizer::new(sharp_s))
            .build_cached(["ss"], &cache)
            .unwrap();
        assert!(!Arc::ptr_eq(&b1, &b2));
        assert!(cache.is_empty());

        // Known configuration: shared through the cache.
        let k1 = NormalizedMatcherBuilder::new()
            .normalizer(Normalizer::new(sharp_s))
            .cache_key("sharp-s-v1")
            .build_cached(["ss"], &cache)
            .unwrap();
        let k2 = NormalizedMatcherBuilder::new()
            .normalizer(Normalizer::new(sharp_s))
            .cache_key("sharp-s-v1")
            .build_cached(["ss"], &cache)
            .unwrap();
        assert!(Arc::ptr_eq(&k1, &k2));
        assert_eq!(1, cache.len());

        // A different key is a different entry.
        let k3 = NormalizedMatcherBuilder::new()
            .normalizer(Normalizer::new(sharp_s))
            .cache_key("sharp-s-v2")
            .build_cached(["ss"], &cache)
            .unwrap();
        assert!(!Arc::ptr_eq(&k1, &k3));
        assert_eq!(2, cache.len());
    }
}
