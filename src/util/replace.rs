/*!
Provides a helper for replacing raw byte matches in a haystack.

The routine in this module is the shared implementation behind
[`AhoCorasick::try_replace_matches_bytes`][1]. It operates on an explicit
sequence of [`Match`] results (rather than running a search itself), so the
replacement fragments and the haystack may both contain arbitrary bytes,
including invalid UTF-8. Bytes not covered by any match are copied to the
output unchanged.

Since callers provide the matches themselves, the sequence is validated:
overlapping matches are explicitly rejected with a [`ReplaceError`][crate::ReplaceError] instead
of being resolved by arbitrarily choosing one of them.

[1]: crate::AhoCorasick::try_replace_matches_bytes
*/

use alloc::vec::Vec;

use crate::util::{
    error::ReplaceError,
    search::{Match, Span},
};

/// Replaces the given raw byte `matches` in `haystack` with the replacement
/// fragment corresponding to each match's pattern ID, and returns the
/// result.
///
/// Bytes in `haystack` that are not covered by any match are copied to the
/// output unchanged. Both `haystack` and the replacement fragments may
/// contain arbitrary bytes (including invalid UTF-8).
///
/// The matches must be non-overlapping and sorted by their start offset,
/// which is precisely what the search iterators on this crate (e.g.,
/// `AhoCorasick::find_iter`) yield. If two matches overlap, an error with
/// the [`ReplaceErrorKind::OverlappingMatches`][crate::ReplaceErrorKind]
/// kind is returned. An error
/// is also returned when a match has a pattern ID with no corresponding
/// replacement in `replace_with`, or when a match's span falls outside of
/// `haystack`.
pub(crate) fn replace_bytes<B: AsRef<[u8]>>(
    haystack: &[u8],
    matches: &[Match],
    replace_with: &[B],
) -> Result<Vec<u8>, ReplaceError> {
    // Validate everything before writing anything, so that an error never
    // leaves the caller with a partially replaced haystack.
    let mut prev: Option<Span> = None;
    for m in matches {
        let span = m.span();
        if span.end > haystack.len() {
            return Err(ReplaceError::span_out_of_bounds(
                span,
                haystack.len(),
            ));
        }
        if let Some(prev) = prev {
            if span.start < prev.end {
                return Err(ReplaceError::overlapping_matches(prev, span));
            }
        }
        if m.pattern().as_usize() >= replace_with.len() {
            return Err(ReplaceError::unknown_pattern_id(m.pattern()));
        }
        prev = Some(span);
    }
    let mut dst = Vec::with_capacity(haystack.len());
    let mut last = 0;
    for m in matches {
        let span = m.span();
        dst.extend_from_slice(&haystack[last..span.start]);
        dst.extend_from_slice(replace_with[m.pattern().as_usize()].as_ref());
        last = span.end;
    }
    dst.extend_from_slice(&haystack[last..]);
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::ReplaceErrorKind;

    #[test]
    fn unmatched_bytes_are_preserved() {
        let haystack = b"\xffa\x80b\xfe";
        let matches = &[Match::must(0, 1..2), Match::must(0, 3..4)];
        let got = replace_bytes(haystack, matches, &[b"XY".as_ref()]).unwrap();
        assert_eq!(b"\xffXY\x80XY\xfe".to_vec(), got);
    }

    #[test]
    fn non_utf8_replacements() {
        let haystack = b"ab";
        let matches = &[Match::must(0, 0..1), Match::must(1, 1..2)];
        let got = replace_bytes(
            haystack,
            matches,
            &[b"\xff".as_ref(), b"\xfe\x80".as_ref()],
        )
        .unwrap();
        assert_eq!(vec![0xFF, 0xFE, 0x80], got);
    }

    #[test]
    fn adjacent_matches_are_ok() {
        let haystack = b"aabb";
        let matches = &[Match::must(0, 0..2), Match::must(1, 2..4)];
        let got =
            replace_bytes(haystack, matches, &[b"x".as_ref(), b"y".as_ref()])
                .unwrap();
        assert_eq!(b"xy".to_vec(), got);
    }

    #[test]
    fn overlapping_matches_are_rejected() {
        let haystack = b"abc";
        let matches = &[Match::must(0, 0..2), Match::must(1, 1..3)];
        let err = replace_bytes(haystack, matches, &[b"x", b"y"]).unwrap_err();
        assert_eq!(
            &ReplaceErrorKind::OverlappingMatches {
                first: Span::from(0..2),
                second: Span::from(1..3),
            },
            err.kind(),
        );
    }

    #[test]
    fn unknown_pattern_id_is_rejected() {
        let haystack = b"abc";
        let matches = &[Match::must(1, 0..1)];
        let err = replace_bytes(haystack, matches, &[b"x"]).unwrap_err();
        assert_eq!(
            &ReplaceErrorKind::UnknownPatternID {
                pattern: crate::PatternID::must(1),
            },
            err.kind(),
        );
    }

    #[test]
    fn out_of_bounds_span_is_rejected() {
        let haystack = b"abc";
        let matches = &[Match::must(0, 1..4)];
        let err = replace_bytes(haystack, matches, &[b"x"]).unwrap_err();
        assert_eq!(
            &ReplaceErrorKind::SpanOutOfBounds {
                span: Span::from(1..4),
                haystack_len: 3,
            },
            err.kind(),
        );
    }
}
