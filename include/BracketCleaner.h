#ifndef __PADDLEVINO_BRACKET_CLEANER_H__
#define __PADDLEVINO_BRACKET_CLEANER_H__

#include <string>
#include <vector>

// Removes OCR noise like "Name[" (a cell border misread as a bracket)
// without touching the image crop: in office text, ()[]{} should always come
// balanced, so an unmatched one on a line is treated as noise and dropped.
//
// Three safeguards keep that from eating legitimate text:
//
//  * an edge restriction -- the artifact is a cell border caught inside the
//    detected box, so it sits at one end of the run ("ID[", "]Address"); an
//    unmatched bracket in the middle is more likely real text whose partner
//    the OCR dropped, and is left alone unless BracketCleanupParams::anywhere;
//  * a per-kind page-level veto -- a pair that genuinely spans two lines
//    ("Preferred Contact Method (select one of the" / "following options)",
//    measured on a real form) looks orphan on both lines, so if a bracket
//    kind has an unmatched open AND an unmatched close on the page, nothing
//    of that kind is touched. The real artifact has no partner anywhere on
//    the page, which is exactly what survives the veto;
//  * a page-level frequency gate -- a source-code page breaks the "always
//    balanced" assumption on purpose ("{" alone at the end of a line, "}"
//    alone on the next), so the cleanup bails out entirely when unmatched
//    brackets are still too common after the veto; see
//    BracketCleanupParams::gateRatio.

enum class BracketCleanupResult {
    Off,             // --clean-orphan-brackets not passed
    Applied,         // gate passed, orphan brackets stripped (possibly none found)
    SkippedCodeGate, // fraction of lines with orphans exceeded gateRatio; page left untouched
};

struct BracketCleanupParams {
    bool enabled = false;
    // Fraction of lines still holding an unmatched bracket after the per-kind
    // veto, above which the page is assumed to be source code and the cleanup
    // is skipped entirely. Measured on a real form (sample-form-1.png): 6
    // artifacts over 46 runs = 0.13, so this leaves usable headroom above that.
    float gateRatio = 0.30f;
    // Minimum number of lines for the gate to apply; short pages don't have
    // enough of a sample and are always cleaned.
    int gateMinLines = 5;
    // By default only an unmatched bracket at the very start or end of a run
    // is removed, which is where the cell-border artifact always lands. Set
    // this to also remove interior ones (higher recall, but it will eat a
    // legitimate "(" whose ")" the OCR dropped).
    bool anywhere = false;
};

// Strips unmatched ()[]{} from each string in `lines`, in place, unless the
// per-kind veto or the per-page gate spares them (see above). Matching is per
// line/string: a bracket must be balanced within the same entry to be an
// outright keep; anything unmatched then goes through the veto and the gate.
BracketCleanupResult cleanOrphanBrackets(std::vector<std::string> &lines, const BracketCleanupParams &params);

#endif //__PADDLEVINO_BRACKET_CLEANER_H__
