#include "BracketCleaner.h"

namespace {

// The three bracket kinds, kept separate because the page-level veto below
// decides per kind, not globally.
enum Kind { Paren = 0, Square = 1, Curly = 2, KindCount = 3 };

int kindOf(char c) {
    switch (c) {
        case '(':
        case ')': return Paren;
        case '[':
        case ']': return Square;
        case '{':
        case '}': return Curly;
        default: return -1;
    }
}

char matchOpen(char close) {
    switch (close) {
        case ')': return '(';
        case ']': return '[';
        case '}': return '{';
        default: return '\0';
    }
}

bool isOpen(char c) { return c == '(' || c == '[' || c == '{'; }

bool isClose(char c) { return c == ')' || c == ']' || c == '}'; }

// Stack-based balance check over a single line: push opens, pop on a matching
// close. Whatever is left unpaired -- opens still on the stack, or closes that
// never found their open -- is a *candidate* for removal, not yet a decision:
// the caller still applies the page-level veto.
//
// Fills `remove` with the candidate positions and tallies, per bracket kind,
// how many unmatched opens and closes this line contributed.
void markCandidates(const std::string &line, std::vector<bool> &remove,
                    int openCount[KindCount], int closeCount[KindCount]) {
    remove.assign(line.size(), false);
    std::vector<size_t> stack; // indices of unmatched opens
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (isOpen(c)) {
            stack.push_back(i);
        } else if (isClose(c)) {
            if (!stack.empty() && line[stack.back()] == matchOpen(c)) {
                stack.pop_back();
            } else {
                remove[i] = true;
                ++closeCount[kindOf(c)];
            }
        }
    }
    for (size_t idx: stack) {
        remove[idx] = true;
        ++openCount[kindOf(line[idx])];
    }
}

std::string applyRemoval(const std::string &line, const std::vector<bool> &remove) {
    std::string out;
    out.reserve(line.size());
    for (size_t i = 0; i < line.size(); ++i) {
        if (!remove[i]) out += line[i];
    }
    return out;
}

} // namespace

BracketCleanupResult cleanOrphanBrackets(std::vector<std::string> &lines, const BracketCleanupParams &params) {
    if (!params.enabled) return BracketCleanupResult::Off;

    int openCount[KindCount] = {0, 0, 0};
    int closeCount[KindCount] = {0, 0, 0};
    std::vector<std::vector<bool>> removals(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        markCandidates(lines[i], removals[i], openCount, closeCount);
    }

    // Page-level veto, per bracket kind. A pair that legitimately spans two
    // lines -- "Preferred Contact Method (select one of the" / "following
    // options)", measured on a real form -- looks orphan to the per-line check on both
    // lines, and stripping it would corrupt good text. So if a kind has BOTH
    // an unmatched open and an unmatched close somewhere on the page, its
    // candidates are most likely two halves of such a pair: keep them all.
    //
    // The genuine OCR artifact has the opposite signature -- a border misread
    // as "[" produces opens with no closing partner anywhere on the page --
    // which is exactly what survives this veto.
    //
    // Comparing counts rather than positions keeps the decision independent
    // of the order the detector emitted the text runs in.
    bool vetoed[KindCount];
    for (int k = 0; k < KindCount; ++k) {
        vetoed[k] = openCount[k] > 0 && closeCount[k] > 0;
    }

    int linesWithOrphans = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        bool any = false;
        for (size_t j = 0; j < removals[i].size(); ++j) {
            if (!removals[i][j]) continue;
            // Edge restriction: the artifact is a cell border that fell inside
            // the detected box, so it lands at one end of the run -- measured,
            // every real one does ("ID[" trailing, "]Address" leading). An
            // unmatched bracket in the middle is far more likely to be real
            // text whose partner the OCR simply dropped ("... Amount Due
            // ($", where the ")" was lost), so it is left alone.
            const bool atEdge = (j == 0) || (j + 1 == lines[i].size());
            if (vetoed[kindOf(lines[i][j])] || (!params.anywhere && !atEdge)) {
                removals[i][j] = false;
            } else {
                any = true;
            }
        }
        if (any) ++linesWithOrphans;
    }

    if (linesWithOrphans == 0) return BracketCleanupResult::Applied;

    if ((int) lines.size() >= params.gateMinLines) {
        float ratio = (float) linesWithOrphans / (float) lines.size();
        if (ratio > params.gateRatio) {
            // Even after the veto, unmatched brackets dominate the page --
            // treat it as source code and leave everything untouched.
            return BracketCleanupResult::SkippedCodeGate;
        }
    }

    for (size_t i = 0; i < lines.size(); ++i) {
        lines[i] = applyRemoval(lines[i], removals[i]);
    }
    return BracketCleanupResult::Applied;
}
