"""HTML report adapter."""

import html
from pathlib import Path

from .core import MACReport, fmt, format_indices

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "report.html"

_BADGE_OK = "<span class='badge ok'>No problematic samples</span>"


def _badge(report_total: int) -> str:
    """Status badge for the report masthead."""
    if report_total:
        return f"<span class='badge warn'>{report_total} flagged sample(s)</span>"
    return _BADGE_OK


class HTMLReport(MACReport):
    """Renderer for the self-contained HTML representation of a [`MACReport`][..MACReport]."""

    def render(self, max_indices: int = 200) -> str:
        """Render the report as a self-contained HTML document.

        If [`figures`][.] is populated, the diagnostic plots are embedded as
        inline base64 PNG images. The document carries no external
        dependencies, so it can be opened directly in a browser or shared.

        Args:
            max_indices: Maximum number of sample indices listed per category.
        """
        esc = html.escape
        replacements = {
            "__TITLE__": esc(self.title),
            "__BADGE__": _badge(self.problems.total),
            "__STATS_CARDS__": self._stat_cards(),
            "__PREDICTOR_ROWS__": self._predictor_rows(),
            "__PROBLEM_ROWS__": self._problem_rows(max_indices),
            "__FIGURES_SECTION__": self._figures_section(),
        }
        template = _TEMPLATE_PATH.read_text(encoding="utf-8")
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        return template

    def _stat_cards(self) -> str:
        esc = html.escape
        return "\n".join(
            "<div class='stat'>"
            f"<div class='stat-value'>{esc(fmt(value))}</div>"
            f"<div class='stat-label'>{esc(label)}</div>"
            "</div>"
            for label, value in self._stat_pairs()
        )

    def _predictor_rows(self) -> str:
        esc = html.escape
        rows = []
        for p in self.predictors:
            if p.flagged:
                status = (
                    f"<td><span class='chip flagged'>VIF &gt; {self.vif_threshold:g}</span></td>"
                )
            else:
                status = "<td><span class='chip ok'>ok</span></td>"
            rows.append(
                "<tr>"
                f"<td>{esc(p.name)}</td>"
                f"<td class='value'>{fmt(p.vif)}</td>"
                f"<td class='value'>{fmt(p.p_value)}</td>"
                f"{status}"
                "</tr>"
            )
        return "\n".join(rows)

    def _problem_rows(self, max_indices: int) -> str:
        esc = html.escape
        return "\n".join(
            "<tr>"
            f"<td>{esc(label)}</td>"
            f"<td class='value'>{len(indices)}</td>"
            f"<td class='indices'>{esc(format_indices(indices, max_indices))}</td>"
            "</tr>"
            for label, indices in self.problems.items()
        )

    def _figures_section(self) -> str:
        esc = html.escape
        if not self.figures:
            return ""
        figures_html = "\n".join(
            "<figure>"
            f"<img src='{f.image}' alt='{esc(f.title)}'>"
            # f"<figcaption>{esc(f.title)}</figcaption>"
            "</figure>"
            for f in self.figures
        )
        return (
            "<section><h2>Diagnostic Plots</h2><div class='figures'>\n"
            f"{figures_html}\n</div></section>"
        )
