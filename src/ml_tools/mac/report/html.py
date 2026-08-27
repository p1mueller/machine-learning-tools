"""HTML report adapter."""

import html
from pathlib import Path

from .core import MACReport, fmt, format_indices

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "report.html"


class HTMLReport(MACReport):
    """Renderer for the self-contained HTML representation of a :class:`MACReport`."""

    def render(self, max_indices: int = 200) -> str:
        """Render the report as a self-contained HTML document.

        If :attr:`figures` is populated, the diagnostic plots are embedded as
        inline base64 PNG images. The document carries no external
        dependencies, so it can be opened directly in a browser or shared.

        Args:
            max_indices: Maximum number of sample indices listed per category.
        """
        replacements = {
            "__TITLE__": self.title,
            "__STATS_ROWS__": self._stats_rows(),
            "__PREDICTOR_ROWS__": self._predictor_rows(),
            "__PROBLEM_ROWS__": self._problem_rows(max_indices),
            "__FIGURES_SECTION__": self._figures_section(),
        }
        template = _TEMPLATE_PATH.read_text(encoding="utf-8")
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        return template

    def _stats_rows(self) -> str:
        esc = html.escape
        return "\n".join(
            f"<tr><td>{esc(label)}</td><td class='value'>{esc(fmt(value))}</td></tr>"
            for label, value in self._stat_pairs()
        )

    def _predictor_rows(self) -> str:
        esc = html.escape
        rows = []
        for p in self.predictors:
            if p.flagged:
                status = f"<td class='flagged'>flagged &gt;{self.vif_threshold:g}</td>"
            else:
                status = "<td class='ok'>ok</td>"
            rows.append(
                f"<tr><td>{esc(p.name)}</td><td class='value'>{fmt(p.vif)}</td>{status}</tr>"
            )
        return "\n".join(rows)

    def _problem_rows(self, max_indices: int) -> str:
        esc = html.escape
        return "\n".join(
            "<tr>"
            f"<td>{esc(label)}</td>"
            f"<td class='value'>{len(indices)}</td>"
            f"<td>{esc(format_indices(indices, max_indices))}</td>"
            "</tr>"
            for label, indices in self.problems.items()
        )

    def _figures_section(self) -> str:
        esc = html.escape
        if not self.figures:
            return ""
        figures_html = "\n".join(
            f"<figure><img src='{f.image}' alt='{esc(f.title)}'></figure>" for f in self.figures
        )
        return (
            "<section><h2>Diagnostic Plots</h2><div class='figures'>\n"
            f"{figures_html}\n</div></section>"
        )
