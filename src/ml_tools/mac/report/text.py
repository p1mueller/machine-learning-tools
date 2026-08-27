"""Plain-text report adapter."""

from .core import MACReport, fmt, format_indices


class TextReport(MACReport):
    """Renderer for the plain-text representation of a [`MACReport`][..MACReport]."""

    def render(self, max_indices: int = 20) -> str:
        """Render the report as a plain text string.

        Args:
            max_indices: Maximum number of sample indices listed per category.
        """
        lines = [self.title, "=" * len(self.title), ""]
        lines += ["Model statistics:"]
        for label, value in self._stat_pairs():
            lines.append(f"  {label:<28} {fmt(value)}")
        lines.append("")
        lines.append("Predictors:")
        lines.append(f"  {'Name':<20} {'VIF':>12} {'p-value':>12}  Status")
        for p in self.predictors:
            status = f"flagged (>{self.vif_threshold:g})" if p.flagged else "ok"
            lines.append(f"  {p.name:<20} {fmt(p.vif):>12} {fmt(p.p_value):>12}  {status}")
        lines.append("")
        problems = self.problems
        lines.append(f"Problematic samples ({problems.total} flagged):")
        for label, indices in problems.items():
            lines.append(f"  {label:<16} {format_indices(indices, max_indices)}")
        return "\n".join(lines + [""])
