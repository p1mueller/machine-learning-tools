"""Markdown report adapter."""

from .core import MACReport, fmt, format_indices, vif_status


class MarkdownReport(MACReport):
    """Renderer for the Markdown representation of a [`MACReport`][..MACReport]."""

    def render(self, max_indices: int = 20) -> str:
        """Render the report as a Markdown string.

        Args:
            max_indices: Maximum number of sample indices listed per category.
        """
        lines = [f"# {self.title}", ""]
        lines += ["## Model Statistics", "", "| Metric | Value |", "| --- | --- |"]
        for label, value in self._stat_pairs():
            lines.append(f"| {label} | {fmt(value)} |")
        lines += [
            "",
            "## Predictors",
            "",
            "| Name | VIF | p-value | Status |",
            "| --- | --- | --- | --- |",
        ]
        vif_df = self.to_df_vif()
        for name, vif in zip(vif_df.index, vif_df["VIF"]):
            p_value = next(p.p_value for p in self.predictors if p.name == name)
            lines.append(
                f"| {name} | {fmt(vif)} | {fmt(p_value)} | {vif_status(vif, self.vif_threshold)} |"
            )
        lines += ["", f"## Problematic Samples ({self.problems.total} flagged)", ""]
        for label, indices in self.problems.items():
            lines.append(f"- **{label}** ({len(indices)}): {format_indices(indices, max_indices)}")
        lines.append("")
        return "\n".join(lines)
