"""Tests for the MAC report adapters (text, markdown, html)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from ml_tools import (
    FigureImage,
    HTMLReport,
    MACReport,
    MarkdownReport,
    ProblemSummary,
    TextReport,
)
from ml_tools.mac.analyze import ModelAdequacyChecker
from ml_tools.mac.config import MACConfig
from ml_tools.mac.report import fmt, format_indices, render_plotters, vif_status


def _make_data(n=80, seed=0):
    """Linear data with one redundant (high-VIF) column and injected outliers."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x[:, 2] = x[:, 0] + 0.05 * rng.normal(size=n)  # collinear -> high VIF
    beta = np.array([1.0, -2.0, 0.5])
    y = x @ beta + rng.normal(size=n)
    y[7] += 8.0  # outlier
    return x, y


@pytest.fixture(scope="session")
def result():
    """Run a full sklearn analysis with a strict config once per test session."""
    x, y = _make_data()
    config = MACConfig(t_threshold=1.5, cook_distance_threshold=0.01, vif_threshold=10.0)
    checker = ModelAdequacyChecker(config=config)
    model = LinearRegression().fit(x, y)
    return checker.analyze_sklearn(x, y, model, plot=False)


@pytest.fixture(scope="session")
def report(result):
    """Build a report once per session (renders all six figures)."""
    metric, masks, plots = result
    return HTMLReport.from_analysis(metric, masks, plots)


def _as(cls, report: "MACReport") -> "MACReport":
    """Rebuild the report as a different adapter class (no figure re-rendering)."""
    return cls(**report.model_dump())


def _report_with_problems(
    outliers: list[int] | None = None,
    high_leverage: list[int] | None = None,
    influential: list[int] | None = None,
) -> "TextReport":
    """A minimal report carrying the given problem indices."""
    return TextReport(
        n_samples=10,
        n_features=2,
        has_intercept=True,
        dof=3,
        r_squared=0.5,
        adj_r_squared=0.45,
        rse=1.25,
        rss=9.0,
        tss=9.0,
        f_statistic=2.0,
        residual_correlation=0.1,
        vif_threshold=10.0,
        predictors=[
            {"name": "x0", "vif": 1.1, "flagged": False},
            {"name": "x1", "vif": 12.0, "flagged": True},
        ],
        problems=ProblemSummary(
            outliers=outliers or [],
            high_leverage=high_leverage or [],
            influential=influential or [],
        ),
    )


class TestConstruction:
    """Construction and data access of the report."""

    def test_from_analysis_fields_match_metric(self, result, report) -> None:
        """All statistics are copied from the MetricSummary."""
        metric, _, _ = result
        assert report.n_samples == metric.n_samples
        assert report.n_features == 3
        assert report.dof == 4
        assert np.isclose(report.r_squared, metric.r_squared)
        assert np.isclose(report.adj_r_squared, metric.adj_r_squared)
        assert np.isclose(report.rse, metric.rse)
        assert np.isclose(report.rss, metric.rss)
        assert np.isclose(report.tss, metric.tss)
        assert np.isclose(report.f_statistic, metric.f_statistic)
        assert np.isclose(report.model_p_value, metric.model_p_value)
        assert np.isclose(report.residual_correlation, metric.residual_correlation)
        assert all(
            np.isclose(p.p_value, pv)
            for p, pv in zip(report.predictors, metric.coefficient_p_values[1:])
        )

    def test_from_analysis_predictors_and_problems(self, result, report) -> None:
        """VIF table and problem indices are derived from metric and masks."""
        _, masks, _ = result
        assert [p.name for p in report.predictors] == ["x0", "x1", "x2"]
        # collinear column must have a very high VIF and be flagged
        assert max(p.vif for p in report.predictors) > 1.0
        assert report.predictors[2].flagged
        assert 7 in report.problems.outliers
        assert report.problems.total == len(report.problems.outliers) + len(
            report.problems.high_leverage
        ) + len(report.problems.influential)
        assert bool(masks.outliers[7])

    def test_from_analysis_returns_subclass(self, report) -> None:
        """from_analysis returns the requested adapter type."""
        assert isinstance(report, HTMLReport)
        assert isinstance(_as(TextReport, report), TextReport)
        assert isinstance(_as(MarkdownReport, report), MarkdownReport)

    def test_from_analysis_renders_figures(self, report) -> None:
        """The six diagnostic plots are captured as base64 data URIs."""
        assert len(report.figures) == 6
        assert all(f.image.startswith("data:image/png;base64,") for f in report.figures)

    def test_config_threshold_is_stored(self, report) -> None:
        """The vif_threshold field reflects the config used."""
        assert report.vif_threshold == 10.0
        assert any(p.flagged for p in report.predictors)

    def test_add_figures_is_non_mutating(self, report) -> None:
        """add_figures returns a new report with the combined figures."""
        extra = [FigureImage(title="extra", image="data:image/png;base64,AAAA")]
        combined = report.add_figures(extra)
        assert len(combined.figures) == len(report.figures) + 1
        assert combined.figures[-1] == extra[0]
        assert combined is not report

    def test_serialisation_helpers(self, report) -> None:
        """to_dict / to_dataframe / to_df_vif expose the same data."""
        d = report.to_dict()
        df = report.to_dataframe()
        vif_df = report.to_df_vif()
        assert d["n_samples"] == int(df["n_samples"].iloc[0])
        assert np.isclose(float(d["rse"]), float(df["rse"].iloc[0]))
        assert list(vif_df.index) == ["x0", "x1", "x2"]
        assert "flagged" in vif_df.columns


class TestTextReport:
    """Plain-text adapter."""

    def test_sections_and_columns(self, report) -> None:
        """All expected sections are present."""
        text = _as(TextReport, report).render()
        for section in [
            "Model Adequacy Report",
            "Model statistics:",
            "Predictors:",
            "Problematic samples (",
            "Outliers",
            "High leverage",
            "Influential",
            "VIF",
        ]:
            assert section in text

    def test_flagged_predictor_marked(self) -> None:
        """Flagged predictors show the threshold in the status column."""
        text = _report_with_problems().render()
        assert "flagged (>10)" in text
        assert "ok" in text

    def test_truncates_long_index_lists(self) -> None:
        """Index lists longer than max_indices are truncated with a hint."""
        text = _report_with_problems(outliers=list(range(30))).render(max_indices=5)
        assert "0, 1, 2, 3, 4, ... (+25 more)" in text

    def test_empty_problems_render_none(self) -> None:
        """Categories without problems render as 'none'."""
        text = _report_with_problems().render()
        line = next(line for line in text.splitlines() if line.strip().startswith("Outliers"))
        assert line.split() == ["Outliers", "none"]

    def test_str_calls_render(self, report) -> None:
        """str(report) delegates to render()."""
        text_report = _as(TextReport, report)
        assert str(text_report) == text_report.render()


class TestMarkdownReport:
    """Markdown adapter."""

    def test_structure(self, report) -> None:
        """The markdown document has the expected headings and tables."""
        md = _as(MarkdownReport, report).render()
        assert md.startswith("# Model Adequacy Report")
        for section in [
            "| Metric | Value |",
            "| Name | VIF | p-value | Status |",
            "## Problematic Samples",
        ]:
            assert section in md
        assert md.endswith("\n")

    def test_vif_statuses(self, report) -> None:
        """Flagged and clean predictors get the right status text."""
        md = _as(MarkdownReport, report).render()
        assert ":warning: flagged (>10)" in md
        assert "ok" in md

    def test_truncates_index_lists(self) -> None:
        """Index lists are truncated with a hint."""
        md = _as(MarkdownReport, _report_with_problems(influential=list(range(40)))).render(
            max_indices=3
        )
        assert "0, 1, 2, ... (+37 more)" in md


class TestHTMLReport:
    """HTML adapter."""

    def test_self_contained_document(self, report) -> None:
        """Rendered output is a complete document with no leftover placeholders."""
        html_doc = report.render()
        assert html_doc.startswith("<!DOCTYPE html>")
        assert html_doc.rstrip().endswith("</html>")
        assert "<style>" in html_doc
        assert "Model Adequacy Report" in html_doc
        for placeholder in [
            "__TITLE__",
            "__BADGE__",
            "__STATS_CARDS__",
            "__PREDICTOR_ROWS__",
            "__PROBLEM_ROWS__",
            "__FIGURES_SECTION__",
        ]:
            assert placeholder not in html_doc

    def test_embeds_all_figures(self, report) -> None:
        """Each diagnostic plot is embedded as a base64 image."""
        html_doc = report.render()
        assert html_doc.count("<figure>") == 6
        assert html_doc.count("data:image/png;base64,") == 6

    def test_no_figures_section_when_empty(self) -> None:
        """Without figures the Diagnostic Plots section is omitted."""
        html_doc = _as(HTMLReport, _report_with_problems()).render()
        assert "Diagnostic Plots" not in html_doc
        assert "<figure>" not in html_doc

    def test_special_characters_are_escaped(self) -> None:
        """Predictor names are HTML-escaped in the output."""
        report = _report_with_problems()
        report.predictors[0].name = "a<b&c"
        html_doc = _as(HTMLReport, report).render()
        assert "a<b&c" not in html_doc
        assert "a&lt;b&amp;c" in html_doc

    def test_problem_rows_show_counts_and_indices(self) -> None:
        """Problem rows contain the count and the index list."""
        html_doc = _as(HTMLReport, _report_with_problems(outliers=[3, 11, 27])).render()
        assert "<td class='value'>3</td>" in html_doc
        assert "3, 11, 27" in html_doc

    def test_truncation_applies(self) -> None:
        """Long index lists are truncated in the HTML output."""
        html_doc = _as(HTMLReport, _report_with_problems(outliers=list(range(50)))).render(
            max_indices=10
        )
        assert "... (+40 more)" in html_doc
        assert "48, 49" not in html_doc

    def test_default_max_indices_differ_per_adapter(self) -> None:
        """text/markdown truncate earlier than html by default."""
        report = _report_with_problems(outliers=list(range(500)))
        assert "... (+480 more)" in report.render()  # text: max 20
        assert "... (+480 more)" in _as(MarkdownReport, report).render()  # max 20
        assert "... (+300 more)" in _as(HTMLReport, report).render()  # max 200

    @pytest.mark.parametrize("cls", [TextReport, MarkdownReport, HTMLReport])
    def test_saves_rendered_content(self, cls: type[MACReport], report, tmp_path: "object") -> None:
        """Save writes exactly the rendered representation."""
        from pathlib import Path

        p = Path(str(tmp_path)) / "out"
        adapter = _as(cls, report)
        adapter.save(str(p))
        assert p.read_text(encoding="utf-8") == adapter.render()


class TestHelpers:
    """Shared formatting helpers."""

    def test_fmt(self) -> None:
        """Fmt compresses whole numbers and maps NaN to n/a."""
        assert fmt(1.0) == "1"
        assert fmt(42.0) == "42"
        assert fmt(3.14159) == "3.142"
        assert fmt(float("nan")) == "n/a"

    def test_format_indices(self) -> None:
        """format_indices truncates and reports the number dropped."""
        assert format_indices([], 10) == "none"
        assert format_indices([1, 2], 10) == "1, 2"
        assert format_indices(list(range(6)), 2) == "0, 1, ... (+4 more)"

    def test_vif_status(self) -> None:
        """vif_status compares against the threshold."""
        assert vif_status(2.0, 5.0) == "ok"
        assert vif_status(6.0, 5.0) == ":warning: flagged (>5)"

    def test_problem_summary_aggregation(self) -> None:
        """ProblemSummary.total sums all categories in display order."""
        problems = ProblemSummary(outliers=[1], high_leverage=[2, 5], influential=[7])
        assert problems.total == 4
        assert [label for label, _ in problems.items()] == [
            "Outliers",
            "High leverage",
            "Influential",
        ]


class TestRenderPlotters:
    """Figure capture utility."""

    def test_returns_base64_images(self, report) -> None:
        """Every figure arrives as a base64 data URI."""
        assert len(report.figures) == 6
        assert all(isinstance(f, FigureImage) for f in report.figures)
        assert all(f.image.startswith("data:image/png;base64,") for f in report.figures)

    def test_reuses_cached_figures(self, result, report) -> None:
        """render_plotters on already-rendered plotters adds no new figures."""
        _, _, plots = result
        before = set(plt.get_fignums())
        figures = render_plotters(plots, dpi=10)
        assert len(figures) == 6
        assert set(plt.get_fignums()) == before

    def test_accepts_plain_dict_and_title_overrides(self, result) -> None:
        """A plain dict of plotters works and titles can be overridden."""
        _, masks, plots = result
        figures = render_plotters(
            {"residuals": plots.residuals},
            masks=masks,
            dpi=10,
            titles={"residuals": "My Residuals"},
        )
        assert [f.title for f in figures] == ["My Residuals"]
        plt.close("all")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
