"""Configuration module for MAC analysis and plotting."""

from pydantic import BaseModel, Field

Color = str | tuple[float, float, float] | tuple[float, float, float, float]


class Colors(BaseModel):
    """Color configuration for MAC plots."""

    marker: Color = Field("#00000000", description="Marker color")
    marker_edge: Color = Field("#0C273A", description="Marker edge color")
    bar: Color = Field("#0C273A", description="Bar color")
    target: Color = Field("#648FFF", description="Target color")
    highlight: Color = Field("#785EF000", description="Highlight color")
    highlight_edge: Color = Field("#785EF0", description="Highlight color")
    #
    outlier: Color = Field("#DC267F00", description="Color for outlier points")
    outlier_edge: Color = Field("#DC267F", description="Edge color for outlier points")
    high_leverage: Color = Field("#FFB00000", description="Color for high leverage points")
    high_leverage_edge: Color = Field("#FFB000", description="Edge color for high leverage points")
    influential: Color = Field("#FE610000", description="Color for influential points")
    influential_edge: Color = Field("#FE6100", description="Edge color for influential points")


class MACConfig(BaseModel):
    """Configuration for MAC analysis and plotting."""

    t_threshold: float = Field(3.5, description="Threshold for t-statistics to identify outliers")
    leverage_threshold_factor: float = Field(
        5.0, gt=1.0, description="Factor to determine leverage threshold for high leverage points"
    )
    cook_distance_threshold: float = Field(
        1.0, gt=0, description="Threshold for Cook's distance to identify influential points"
    )
    vif_threshold: float = Field(10.0, gt=1, description="Threshold for Variance Inflation Factor")
    vif_lower_bound: float = Field(
        5.0, gt=0, description="Lower bound for Variance Inflation Factor"
    )
    lowess_rel_delta: float = Field(
        0.05, ge=0.0, le=1.0, description="Relative delta for LOWESS smoothing"
    )
    lowess_frac: float = Field(
        2 / 3, gt=0, le=1, description="Fraction of data used for LOWESS smoothing"
    )
    marker: str = Field("o", description="Marker style for scatter plots")
    base_width: float = Field(5.5, gt=0, description="Base width for plots in inches")
    base_height: float = Field(4.5, gt=0, description="Base height for plots in inches")
    bar_height: float = Field(0.35, gt=0, description="Height of bars in bar plots")
    plot_bloat_height: float = Field(
        0.5, gt=0, description="Additional height added to plots for axes labels and titles"
    )
    grid_show: bool = Field(True, description="Whether to show grid lines in plots")
    grid_alpha: float = Field(0.5, ge=0.0, le=1.0, description="Alpha transparency for grid lines")
    grid_below: bool = Field(
        True, description="Whether to draw grid lines below other plot elements"
    )
    colors: Colors = Field(default_factory=Colors, description="Color configuration for plots")


def get_default_config() -> MACConfig:
    """Returns the default configuration for MAC analysis."""
    return MACConfig()
