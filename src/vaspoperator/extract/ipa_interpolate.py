import logging

import polars as pl

from vaspoperator.calculation.ipa import StepIPA

logger = logging.getLogger("IPA interpolate")


def interpolate_ipa_to_wavelength(
    df_ipa_dependency: pl.LazyFrame, target_wl: float
):
    """
    Interpolate optical properties to a specific wavelength.

    Parameters:
    -----------
    df_ipa_dependency : polars.DataFrame or polars.LazyFrame
        Input DataFrame with wavelength-dependent optical properties
    target_wl : float
        Target wavelength in nanometers

    Returns:
    --------
    list[polars.DataFrame]
        List of DataFrames with interpolated values at target_wl for each group
    """
    df_wl = []

    df = (
        df_ipa_dependency
        if hasattr(df_ipa_dependency, "collect")
        else df_ipa_dependency.lazy()
    )

    group_cols = ["material_id", "TS"]
    value_cols = [
        col
        for col in df.collect_schema().names()
        if col not in group_cols + ["wavelength_nm"]
    ]
    schema = StepIPA.get_polars_schema()["dependency"]

    group_values = df.select(group_cols).unique().collect()

    for group_key in group_values.iter_rows(named=True):
        group_df = df.filter(
            (pl.col("material_id") == group_key["material_id"])
            & (pl.col("TS") == group_key["TS"])
        )

        null_row = {col: [None] for col in schema}
        null_row.update(
            {
                "material_id": [group_key["material_id"]],
                "TS": [group_key["TS"]],
                "wavelength_nm": [target_wl],
            }
        )

        df_null = pl.DataFrame(null_row, schema=schema).lazy()

        interpolated = (
            pl.concat([group_df, df_null])
            .with_columns(
                pl.col(col).interpolate_by("wavelength_nm")
                for col in value_cols
            )
            .filter(pl.col("wavelength_nm") == target_wl)
        )

        df_wl.append(interpolated)

    return pl.concat(df_wl) if len(df_wl) > 0 else None
