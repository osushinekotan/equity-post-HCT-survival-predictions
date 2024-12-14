import abc
from itertools import combinations

import numpy as np
import polars as pl


class BaseEncoder:
    fitted: bool
    prefix: str

    @abc.abstractmethod
    def fit(self, df: pl.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)


class AggregateEncoder(BaseEncoder):
    def __init__(
        self,
        group_keys: list[str] | str,
        agg_exprs: list[pl.Expr],
        prefix: str = "f_",
    ) -> None:
        """aggregate numerical columns by group keys

        Args:
            group_keys (list[str] | str): list or single string of column names to group by
            agg_exprs (list[pl.Expr]): list of aggregation expressions (e.g. pl.col("colname").mean().alias("mean_colname"))
                - set alias as the output column name
                - `agg_{expr_alias}_{group_key_name}` will be used as the output column name
        """
        self.group_keys = group_keys
        self.agg_exprs = agg_exprs
        self.prefix = prefix

        self.mapping_df: pl.DataFrame | None = None
        self.fitted = False

    @property
    def group_key_name(self) -> str:
        if isinstance(self.group_keys, str):
            return self.group_keys
        return "_x_".join(self.group_keys)

    @property
    def aggregated_colnames(self) -> dict[str, str]:
        return {name: f"{name}_grpby_{self.group_key_name}" for name in self.expr_output_names}

    @property
    def expr_output_names(self) -> list[str]:
        return [expr.meta.output_name() for expr in self.agg_exprs]

    def fit(self, df: pl.DataFrame) -> None:
        agg_exprs = self.agg_exprs.copy()
        first_expr = agg_exprs.pop(0)
        self.mapping_df = df.group_by(self.group_keys, maintain_order=True).agg(first_expr)
        for expr in agg_exprs:
            self.mapping_df = self.mapping_df.join(
                df.group_by(self.group_keys, maintain_order=True).agg(expr),
                on=self.group_keys,
                how="left",
            )

        self.mapping_df = self.mapping_df.rename(self.aggregated_colnames)
        self.fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise ValueError("fit() method should be called before transform()")
        return (
            df.join(self.mapping_df, on=self.group_keys, how="left")
            .select(list(self.aggregated_colnames.values()))
            .select(pl.all().name.prefix(self.prefix))
        )

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)


class RawEncoder(BaseEncoder):
    def __init__(self, columns: list[str], prefix: str = "f_") -> None:
        self.columns = columns
        self.fitted = False
        self.prefix = prefix

    def fit(self, df: pl.DataFrame) -> None:
        self.fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise ValueError("fit() method should be called before transform()")
        return df.select(self.columns).select(pl.all().name.prefix(self.prefix))


class NullFlagEncoder(BaseEncoder):
    def __init__(
        self,
        columns: list[str] | None = None,
        prefix: str = "f_",
        suffix: str = "_is_null",
    ) -> None:
        self.columns = columns or []
        self.fitted = False
        self.prefix = prefix

        self.suffix = suffix

    def fit(self, df: pl.DataFrame) -> None:
        self.fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if len(self.columns) == 0:
            self.columns = df.columns

        return (
            df.select(pl.col(self.columns).is_null())
            .select(pl.all().name.prefix(self.prefix))
            .select(pl.all().name.prefix(self.suffix))
        )


class ArithmeticCombinationEncoder(BaseEncoder):
    """Calculate features by arithmetic combinations of numerical columns using polars."""

    def __init__(
        self,
        input_cols=None,
        include_cols=None,
        exclude_cols=None,
        operator="+",
        output_prefix="",
        output_suffix="_combi",
        drop_origin=False,
        r=2,
    ):
        self._input_cols = input_cols or []
        self._include_cols = include_cols or []
        self._exclude_cols = exclude_cols or []
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._r = r
        self._operator = operator
        self._drop_origin = drop_origin

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit to data frame, then transform it.

        Args:
            df (pl.DataFrame): Input data frame.
        Returns:
            pl.DataFrame: Output data frame.
        """
        new_df = df.clone()

        if not self._input_cols:
            self._input_cols = [col for col in new_df.columns if col not in self._exclude_cols]

        return self.transform(new_df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform data frame.

        Args:
            df (pl.DataFrame): Input data frame.
        """
        new_df = df.clone()
        generated_cols = []

        n_fixed_cols = len(self._include_cols)

        for cols_pairs in combinations(self._input_cols, r=self._r - n_fixed_cols):
            fixed_cols_str = f"{self._operator}".join(self._include_cols)
            pairs_cols_str = f"{self._operator}".join(sorted(list(cols_pairs)))
            new_col = (
                self._output_prefix
                + fixed_cols_str
                + (self._operator if fixed_cols_str else "")
                + pairs_cols_str
                + self._output_suffix
            )
            if new_col in generated_cols:
                continue

            concat_cols = self._include_cols + list(cols_pairs)

            # Perform arithmetic operation
            if self._operator == "+":
                new_df = new_df.with_columns(pl.sum_horizontal(*concat_cols).alias(new_col))
            elif self._operator == "*":
                arr = new_df.select(concat_cols).to_numpy()
                new_df = new_df.with_columns(pl.Series(np.prod(arr, axis=1)).alias(new_col))
            else:
                raise ValueError(f"Operator {self._operator} is not supported.")

            generated_cols.append(new_col)

        if self._drop_origin:
            return new_df.select(generated_cols)

        return new_df


class StringCombinationEncoder(BaseEncoder):
    """Create new features by concatenating string columns using polars."""

    def __init__(
        self,
        input_cols=None,
        include_cols=None,
        exclude_cols=None,
        separator=" ",
        output_prefix="",
        output_suffix="_concat",
        drop_origin=False,
        r=2,
    ):
        self._input_cols = input_cols or []
        self._include_cols = include_cols or []
        self._exclude_cols = exclude_cols or []
        self._separator = separator
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._drop_origin = drop_origin
        self._r = r

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit to data frame, then transform it.

        Args:
            df (pl.DataFrame): Input data frame.
        Returns:
            pl.DataFrame: Output data frame.
        """
        new_df = df.clone()

        if not self._input_cols:
            self._input_cols = [
                col for col in new_df.columns if col not in self._exclude_cols and new_df[col].dtype == pl.Utf8
            ]

        return self.transform(new_df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform data frame.

        Args:
            df (pl.DataFrame): Input data frame.
        """
        new_df = df.clone()
        generated_cols = []

        n_fixed_cols = len(self._include_cols)

        for cols_pairs in combinations(self._input_cols, r=self._r - n_fixed_cols):
            concat_cols = self._include_cols + list(cols_pairs)
            cols_str = self._separator.join(concat_cols)
            new_col = self._output_prefix + cols_str + self._output_suffix
            if new_col in generated_cols:
                continue

            # Perform string concatenation
            new_df = new_df.with_columns(pl.concat_str(concat_cols, separator=self._separator).alias(new_col))

            generated_cols.append(new_col)

        if self._drop_origin:
            return new_df.select(generated_cols)

        return new_df
