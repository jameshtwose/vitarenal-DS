from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def diagnostic_plots(
    model_fit: "fitted linear model" = None,  # type: ignore
    X: Union[pd.DataFrame, None] = None,
    y: Union[pd.Series, None] = None,
    figsize: tuple = (12, 12),
    limit_cooks_plot: bool = False,
    subplot_adjust_args: dict = {"wspace": 0.3, "hspace": 0.3},
):

    r"""This function is an extension of the diagnostic_plots function. This is done to include the use of models
    defined using pymer4, statsmodels, and sklearn*. Specifically the currently accepted models are:
    - pymer4.models.Lm.Lm
    - pymer4.models.Lmer.Lmer
    - sklearn.linear_model._base.LinearRegression
    - sklearn.linear_model._coordinate_descent.Lasso
    - sklearn.linear_model._ridge.Ridge
    - statsmodels.regression.linear_model.RegressionResultsWrapper
    - statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper
    *The sklearn models listed have been tried, although in theory all variants of the sklearn linear regression
     models should work with this function
    Parameters
    ----------
    model_fit:
        A fitted linear regression model, ideally one of the ones listed above.
    X:
        The array of predictors used to fit the model. This is only required if the inputted model is an sklearn
        model. For all other models just the fitted model is sufficient.
    y:
        The target array used to fit the model. This is only required if the inputted model is an sklearn
        model. For all other models just the fitted model is sufficient.
    figsize: tuple
        Width and height of the figure in inches
    limit_cooks_plot: bool
        Whether to apply a y-limit to the cooks distances plot (i.e. would you like to see the cooks distances
        better, or the individual scatter points better?)
    subplot_adjust_args: dict
        A dictionary of arguments to change the dimensions of the subplots. This is useful to include if the chosen
        figsize is making the plot labels overlap.
    Returns
    ----------
    diagplot: matplotlib.figure.Figure
        Figure with four diagnostic plots: residuals vs fitted, QQplot, scale location, residuals vs leverage
    axes: np.array
        An array of the associated axes of the four subplots.
    Examples
    ----------
    >>> ## Pymer4 example
    >>> import seaborn as sns
    >>> from pymer4 import Lmer
    >>> from neuropy.frequentist_statistics import diagnostic_plots
    >>> data = sns.load_dataset(name="mpg")
    >>> model_fit = Lmer("mpg ~ cylinders + displacement + weight + acceleration + (1 | model_year)", data=data)
    >>> model_summary = model_fit.fit(conf_int = "profile")
    >>> fig, axs = diagnostic_plots(model_fit=model_fit,
    ...                                         X=None,
    ...                                         y=None,
    ...                                         figsize = (8,8),
    ...                                         limit_cooks_plot = False,
    ...                                         subplot_adjust_args={"wspace": 0.3, "hspace": 0.3}
    ...                                        )
    >>> ## sklearn example
    >>> import seaborn as sns
    >>> from sklearn.linear_model import LinearRegression
    >>> from neuropy.frequentist_statistics import diagnostic_plots
    >>> data = sns.load_dataset(name="mpg")
    >>> X = data[['cylinders', 'displacement', 'weight', 'acceleration']]
    >>> y = data["mpg"]
    >>> model_fit = LinearRegression().fit(X, y)
    >>> fig, axs = diagnostic_plots(model_fit=model_fit,
    ...                                         X=X,
    ...                                         y=y,
    ...                                         figsize = (8,8),
    ...                                         limit_cooks_plot = False,
    ...                                         subplot_adjust_args={"wspace": 0.3, "hspace": 0.3}
    ...                                        )
    >>> ## statsmodels example
    >>> import seaborn as sns
    >>> import statsmodels.api as sm
    >>> from neuropy.frequentist_statistics import diagnostic_plots
    >>> data = sns.load_dataset(name="mpg")
    >>> X = data[['cylinders', 'displacement', 'weight', 'acceleration']]
    >>> y = data["mpg"]
    >>> model_fit = sm.MixedLM(endog=y, exog=X, groups=data["model_year"]).fit(method=["lbfgs"])
    >>> fig, axs = diagnostic_plots(model_fit=model_fit,
    ...                                         X=None,
    ...                                         y=None,
    ...                                         figsize = (8,8),
    ...                                         limit_cooks_plot = False,
    ...                                         subplot_adjust_args={"wspace": 0.3, "hspace": 0.3}
    ...                                        )
    """

    def _get_model_fit(model_fit):
        source = str(type(model_fit))

        if not any(
            substring in source for substring in ["pymer4", "sklearn", "statsmodels"]
        ):
            raise ValueError(
                f"The model {source}, is currently not supported by this function."
            )

        if "pymer4" in source:
            model_fitted_y = pd.Series(model_fit.fits)
            model_residuals = pd.Series(model_fit.residuals)
            model_norm_residuals = (
                pd.Series(model_fit.residuals) / model_fit.residuals.std()
            )

            model_abs_resid = np.abs(model_residuals)
            model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

        if "sklearn" in source:

            if X is None and not any(
                substring in str(type(X))
                for substring in ["DataFrame", "Series", "ndarray"]
            ):
                raise ValueError(
                    f"""The fitted model specified was an sklearn model which requires X to not be None and one of
                    either a pandas DataFrame or Series, or a numpy ndarray. The X supplied was of type: {str(type(X))}"""
                )

            if y is None and not any(
                substring in str(type(y)) for substring in ["Series", "ndarray"]
            ):
                raise ValueError(
                    f"""The fitted model specified was an sklearn model which requires y to not be None and one of
                    either a pandas Series, or a numpy ndarray. The y supplied was of type: {str(type(y))}"""
                )

            model_fitted_y = model_fit.predict(X)
            model_residuals = y - model_fit.predict(X)
            model_norm_residuals = model_residuals / model_residuals.std()

            model_abs_resid = np.abs(model_residuals)
            model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

        if "statsmodels" in source:
            # model values
            model_fitted_y = model_fit.fittedvalues
            # model residuals
            model_residuals = model_fit.resid
            # absolute residuals
            model_abs_resid = np.abs(model_residuals)

            if "regression.mixed_linear_model" in source:
                # studentized residuals
                model_norm_residuals = model_residuals / model_residuals.std()
                # absolute square root residuals
                model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

            if "regression.linear_model" in source:
                # normalized residuals
                model_norm_residuals = pd.Series(
                    model_fit.get_influence().resid_studentized_internal
                )
                model_norm_residuals.index.name = "index"
                model_norm_residuals.name = "model_norm_residuals"

                # root square absolute normalized residuals
                model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

        return (
            model_fitted_y,
            model_residuals,
            model_norm_residuals,
            model_abs_resid,
            model_norm_residuals_abs_sqrt,
        )

    def _get_model_leverage(model_fit):
        source = str(type(model_fit))

        if "statsmodels.regression.linear" in source:
            # leverage, from statsmodels internals
            model_leverage = pd.Series(
                model_fit.get_influence().hat_matrix_diag.transpose()
            )

        else:
            if "pymer4" in source:
                if "Lm." in source:
                    temp = model_fit.design_matrix.drop("Intercept", axis=1).values
                if "Lmer" in source:
                    temp = model_fit.design_matrix.drop("(Intercept)", axis=1).values
            if "sklearn" in source:
                temp = X.values
            if "statsmodels" in source:
                # get the data used to predict the y (i.e. X or the exogenous variables)
                temp = model_fit.model.data.exog

            hat_temp = temp.dot(np.linalg.inv(temp.T.dot(temp)).dot(temp.T))
            hat_temp_diag = np.diagonal(hat_temp)
            model_leverage = pd.Series(hat_temp_diag)

        model_leverage.name = "model_leverage"
        model_leverage.index.name = "index"

        return model_leverage

    def _get_num_of_parameters(model_fit):
        source = str(type(model_fit))

        if "pymer4" in source:
            if "Lm." in source:
                number_of_parameters = model_fit.coefs.drop("Intercept").shape[0]
            if "Lmer" in source:
                if type(model_fit.ranef) == list:
                    number_of_parameters = len(model_fit.fixef[0].columns) + len(
                        list(model_fit.ranef)
                    )
                else:
                    number_of_parameters = len(model_fit.fixef.columns) + len(
                        model_fit.ranef.columns
                    )

        if "sklearn" in source:
            number_of_parameters = len(model_fit.coef_)

        if "statsmodels" in source:
            number_of_parameters = len(model_fit.params)

        return number_of_parameters

    (
        model_fitted_y,
        model_residuals,
        model_norm_residuals,
        model_abs_resid,
        model_norm_residuals_abs_sqrt,
    ) = _get_model_fit(model_fit)
    model_leverage = _get_model_leverage(model_fit)
    number_of_parameters = _get_num_of_parameters(model_fit)

    ## PLOTTING STUFF

    # create figure with 4 subplots
    diagplot, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    #     _ = plt.subplots_adjust(wspace=0.6, hspace=0.6)
    _ = plt.subplots_adjust(**subplot_adjust_args)
    _ = plt.suptitle("Model diagnostics")
    # First plot: Residuals vs fitted
    _ = sns.regplot(
        x=model_fitted_y,
        y=model_residuals,
        scatter=True,
        lowess=True,
        line_kws={"color": "orange", "lw": 1, "alpha": 0.8},
        scatter_kws={"alpha": 0.3},
        ax=diagplot.axes[0],
    )
    x_range = np.linspace(min(model_fitted_y), max(model_fitted_y), 50)
    _ = diagplot.axes[0].plot(
        x_range,
        np.repeat(0, len(x_range)),
        lw=1,
        ls=":",
        color="grey",
    )
    _ = diagplot.axes[0].set_title("Residuals vs Fitted")
    _ = diagplot.axes[0].set_xlabel("Fitted values")
    _ = diagplot.axes[0].set_ylabel("Residuals")
    margin_res = 0.10 * (max(model_residuals) - min(model_residuals))
    _ = diagplot.axes[0].set_ylim(
        min(model_residuals) - margin_res, max(model_residuals) + margin_res
    )

    # annotations: top 3 absolute residuals
    abs_resid_top_3 = model_abs_resid.sort_values(ascending=False)[:3]
    for i in abs_resid_top_3.index:
        _ = diagplot.axes[0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]))

    res = stats.probplot(model_norm_residuals, dist="norm", plot=None, rvalue=True)
    ordered_theoretical_quantiles = res[0][0]
    ordered_residuals = res[0][1]
    slope = res[1][0]
    intercept = res[1][1]
    r_value = res[1][2]

    _ = diagplot.axes[1].scatter(
        ordered_theoretical_quantiles, ordered_residuals, alpha=0.3
    )
    _ = diagplot.axes[1].plot(
        ordered_theoretical_quantiles,
        slope * ordered_theoretical_quantiles + intercept,
        "orange",
    )
    _ = diagplot.axes[1].plot([], [], ls="", label=f"$R^2={round(r_value ** 2, 3)}$")
    _ = diagplot.axes[1].legend(loc="lower right")

    # _ = diagplot.axes[1].get_lines()[1].set_markerfacecolor(NeurocastColors.ORANGE)
    _ = diagplot.axes[1].set_title("Normal Q-Q")
    _ = diagplot.axes[1].set_xlabel("Theoretical Quantiles")
    _ = diagplot.axes[1].set_ylabel("Standardized Residuals")

    abs_norm_resid_top_3 = np.abs(model_norm_residuals).sort_values(ascending=False)[:3]
    norm_resid_top_3 = model_norm_residuals[abs_norm_resid_top_3.index]
    ordered_df = pd.DataFrame(
        {
            "ordered_residuals": ordered_residuals,
            "ordered_theoretical_quantiles": ordered_theoretical_quantiles,
        }
    )

    for i in abs_norm_resid_top_3.index:
        #         index = np.where(pd.Series(ordered_residuals).index == i)[0][0]

        _ = diagplot.axes[1].annotate(
            i,
            xy=(
                ordered_df.loc[
                    ordered_df["ordered_residuals"] == norm_resid_top_3[i],
                    "ordered_theoretical_quantiles",
                ].values[0],
                ordered_df.loc[
                    ordered_df["ordered_residuals"] == norm_resid_top_3[i],
                    "ordered_residuals",
                ].values[0],
            ),
        )

    # Third plot: scale location
    _ = sns.regplot(
        x=model_fitted_y,
        y=model_norm_residuals_abs_sqrt,
        scatter=True,
        ci=False,
        lowess=True,
        line_kws={"color": "orange", "lw": 1, "alpha": 0.8},
        scatter_kws={"alpha": 0.3},
        ax=diagplot.axes[2],
    )
    _ = diagplot.axes[2].set_title("Scale-Location")
    _ = diagplot.axes[2].set_xlabel("Fitted values")
    _ = diagplot.axes[2].set_ylabel(r"$\sqrt{|Standardized Residuals|}$")
    # annotations: top 3 absolute normalized residuals
    for i in abs_norm_resid_top_3.index:
        _ = diagplot.axes[2].annotate(
            i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i])
        )

    # Fourth plot: residuals vs leverages
    _ = sns.regplot(
        x=model_leverage,
        y=model_norm_residuals,
        scatter=True,
        ci=False,
        lowess=True,
        line_kws={"color": "orange", "lw": 1, "alpha": 0.8},
        scatter_kws={"alpha": 0.3},
        ax=diagplot.axes[3],
    )
    _ = diagplot.axes[3].set_xlim(0, max(model_leverage) + 0.01)

    if limit_cooks_plot:
        _ = diagplot.axes[3].set_ylim(
            min(model_norm_residuals) - 0.5, max(model_norm_residuals) + 0.5
        )
    _ = diagplot.axes[3].set_title("Residuals vs Leverage")
    _ = diagplot.axes[3].set_xlabel("Leverage")
    _ = diagplot.axes[3].set_ylabel("Standardized Residuals")

    # annotations: top 3 levarages
    leverage_top_3 = model_leverage.sort_values(ascending=False)[:3]
    for i in leverage_top_3.index:
        _ = diagplot.axes[3].annotate(
            i, xy=(model_leverage[i], model_norm_residuals[i])
        )
    # extra lines to indicate Cook's distances
    x_range = np.linspace(0.001, max(model_leverage), 50)

    def cooksdistances(boundary):
        return lambda x: np.sqrt((boundary * number_of_parameters * (1 - x)) / x)

    for line in [0.5, 1]:
        l_formula = cooksdistances(line)
        for place in [1, -1]:
            cooks_line = plt.plot(
                x_range,
                place * l_formula(x_range),
                lw=1,
                ls="--",
                color="orange",
            )
            y_text = place * l_formula(max(model_leverage) + 0.01)
            if (
                min(model_norm_residuals) - 0.5
                < y_text
                < max(model_norm_residuals) + 0.5
            ):
                _ = plt.text(
                    max(model_leverage) + 0.01,
                    y_text,
                    str(line),
                    color="orange",
                )
    _ = diagplot.axes[3].legend(
        cooks_line[:2], ["Cook's distance"], handlelength=3, loc="lower right"
    )

    return diagplot, axes