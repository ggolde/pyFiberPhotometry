import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotnine import * #type: ignore
from plotnine.ggplot import ggplot as GGPlot
from plotnine.composition import Compose
from matplotlib import colors as mcolors

######################
#region --- THEMES ---
######################
def get_default_theme():
    default_text_theme = theme(
        text=element_text(color='black', family='Arial'),

        # title & axis labels
        plot_title=element_text(size=10, weight='bold'),
        axis_title_x=element_text(size=8),
        axis_title_y=element_text(size=8),

        # tick labels
        axis_text_x=element_text(size=6),
        axis_text_y=element_text(size=6),

        # strip
        strip_text=element_text(color='black', fontweight='bold', size=8),
    )

    default_legend_theme = theme(
        # text
        legend_title=element_text(size=8, weight='bold'),
        legend_text=element_text(size=6),
    )

    default_line_theme = theme(
        # global
        axis_line=element_line(color='black'),

        # tick marks
        axis_ticks=element_line(size=0.5),
        axis_ticks_length_major=2,
        axis_ticks_length_minor=0,

        # axis lines (spines)
        axis_line_x=element_line(size=0.5),
        axis_line_y=element_line(size=0.5),

        # disable grid
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),

        # disable background
        panel_background=element_blank(),
        plot_background=element_blank(),
    )

    default_strip_theme = theme(
        strip_background=element_blank(),

    )

    combined = (
        default_text_theme
        + default_line_theme
        + default_legend_theme
        + default_strip_theme
    )

    return combined

#endregion

##########################
#region --- EXPERIMENT ---
##########################
def plot_experiment_traces(
        df: pd.DataFrame,
        label_col: str,
        only_traces: list[str],
        title: str = '',
        x_label: str = 'Time (s)',
        y_label: str = 'Signal (a.u.)',
        legend_label: str = 'Source',
        color_values: dict[str, str] | None = None,
        line_kwargs: dict = {},
        theme_kwargs: dict = {},
        ) -> GGPlot:
    # clean df
    df = df.loc[df[label_col].isin(only_traces), :].copy()
    clean_labels = {label : label.replace('_', ' ').capitalize() for label in only_traces}

    # handle draw order
    draw_order = [
        'raw_signal', 
        'filtered_signal',
        'raw_isosbestic', 
        'filtered_isosbestic',
        'fitted_reference', 
        'processed_signal',
    ]
    draw_order = [label for label in draw_order if label in only_traces]
    df[label_col] = pd.Categorical(
        df[label_col],
        categories=draw_order,
        ordered=True
    )
        
    # defaults
    line_kwargs = dict() | line_kwargs
    theme_kwargs = dict() | theme_kwargs

    p = (
        ggplot(df, aes(x='time', y='value', color=label_col, group=label_col))
        + geom_line(**line_kwargs)
        + labs(
            title=title,
            x=x_label,
            y=y_label,
            color=legend_label,
        )
        + scale_color_manual(
            values=color_values,
            labels=clean_labels,
        )
        + get_default_theme()
    )

    return p

def plot_experiment_dashboard_full(
        time: np.ndarray,
        raw_sig: np.ndarray,
        raw_ref: np.ndarray | None,
        fit_ref: np.ndarray,
        processed_sig: np.ndarray,
        title: str = '',
        processed_units: str = 'Unspecified',
        ) -> Compose:
    
    top_df = pd.DataFrame({
        'time' : np.repeat(time, 4),
        'value' : np.concat([raw_sig, raw_ref, fit_ref]),
        'source' : np.concat([
            np.repeat('Raw signal', len(raw_sig)), 
            np.repeat('Raw reference', len(raw_ref)),
            np.repeat('Fitted reference', len(fit_ref)),
        ])
    })

    bottom_df = pd.DataFrame({
        'time' : time,
        'value' : processed_sig,
        'source' : 'Processed signal',
    })

    p_top = (
        ggplot(top_df, aes(x='time', y='value', color='source'))
        + geom_line(
            alpha=0.8
        )
        + labs(
            title=f'{title}\nRaw traces',
            x='Time (s)',
            y='Fluorescence intensity (a.u.)',
            color='Source',
        )
        + scale_color_discrete(
            values = {
                'Raw signal' : 'blue',
                'Raw reference' : 'darkgrey',
                'Fitted reference' : 'orange',
            }
        )
        + get_default_theme()
    )

    p_bot = (
        ggplot(bottom_df, aes(x='time', y='value', color='source'))
        + geom_line()
        + labs(
            title='Processed signal',
            x='Time (s)',
            y=processed_units,
            color='Source'
        )
        + scale_color_discrete(
            values = {
                'Processed signal' : 'blue',
            }
        )
        + get_default_theme()
    )

    p = (p_top / p_bot)
    return p

#endregion

##########################
#region --- TRIAL DATA ---
##########################
def plot_photometry_data(
        long_df: pd.DataFrame,
        label_col: str | None,
        group_col: str | None,
        err_layer: str | None,
        line_kwargs: dict = {},
        ribbon_kwargs: dict = {},
        theme_kwargs: dict = {},
        ) -> GGPlot:
    # handle None inputs
    if err_layer is not None:
        long_df['ymin'] = long_df['signal'] - long_df[err_layer]
        long_df['ymax'] = long_df['signal'] + long_df[err_layer]

    if label_col is None:
        label_col = 'trial_idx'
        show_legend = False
    else:
        show_legend = True

    # defaults

    line_kwargs = dict(
        show_legend=show_legend
    ) | line_kwargs
    ribbon_kwargs = dict(
        alpha=0.3,
        size=0,
        show_legend=show_legend,
    ) | ribbon_kwargs
    theme_kwargs = dict(
        panel_spacing=0.02,
    ) | theme_kwargs

    # construct plot
    p = (ggplot(long_df, aes(x='time', y='signal', color=label_col, group='trial_idx')))

    if group_col is not None:
        p = p + facet_wrap(facets=group_col)

    if err_layer is not None:
        p = p + geom_ribbon(
            aes(x='time', ymin='ymin', ymax='ymax', fill=label_col, group='trial_idx'),
            **ribbon_kwargs
        )

    p = (
        p
        + geom_line(**line_kwargs)
        + labs(
            x='Time (s)',
            y='Signal',
        )
        + get_default_theme()
        + theme(**theme_kwargs)
    )

    return p

#endregion

################################
#region --- SIMULATED LAYERS ---
################################
def _sim_clean_trace_labels() -> dict[str, str]:
    trace_order = [
        'experimental',
        'isosbestic'
    ]
    return {label : label.replace('_', ' ').capitalize() for label in trace_order}

def _sim_clean_layer_labels(condensed: bool) -> dict[str, str]:
    if condensed:
        layer_order = [
            'neural_trace',
            'photobleaching',
            'artifacts',
            'noise',
            'full_signal'
        ]

    else:
        layer_order = [
            'events',
            'dynamic_noise',
            'photobleaching',
            'movement_artifacts',
            'spike_artifacts',
            'jump_artifacts',
            'multiplicative_noise',
            'gaussian_noise',
            'full_signal'
        ]

    return {label : label.replace('_', ' ').capitalize() for label in layer_order}

def plot_simulated_layers(
        df: pd.DataFrame,
        condensed: bool = True,
        line_kwargs: dict = {},
        theme_kwargs: dict = {},
        ) -> GGPlot:
    # clean and order trace and layer columns
    trace_labels = _sim_clean_trace_labels()
    df['trace'] = pd.Categorical(
        df['trace'].map(trace_labels),
        list(trace_labels.values()),
        ordered=True,
    )

    layer_labels = _sim_clean_layer_labels(condensed)
    df['layer'] = pd.Categorical(
        df['layer'].map(layer_labels),
        list(layer_labels.values()),
        ordered=True,
    )

    # clean theme kwargs
    theme_kwargs = dict(
        strip_text_y=element_text(angle=0, ha="left"),
    ) | theme_kwargs

    p = (
        ggplot(df, aes(x='time', y='value'))
        + facet_grid(
            rows='layer',
            cols='trace',
            scales='free_y',
        )
        + geom_line(**line_kwargs)
        + labs(
            x = 'Time (s)',
            y = '',
        )
        + get_default_theme()
        + theme(**theme_kwargs)
    )

    return p

def plot_simulated_traces(
        df: pd.DataFrame,
        line_kwargs: dict = {},
        theme_kwargs: dict = {},
        ) -> GGPlot:
    # clean and order trace and layer columns
    trace_labels = _sim_clean_trace_labels()
    df['trace'] = pd.Categorical(
        df['trace'].map(trace_labels),
        list(trace_labels.values()),
        ordered=True,
    )

    # clean theme kwargs
    theme_kwargs = dict() | theme_kwargs

    p = (
        ggplot(df, aes(x='time', y='value'))
        + facet_wrap(
            facets='trace',
            nrow=2,
            scales='fixed'
        )
        + geom_line(**line_kwargs)
        + labs(
            x = 'Time (s)',
            y = 'Fluorescence intensity (a.u.)',
        )
        + get_default_theme()
        + theme(**theme_kwargs)
    )

    return p

#endregion

############################
#region --- CLUSTER TEST ---
############################

def plot_cluster_test(
        df: pd.DataFrame,

        title: str = '',
        y_lab: str = '',

        unsig_color: str = '#7a7a7a',
        cmap: str = 'viridis_r',
        n_cmap: int = 6,
        p_thr: float = 0.05,

        line_kwargs: dict = {},
        point_kwargs: dict = {},
        ribbon_kwargs: dict = {},
        theme_kwargs: dict = {},
        ) -> GGPlot:

    # handle pvalue color
    p_max = 1.0
    p_min = 1e-10

    t_min = np.log10(p_min)
    t_max = np.log10(p_max)
    t_thr = np.log10(p_thr)

    def to_pos(p):
        """Map p in [p_min, p_max] to [0, 1] on a log10 scale."""
        return (np.log10(p) - t_min)/(t_max - t_min)
    
    thr_pos = to_pos(p_thr)
    
    # sample cmap only over the significant region [p_min, p_thr]
    vir = plt.get_cmap(cmap)
    sig_positions = np.linspace(0, thr_pos, n_cmap)
    sig_colors = [mcolors.to_hex(vir(x / thr_pos)) for x in sig_positions]

    # Duplicate threshold location to create a hard jump to grey
    colors = sig_colors + [unsig_color, unsig_color]
    mm_positions = thr_pos * (sig_positions - sig_positions[0]) / (sig_positions[-1] - sig_positions[0])
    values = list(mm_positions) + [thr_pos, 1.0]
    
    # clean kwargs defaults
    line_kwargs = dict() | line_kwargs

    point_kwargs = dict(
        alpha=0.6,
        size=2.0,
        stroke=0,
        shape='o',
        color='black'
    ) | point_kwargs

    ribbon_kwargs = dict(
        alpha=0.3,
        size=0,
    ) | ribbon_kwargs

    theme_kwargs = dict() | theme_kwargs

    p = (
        ggplot(df, aes(x='time', y='value', color='source', group='source'))
        + geom_ribbon(
            aes(x='time', ymin='ymin', ymax='ymax', fill=after_scale('color')),
            **ribbon_kwargs
        )
        + geom_point(
            data=df.loc[~df['pvalue'].isna()],
            mapping=aes(x='time', y='value', fill='pvalue'),
            **point_kwargs
        )
        + geom_line(
            **line_kwargs
        )
        + scale_fill_gradientn(
            name='p-value',
            colors=colors,
            values=values,
            trans='log10', limits=(1e-10, 1.0),
            breaks=[1.0, 0.05, 1e-3, 1e-6, 1e-10],
            labels=['1.0', r'$\bf{0.05}$', '1e-3', '1e-6', '1e-10']
        )
        + labs(
            x = 'Time (s)',
            y = y_lab,
            title = title,
        )
        + get_default_theme()
        + theme(**theme_kwargs)
    )

    return p

#endregion

##########################
#region --- FMM RESULT ---
##########################

def plot_FMM_result(
        df: pd.DataFrame,
        line_kwargs: dict = {},
        hline_kwargs: dict = {},
        ribbon_inner_kwargs: dict = {},
        ribbon_outer_kwargs: dict = {},
        theme_kwargs: dict = {},
        ) -> GGPlot:
    
    # default params
    line_kwargs = dict(
        alpha=1.0,
        color='black',
        linetype='solid',
        size=0.5,
    ) | line_kwargs

    hline_kwargs = dict(
        alpha=1.0,
        color='black',
        linetype='dashed',
        size=0.3,
    ) | hline_kwargs

    ribbon_inner_kwargs = dict(
        size=0,
        alpha=0.6,
        fill='grey',
    ) | ribbon_inner_kwargs

    ribbon_outer_kwargs = dict(
        size=0,
        alpha=0.6,
        fill='darkgrey',
    ) | ribbon_outer_kwargs

    theme_kwargs = dict() | theme_kwargs

    # build plot
    p = (
        ggplot(df, aes(x='time', y='value'))
        + facet_wrap(
            'term'
        )
        + geom_hline(
            aes(yintercept=0),
            **hline_kwargs
        )
        + geom_ribbon(
            aes(x='time', ymin='lower_joint', ymax='upper_joint'),
            **ribbon_outer_kwargs
        )
        + geom_ribbon(
            aes(x='time', ymin='lower', ymax='upper'),
            **ribbon_inner_kwargs
        )
        + geom_line(
            aes(x='time', y='value'),
            **line_kwargs
        )
        + get_default_theme()
        + theme(
            **theme_kwargs
        )
    )

    return p

#endregion
