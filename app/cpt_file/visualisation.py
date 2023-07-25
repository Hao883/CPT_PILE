from math import floor
import math
import numpy as np
import pandas as pd
from munch import Munch, unmunchify
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from viktor.geo import GEFData, SoilLayout

from .soil_layout_conversion_functions import (
    Classification,
    convert_input_table_field_to_soil_layout,
    filter_nones_from_params_dict,
)

def visualise_cpt(cpt_params: Munch):
    # parse input file and user input
    classification = Classification(cpt_params.classification)
    cpt_params = unmunchify(cpt_params)
    parsed_cpt = GEFData(filter_nones_from_params_dict(cpt_params))
    qc = [q for q in parsed_cpt.qc]
    rf = [rf*100 for rf in parsed_cpt.Rf]
    depth = [el * 1e-3 for el in parsed_cpt.elevation]
    df = pd.DataFrame({'qc': qc, 'depth': depth,'rf':rf})
    # potential adjustment future
    Gs=2.65
    df['gamma'] = 9.81*(0.27*np.log(df['rf'])+0.36*(np.log(1e3*df['qc']/101.325)+1.236))

    df['depth_diff'] = abs(df['depth'].diff())
    df['depth_diff'].fillna(0, inplace=True)

    df['unit_sigma'] = df['gamma']*0.01
    df['sigma'] = df['unit_sigma'].cumsum()

    # df['Qt'] = df['qc']=

    soil_layout_original = SoilLayout.from_dict(cpt_params["soil_layout_original"])
    soil_layout_user = convert_input_table_field_to_soil_layout(
        bottom_of_soil_layout_user=cpt_params["bottom_of_soil_layout_user"],
        soil_layers_from_table_input=cpt_params["soil_layout"],
        soils=classification.soil_mapping,
    )

    # Create plotly figure
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        horizontal_spacing=0,
        column_widths=[1.5,1.5,3,2],
        subplot_titles=("Cone Resistance", "Friction ratio","SBT Index Ic", "Soil Layout")
    )

    # add left side of the figure: Qc and Rf plot
    fig.add_trace(  # Add the qc curve
        go.Scatter(
            name="Cone Resistance",
            x=parsed_cpt.qc,
            y=[el * 1e-3 for el in parsed_cpt.elevation],
            mode="lines",
            line=dict(color="mediumblue", width=1),
            legendgroup="Cone Resistance",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(  # Add the Rf curve
        go.Scatter(
            name="Friction ratio",
            x=[rfval * 100 if rfval else rfval for rfval in parsed_cpt.Rf],
            y=[el * 1e-3 if el else el for el in parsed_cpt.elevation],
            mode="lines",
            line=dict(color="red", width=1),
            legendgroup="Friction ratio",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(  # Add the Rf curve
        go.Scatter(
            name="Soil Behaviour Type Index Ic",
            x=df['unit_sigma'],
            y=[el * 1e-3 if el else el for el in parsed_cpt.elevation],
            mode="lines",
            line=dict(color="orange", width=1),
            legendgroup="Ic",
        ),
        row=1,
        col=3,
    )
    # add the bars on the right side of the plot
    add_soil_layout_to_fig(fig, soil_layout_original, soil_layout_user)

    # plot phreatic level
    fig.add_hline(
        y=cpt_params["ground_water_level"],
        line=dict(color="Blue", dash="dash", width=1),
        row="all",
        col="all",
    )

    update_fig_layout(fig, parsed_cpt)
    return fig

def visualise_pile(cpt_params: Munch, PILE_params: Munch):
    # parse input file and user input
    classification = Classification(cpt_params.classification)
    PILE_params = unmunchify(PILE_params)
    cpt_params = unmunchify(cpt_params)

    load = PILE_params["Load"]
    Diameter = PILE_params["Diameter"]
    pi = 3.1415926
    # Getting cpt data from files with cpt params
    parsed_cpt = GEFData(filter_nones_from_params_dict(cpt_params))
    soil_layout_original = SoilLayout.from_dict(cpt_params["soil_layout_original"])
    soil_layout_user = convert_input_table_field_to_soil_layout(
        bottom_of_soil_layout_user=cpt_params["bottom_of_soil_layout_user"],
        soil_layers_from_table_input=cpt_params["soil_layout"],
        soils=classification.soil_mapping,
    )

    qc = [q for q in parsed_cpt.qc]
    rf = [rf for rf in parsed_cpt.Rf]
    depth = [el * 1e-3 for el in parsed_cpt.elevation]
    df = pd.DataFrame({'qc': qc, 'depth': depth,'rf':rf})


    # fsol = fsl (1-eB · qc), then fs = a* fsol (kPa), but qc is MPa
    # need to update B for different soils
    B=-0.21
    f_sl = 138
    # a is installation factor, need to update a based on soil type
    a = 0.5
    # limit is Limiting values of ultimate shaft resistance based on soil type
    limit = 300
    # f_s is kPa
    f_s = [min(a * (f_sl * (1 - math.exp(B*q))) ,limit) for q in parsed_cpt.qc]
    # USF = unit Ultimate Shaft Resistance in MPa
    interval = depth[10]-depth[11]
    USF = [s*interval*pi*Diameter for s in f_s]
    sum_USF = [sum(USF[:i+1]) for i in range(len(USF))]
    #     df['sum_USF'] is the shaft capacity in kN
    df['sum_USF'] = sum_USF
    # fb = kc · qca, kc = bearing factor, depending on soil type and pile class; qca = equivalent average cone resistance at the base.
    k_c = 0.3

    # get the mean qc in 3D range of the depth
    n = int(3*Diameter / interval)

    # First step: calculate rolling mean of qc
    df['rolling_mean_qc'] = df['qc'].rolling(window=n ,min_periods=1).mean()

    # Second step: eliminate values outside 0.7 - 1.3 range of rolling mean
    df['qc_filtered'] = df.apply(
        lambda row: row['qc'] if 0.7 * row['rolling_mean_qc'] <= row['qc'] <= 1.3 * row['rolling_mean_qc'] else np.nan,
        axis=1)

    # Third step: calculate the mean of the new qc_filtered column
    df['filtered_rolling_mean_qc'] = k_c * df['qc_filtered'].rolling(window=n,min_periods=1).mean()
    # Base is in kN
    df['Base'] = df['filtered_rolling_mean_qc']*1000*0.25*pi*Diameter*Diameter
    # overall axial capacity in kN

    df['Overall'] = df['sum_USF'] + df['Base']

    df['intersect'] = np.isclose(df['Overall'], load, atol=0.1)

    # Create plotly figure
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.1,
        column_widths=[0.4, 0.4, 0.4],
        # subplot_titles=("Unit Ultimate Shaft Resistance", "Unit Ultimate Base Resistance", "Overall Pile Capacity"),
    )

    fig.add_trace(  # Add the shaft
        go.Scatter(
            name="Unit Ultimate Shaft Resistance",
            x=f_s,
            y=[el * 1e-3 for el in parsed_cpt.elevation],
            mode="lines",
            line=dict(color="mediumblue", width=1),
            legendgroup="Unit Shaft Resistance",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(  # Add base
        go.Scatter(
            name="Rolling Ultimate Base Resistance",
            x=df['filtered_rolling_mean_qc'],
            y=[el * 1e-3 for el in parsed_cpt.elevation],
            mode="lines",
            line=dict(color="red", width=1),
            legendgroup="Rolling Base Resistance",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(  # Add base
        go.Scatter(
            name="Overall pile capacity",
            x=df['Overall'],
            y=[el * 1e-3 for el in parsed_cpt.elevation],
            mode="lines",
            line=dict(color="red", width=2),
            legendgroup="Overall pile capacity",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(  # Add base
        go.Scatter(
            name="Base resistance capacity",
            x=df['Base'],
            y=[el * 1e-3 for el in parsed_cpt.elevation],
            mode="lines",
            line=dict(color="orange", width=1, dash='longdash'),
            legendgroup="Base capacity",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(  # Add base
        go.Scatter(
            name="Shaft resistance capacity",
            x=df['sum_USF'],
            y=[el * 1e-3 for el in parsed_cpt.elevation],
            mode="lines",
            line=dict(color="blue", width=1, dash='dashdot'),
            legendgroup="Shaft capacity",
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            name="Reaction Load",
            x=load * np.ones(100),
            y=np.linspace(min(el * 1e-3 for el in parsed_cpt.elevation), 0, 100),
            mode="lines",
            line=dict(color="purple", width=1,dash='dash'),
            legendgroup="Reaction Load",
        ),
        row=1,
        col=3,
    )
    nearest_index = (df['Overall'] - load).abs().idxmin()
    pile_tip = df.loc[nearest_index, 'depth']

    fig.add_trace(
        go.Scatter(
            name="Required least Pile Tip Level",
            x=np.linspace(0, max(df['Overall']), 100),
            y=pile_tip * np.ones(100),
            mode="lines",
            line=dict(color="black", width=2, dash='dot'),
            legendgroup="Pile Tip",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            name="Intersection",
            x=[load],
            y=[pile_tip],
            mode="markers",
            # line=dict(color="black", width=2, dash='dot'),
            legendgroup="Pile Tip",
        ),
        row=1,
        col=3,
    )
    update_pile_fig_layout(fig, parsed_cpt)
    return fig

def update_pile_fig_layout(fig, parsed_cpt):
    """Updates layout of the figure and formats the grids"""
    fig.update_layout(barmode="stack", template="plotly_white", legend=dict(x=1.15, y=0.5))
    fig.update_annotations(font_size=12)
    # Format axes and grids per subplot
    standard_grid_options = dict(showgrid=True, gridwidth=1, gridcolor="LightGrey")
    standard_line_options = dict(showline=True, linewidth=2, linecolor="LightGrey")

    # update x-axis for Qc
    fig.update_xaxes(
        row=1,
        col=1,
        **standard_line_options,
        **standard_grid_options,
        # range=[0, 30],
        # tick0=0,
        # dtick=5,
        title_text="Unit Shaft Resistance [kPa]",
        title_font=dict(color="mediumblue"),
    )
    # update x-axis for Rf
    fig.update_xaxes(
        row=1,
        col=2,
        **standard_line_options,
        **standard_grid_options,
        # range=[9.9, 0],
        # tick0=0,
        # dtick=5,
        title_text="Rolling Base Resistance [MPa]",
        title_font=dict(color="red"),
    )
    fig.update_xaxes(
        row=1,
        col=3,
        **standard_line_options,
        **standard_grid_options,
        # range=[0, 50],
        # tick0=0,
        # dtick=5,
        title_text="Overall pile capactiy [kN]",
        title_font=dict(color="black"),
    )

    # update all y axis to ensure they line up
    fig.update_yaxes(
        row=1,
        col=1,
        **standard_grid_options,
        title_text="Depth [m]",
        tick0=floor(parsed_cpt.elevation[-1] / 1e3) - 5,
        dtick=5,
    )  # for Qc

    fig.update_yaxes(
        row=1,
        col=2,
        **standard_line_options,
        **standard_grid_options,  # for Rf
        tick0=floor(parsed_cpt.elevation[-1] / 1e3) - 5,
        dtick=5,
    )

    fig.update_yaxes(
        row=1,
        col=4,
        **standard_line_options,  # for soil layouts
        tick0=floor(parsed_cpt.elevation[-1] / 1e3) - 5,
        dtick=1,
        showticklabels=True,
        side="right",
    )


def update_fig_layout(fig, parsed_cpt):
    """Updates layout of the figure and formats the grids"""
    fig.update_layout(barmode="stack", template="plotly_white", legend=dict(x=1.15, y=0.5))
    fig.update_annotations(font_size=12)
    # Format axes and grids per subplot
    standard_grid_options = dict(showgrid=True, gridwidth=1, gridcolor="LightGrey")
    standard_line_options = dict(showline=True, linewidth=2, linecolor="LightGrey")

    # update x-axis for Qc
    fig.update_xaxes(
        row=1,
        col=1,
        **standard_line_options,
        **standard_grid_options,
        range=[0, 30],
        tick0=0,
        dtick=5,
        title_text="qc [MPa]",
        title_font=dict(color="mediumblue"),
    )
    # update x-axis for Rf
    fig.update_xaxes(
        row=1,
        col=2,
        **standard_line_options,
        **standard_grid_options,
        range=[0, 9.9],
        tick0=0,
        dtick=5,
        title_text="Rf [%]",
        title_font=dict(color="red"),
    )
    # update x-axis for Ic
    fig.update_xaxes(
        row=1,
        col=3,
        **standard_line_options,
        **standard_grid_options,
        # range=[0, 500],
        # tick0=0,
        # dtick=0.1,
        title_font=dict(color="red"),
    )

    # update all y axis to ensure they line up
    fig.update_yaxes(
        row=1,
        col=1,
        **standard_grid_options,
        title_text="Depth (m)",
        tick0=floor(parsed_cpt.elevation[-1] / 1e3) - 5,
        dtick=1,
    )  # for Qc

    fig.update_yaxes(
        row=1,
        col=2,
        **standard_line_options,
        **standard_grid_options,  # for Rf
        tick0=floor(parsed_cpt.elevation[-1] / 1e3) - 5,
        dtick=1,
    )

    fig.update_yaxes(
        row=1,
        col=4,
        **standard_line_options,  # for soil layouts
        tick0=floor(parsed_cpt.elevation[-1] / 1e3) - 5,
        dtick=1,
        showticklabels=True,
        side="right",
    )


def add_soil_layout_to_fig(fig, soil_layout_original, soil_layout_user):
    """Add bars for each soil type separately in order to be able to set legend labels"""
    unique_soil_types = {
        layer.soil.properties.ui_name for layer in [*soil_layout_original.layers, *soil_layout_user.layers]
    }
    for ui_name in unique_soil_types:
        original_layers = [layer for layer in soil_layout_original.layers if layer.soil.properties.ui_name == ui_name]
        interpreted_layers = [layer for layer in soil_layout_user.layers if layer.soil.properties.ui_name == ui_name]
        soil_type_layers = [
            *original_layers,
            *interpreted_layers,
        ]  # have a list of all soils used in both figures

        # add the bar plots to the figures
        fig.add_trace(
            go.Bar(
                name=ui_name,
                x=["Original"] * len(original_layers) + ["Interpreted"] * len(interpreted_layers),
                y=[-layer.thickness * 1e-3 for layer in soil_type_layers],
                width=0.5,
                marker_color=[f"rgb{layer.soil.color.rgb}" for layer in soil_type_layers],
                hovertext=[
                    f"Soil Type: {layer.soil.properties.ui_name}<br>"
                    f"Top of layer: {layer.top_of_layer * 1e-3:.2f}<br>"
                    f"Bottom of layer: {layer.bottom_of_layer * 1e-3:.2f}"
                    for layer in soil_type_layers
                ],
                hoverinfo="text",
                base=[layer.top_of_layer * 1e-3 for layer in soil_type_layers],
            ),
            row=1,
            col=4,
        )
