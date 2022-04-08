#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:06:20 2021

@author: marlinberger

Load saved figures to adjust them manually or in the code. Every plot gets not
only saved as png but also as pickled-file to rework it here, wihhout needing
to load complete datasets again and redo the predictions (as this might be
very time consuming)
"""
# python packages
import matplotlib.pyplot as plt

# own packages
from motion_tracking_helpers import motion_tracking_helpers as mt_hlp
from motion_tracking_helpers import mt_neural_helpers as neu_hlp
from motion_tracking_helpers import custom_layers as cus_lay

# USER INPUTS
# ----------------
# the name of the folder in 99_Architectures/_saves, in which the desired model
# lies
plot_name = "overview_metrics__Dev_Pack_small_val__2022-02-09--14-21-00--" \
            "glamorous-sweep-178"
# ----------------

def rework(plot_name):
    """rework a previously saved figure

    Args:
        plot_name (str): the name of the plot in "06_Analyze/_plot-results" to
                         load and rework. It's the name without file ending
    """
    # create the path that leads to the "_plot-results" dir
    save_path = (
            mt_hlp.get_abs_script_path(__file__) /
            mt_hlp.name_constants("ANALYZE_DIR_NAME") /
            mt_hlp.name_constants("PLOT_SAVES")
        )
    
    # load the figure
    f = mt_hlp.load_pickeled(
        plot_name, save_path
    )

    # get axes, possibly play with them
    ax_list = f.axes
    ax1 = ax_list[0]

    # set layout details, possibly play around here
    f.set_dpi(200)
    f.tight_layout()

    # display the plot
    plt.show()




if __name__ == "__main__":
    rework(plot_name)
