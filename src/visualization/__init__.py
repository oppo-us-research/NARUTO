"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def init_visualizer(main_cfg, info_printer):
    """initialize Visualizer

    Args:
        main_cfg (mmengine.Config): Configuration
        info_printer (InfoPrinter): information printer

    Returns:
        visualizer (Visualizer): visualizer module

    """
    ##################################################
    ### HabitatSim
    ##################################################
    if main_cfg.visualizer.method == "naruto":
        info_printer("Initialize Visualizer...", 0, "NARUTO")
        from src.visualization.naruto_visualizer import NARUTOVisualizer
        visualizer = NARUTOVisualizer(
            main_cfg,
            info_printer
            ) 
    else:
        assert False, f"Visualizer choices: [naruto]. Current option: [{main_cfg.visualizer}]"
    return visualizer