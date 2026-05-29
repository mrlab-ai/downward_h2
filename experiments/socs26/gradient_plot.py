import itertools
import logging
import math
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

from downward.reports import PlanningReport
from lab import tools

class GradientWriter:
    @classmethod
    def write(cls, report, filename):
        save_format = report.output_format
        if save_format == "tex":
            save_format = "pgf"
        
        fig, ax = plt.subplots(figsize=report.matplotlib_options.get("figure.figsize", [8, 8]))
        
        # 1. Collect all ratios
        ratios = []
        for coords in report.categories.values():
            ratios.extend([c[2] for c in coords if c[2] is not None])

        # 2. Determine symmetric dynamic bounds with "Sensitivity"
        if ratios:
            # Filter out perfect 1.0s to find the actual deviation
            deviations = [r for r in ratios if not math.isclose(r, 1.0, rel_tol=1e-5)]
            if deviations:
                actual_min = min(deviations)
                actual_max = max(deviations)
                
                # Calculate max log distance
                log_max_dist = max(abs(math.log10(actual_max)), abs(math.log10(actual_min)))
                
                # If the spread is tiny (e.g. all within 10%), force a minimum spread 
                # so the colors aren't all grey.
                log_max_dist = max(log_max_dist, 0.1) 
                
                vmax = 10**log_max_dist
                vmin = 1.0 / vmax
            else:
                vmin, vmax = 0.8, 1.2
        else:
            vmin, vmax = 0.1, 10.0

        # Using LogNorm ensures the color scale is logarithmic
        norm = LogNorm(vmin=vmin, vmax=vmax)

        sc = None
        for category, coords in report.categories.items():
            if not coords: continue
            
            x_vals = [c[0] for c in coords]
            y_vals = [c[1] for c in coords]
            r_vals = [max(vmin, min(vmax, c[2])) for c in coords]
            
            sc = ax.scatter(
                x_vals, y_vals,
                c=r_vals,
                cmap=report.cmap, # RdYlBu_r provides high contrast
                marker='o',
                norm=norm,
                edgecolors='black',
                linewidths=0.2,
                alpha=0.9,
                s=40,
                zorder=3
            )

        if sc:
            cbar = plt.colorbar(sc, ax=ax, extend='both')
            cbar.set_label(f"Ratio of {report.gradient_attribute} (Alg2 / Alg1)")
            
            # Create a spread of ticks between min and max
            # We use 1.0, the bounds, and two intermediate points
            ticks = [vmin, 1.0, vmax]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{vmin:.2f}x", "1x", f"{vmax:.2f}x"])

        # Axis and Scale Logic
        ax.set_xscale(report.xscale)
        ax.set_yscale(report.yscale)
        ax.set_xlabel(report.xlabel)
        ax.set_ylabel(report.ylabel)
        ax.set_title(report.title)
        
        # Reference diagonal
        all_coords = [v for coords in report.categories.values() for c in coords for v in c[:2] if v is not None]
        if all_coords:
            limit = max(all_coords) * 1.05
            ax.plot([0, limit], [0, limit], color='black', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)

        plt.savefig(filename, format=save_format, bbox_inches='tight', dpi=200)
        plt.close(fig)

class ScatterPlotReport(PlanningReport):
    def __init__(self, relative=False, show_missing=True, 
                 gradient_attribute="expansions_until_last_jump", cmap="RdYlBu_r",
                 title=None, scale=None, xlabel="", ylabel="", matplotlib_options=None, **kwargs):
        
        kwargs.pop("get_category", None)
        kwargs.setdefault("format", "png")
        PlanningReport.__init__(self, **kwargs)
        
        self.relative = relative
        self.attribute = self.attributes[0]
        self.gradient_attribute = gradient_attribute
        self.cmap = cmap
        self.show_missing = show_missing
        self.writer = GradientWriter
        
        self.title = title if title is not None else f"Time: {self.attribute}"
        self.xscale = scale or "linear"
        self.yscale = "log" if self.relative else self.xscale
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.matplotlib_options = matplotlib_options or {"figure.figsize": [8, 8]}

    def has_multiple_categories(self):
        return False

    def _fill_categories(self):
        categories = defaultdict(list)
        for runs in self.problem_runs.values():
            if len(runs) != 2: continue
            run1, run2 = runs
            
            x, y = run1.get(self.attribute), run2.get(self.attribute)
            
            # Clip expansions to 1
            g1_raw = run1.get(self.gradient_attribute)
            g2_raw = run2.get(self.gradient_attribute)
            g1 = max(1, g1_raw) if g1_raw is not None else None
            g2 = max(1, g2_raw) if g2_raw is not None else None
            
            ratio = (g2 / float(g1)) if (g1 and g2) else 1.0

            if self.show_missing or (x is not None and y is not None):
                # Using 0 as we are in linear scale now
                x_val = x if x is not None else 0
                y_val = y if y is not None else 0
                categories[None].append((x_val, y_val, ratio))
        return categories

    def write(self):
        suffix = "." + self.output_format
        if not self.outfile.endswith(suffix):
            self.outfile += suffix
        tools.makedirs(os.path.dirname(self.outfile))
        
        self.categories = self._fill_categories()
        self.plot_diagonal_line = not self.relative
        self.xlabel = self.xlabel or self.algorithms[0]
        self.ylabel = self.ylabel or self.algorithms[1]
        
        self.writer.write(self, self.outfile)