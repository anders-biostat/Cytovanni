"""Arcsinh axis scale for cytometry data: ``ax.set_xscale("arcsinh", cofactor=...)``.

The scale shows ``arcsinh(x / cofactor)``: linear around 0, logarithmic far from it. Major ticks are 0 and +-10^p,
labelled in scientific notation like a log axis; the powers close to 0, which crowd together on this scale, are left
out as far as needed for the labels to fit. Minor ticks are at 2..9 x 10^p, and continue to the unlabelled powers
below the labels as far as they stay apart. If that leaves fewer than two labels (no power of ten in the view, or
one without the 0), labels at m x 10^p are added (2 and 5 preferred, e.g. 2x10^3 and 5x10^3). All of this depends on
the size of the axis and the font of the tick labels and is recomputed at every draw. On a y axis the labels are
aligned by their digits: the 0 stands under the 0 of 10^p, the exponents stick out towards the axis.

Two ways to use it:

- data in its own units: the scale, ``ax.set_xscale("arcsinh", cofactor=...)`` (registered on ``import cytovanni``);
- data already transformed to ``arcsinh(x / cofactor)`` (datashader images, histograms of transformed data): only the
  ticks, `set_arcsinh_ticks`.
"""
import itertools

import numpy as np
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.figure import Figure

__all__ = ["ArcSinhScale", "ArcSinhLocator", "ArcSinhFormatter", "set_arcsinh_ticks"]


class ArcSinhTransform(mtransforms.Transform):
    """``arcsinh(x / cofactor)``."""
    input_dims = output_dims = 1

    def __init__(self, cofactor):
        super().__init__()
        self.cofactor = cofactor

    def transform_non_affine(self, values):
        return np.arcsinh(np.asarray(values) / self.cofactor)

    def inverted(self):
        return InvertedArcSinhTransform(self.cofactor)


class InvertedArcSinhTransform(mtransforms.Transform):
    """``sinh(y) * cofactor``."""
    input_dims = output_dims = 1

    def __init__(self, cofactor):
        super().__init__()
        self.cofactor = cofactor

    def transform_non_affine(self, values):
        return np.sinh(np.asarray(values)) * self.cofactor

    def inverted(self):
        return ArcSinhTransform(self.cofactor)


def _mantissa_exponent(value):
    """``(m, p)`` with ``|value| = m 10^p``, m an integer 1..9, or None if `value` is not of that form."""
    p = int(np.floor(np.log10(abs(value)) + 1e-9))
    m = abs(value) / 10.**p
    if abs(m - np.round(m)) > 1e-6 * m or not 1 <= np.round(m) <= 9:
        return None
    return int(np.round(m)), p


def format_label(value):
    """Tick label: ``0``, ``+-10^p`` or ``+-m x 10^p`` (mathtext in the tick label font, like a log axis); "" for
    other values."""
    if value == 0:
        return r"$\mathdefault{0}$"
    mp = _mantissa_exponent(value)
    if mp is None:
        return ""
    m, p = mp
    sign = "-" if value < 0 else ""
    if m == 1:
        return r"$\mathdefault{%s10^{%d}}$" % (sign, p)
    return r"$\mathdefault{%s%d\times10^{%d}}$" % (sign, m, p)


def _base_label(label):
    """`label` without its exponent (``-10`` of ``-10^{3}``); labels without one are returned unchanged."""
    return label.split("^{")[0] + "}$" if "^{" in label else label


def _label_box(label, fontprops, figure):
    """Width, height and ascent (height above the baseline) of the box matplotlib lays a one-line label out in, in
    points. Measured with matplotlib's own text layout, which makes the box at least as high as ordinary text in the
    font (the exact rule differs between matplotlib versions)."""
    renderer = figure._get_renderer()
    text = mtext.Text(0, 0, label, fontproperties=fontprops, va="baseline")
    text.set_figure(figure)
    box = text.get_window_extent(renderer)
    scale = renderer.points_to_pixels(1.)
    return box.width / scale, box.height / scale, box.y1 / scale


class ArcSinhLocator(mticker.Locator):
    """Major or minor ticks of an arcsinh axis (see the module description).

    Parameters
    ----------
    cofactor : float
        Cofactor of the arcsinh transform, in the units of the data.
    minor : bool, default False
        Locate the minor ticks instead of the major ones.
    transformed : bool, default False
        The axis coordinates are already ``arcsinh(x / cofactor)`` (`set_arcsinh_ticks`), not the data itself.
    minpower : int, optional
        Fixed smallest labelled power: label +-10^p for all ``|p| >= minpower``, regardless of the space. By default,
        the smallest power for which the labels do not overlap, decided per side of 0.
    label_gap : float, default .5
        Smallest free space between two major tick labels, in units of the label font size.
    zero_gap : float, optional
        The same next to the 0 label, where the powers crowd together: default 1 on an x axis, 2 on a y axis.
    minor_spacing : float, default 2.5
        Smallest distance between two minor ticks, in points.
    """

    def __init__(self, cofactor, minor=False, transformed=False, minpower=None, label_gap=.5, zero_gap=None, minor_spacing=2.5):
        self.cofactor = cofactor
        self.minor = minor
        self.transformed = transformed
        self.minpower = minpower
        self.label_gap = label_gap
        self.zero_gap = zero_gap
        self.minor_spacing = minor_spacing

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    # --- geometry of the axis ------------------------------------------------------------------------------------
    def _onx(self):
        return self.axis is None or self.axis.axis_name == "x"

    def _length_points(self):
        """Length of the axis in points (a default if it is not attached to an axes)."""
        if self.axis is None or self.axis.axes is None:
            return 300.
        ax = self.axis.axes
        bbox = ax.get_position(original=False).transformed(ax.figure.transFigure)
        length = bbox.width if self._onx() else bbox.height
        return length * 72. / ax.figure.dpi

    def _label_font(self):
        """Font of the tick labels and the figure to measure them in (a stand-in if not attached to an axes)."""
        if self.axis is None or self.axis.axes is None:
            from matplotlib.font_manager import FontProperties
            return FontProperties(), Figure()
        return self.axis.get_major_ticks(numticks=1)[0].label1.get_fontproperties(), self.axis.axes.figure

    # --- ticks ---------------------------------------------------------------------------------------------------
    def tick_values(self, vmin, vmax):
        if self.transformed:
            vmin, vmax = np.sinh(vmin) * self.cofactor, np.sinh(vmax) * self.cofactor
        vmin, vmax = min(vmin, vmax), max(vmin, vmax)
        t = lambda x: np.arcsinh(np.asarray(x, dtype=float) / self.cofactor)
        span = t(vmax) - t(vmin)
        if not np.isfinite(span) or span <= 0:
            return np.zeros(0)
        scale = self._length_points() / span                  # points per arcsinh unit
        pos = lambda x: t(x) * scale

        fontprops, figure = self._label_font()
        fontsize = fontprops.get_size_in_points()
        extents = {}

        def extent(value):
            """Width (x axis) or height (y axis) of the label of `value`, in points."""
            if value not in extents:
                w, h, _ = _label_box(format_label(value), fontprops, figure)
                extents[value] = w if self._onx() else h
            return extents[value]

        pmax = int(np.ceil(np.log10(max(abs(vmin), abs(vmax), 1e-300))))
        plow = int(np.floor(np.log10(self.cofactor))) - 4     # below: 10^p is at most 1e-4 cofactors from 0
        plow = min(plow, pmax - 1)

        zero_gap = self.zero_gap if self.zero_gap is not None else (1. if self._onx() else 2.)
        need = lambda a, b: (extent(a) + extent(b)) / 2 \
            + (zero_gap if a == 0 or b == 0 else self.label_gap) * fontsize
        tol = 1e-9 * max(abs(vmin), abs(vmax))               # ticks on the limits count as inside
        major, minor = [0.] if vmin - tol <= 0 <= vmax + tol else [], []

        for sign in [s for s, present in [(1, vmax > 0), (-1, vmin < 0)] if present]:
            lo, hi = (max(vmin, 0), vmax) if sign > 0 else (-min(vmax, 0), -vmin)  # |x| range on this side
            lo, hi = lo - tol, hi + tol
            stride = self._stride(sign, hi, pos, need)
            first = self._first_power(sign, lo, hi, plow, pmax, stride, scale, pos, need, vmin <= 0 <= vmax)
            labelled = [p for p in range(first, pmax + 1) if p % stride == 0]
            major += [sign * 10.**p for p in labelled if lo <= 10.**p <= hi]
            for p in range(plow, pmax + 1):
                subs = np.arange(1, 10) * 10.**p
                if np.diff(pos(np.arange(1, 11) * 10.**p)).min() < self.minor_spacing:
                    # the 2..9 subs do not fit: keep only an unlabelled 10^p, if the decades are apart enough
                    if p in labelled or pos(10.**(p + 1)) - pos(10.**p) < self.minor_spacing:
                        continue
                    subs = subs[:1]
                elif p in labelled:
                    subs = subs[1:]
                minor += [sign * v for v in subs if lo <= v <= hi]

        if len(major) < 2:
            major += self._extra_labels(major, vmin - tol, vmax + tol, plow, pmax, pos, need)
        ticks = np.asarray(sorted(minor if self.minor else major))
        return t(ticks) if self.transformed else ticks

    # mantissas by how good they look as the only labels of an axis
    _MANTISSA_RANK = {2: 0, 5: 0, 3: 1, 4: 2, 6: 2, 7: 2, 8: 2, 9: 2}

    def _extra_labels(self, major, vmin, vmax, plow, pmax, pos, need):
        """Labels at m x 10^p (m = 2..9) when 0 and the powers of ten give fewer than two labels in the view.

        Adds as many as are missing to have two labels: of the combinations whose labels do not overlap, the one with
        the most round mantissas (2 and 5, then 3, then the others), then with the labels furthest apart.
        """
        candidates = [sign * m * 10.**p for p in range(plow, pmax + 1) for m in range(2, 10) for sign in (1, -1)]
        candidates = [v for v in candidates if vmin <= v <= vmax]
        missing = 2 - len(major)
        best = None
        for k in range(missing, 0, -1):
            for combo in itertools.combinations(candidates, k):
                labels = sorted(major + list(combo))
                gaps = [pos(b) - pos(a) - need(a, b) for a, b in zip(labels[:-1], labels[1:])]
                if gaps and min(gaps) < 0:
                    continue
                rank = max(self._MANTISSA_RANK[_mantissa_exponent(v)[0]] for v in combo)
                key = (rank, -min(gaps, default=0.))
                if best is None or key < best[0]:
                    best = (key, list(combo))
            if best is not None:
                return best[1]
        return []

    def _stride(self, sign, hi, pos, need):
        """Decades between labels on one side of 0: 1, unless the outermost decades are too short for their labels."""
        p = int(np.floor(np.log10(hi)))                         # outermost power within the view on this side
        for stride in range(1, 50):
            if pos(10.**p) - pos(10.**(p - stride)) >= need(sign * 10.**p, sign * 10.**(p - stride)):
                return stride
        return 50

    def _first_power(self, sign, lo, hi, plow, pmax, stride, scale, pos, need, with_zero):
        """Smallest labelled power on one side of 0: all neighbouring labels (and the 0) are far enough apart."""
        if self.minpower is not None:
            return self.minpower
        candidates = [p for p in range(plow, pmax + 1) if p % stride == 0]
        for first in candidates:
            ticks = ([0.] if with_zero else []) + [10.**p for p in candidates if p >= first and lo <= 10.**p <= hi]
            if all(pos(b) - pos(a) >= need(sign * a, sign * b) for a, b in zip(ticks[:-1], ticks[1:])):
                return first
        return pmax + 1


class ArcSinhFormatter(mticker.Formatter):
    """Labels of `ArcSinhLocator` ticks: ``0``, ``+-10^p`` and ``+-m x 10^p`` (mathtext, tick label font).

    Parameters
    ----------
    cofactor : float
        Cofactor of the arcsinh transform, in the units of the data.
    transformed : bool, default False
        The axis coordinates are already ``arcsinh(x / cofactor)``.
    """

    def __init__(self, cofactor, transformed=False):
        self.cofactor = cofactor
        self.transformed = transformed

    def __call__(self, x, pos=None):
        value = np.sinh(x) * self.cofactor if self.transformed else x
        if abs(value) < 1e-9 * self.cofactor:
            return format_label(0)
        return format_label(value)

    def format_data(self, x):
        value = np.sinh(x) * self.cofactor if self.transformed else x
        return "%.4g" % value

    def format_ticks(self, values):
        labels = super().format_ticks(values)
        if self.axis is not None and len(values):
            self._align_labels(labels)
        return labels

    def _align_labels(self, labels):
        """Put the baseline of the 0 label on that of the power labels and, on a y axis, the labels under each other
        by their digits.

        Vertically: tick labels are aligned by their top (x axis) or the middle of their height above the baseline
        (y axis), and the superscript makes the power labels taller than the 0: the 0 is moved down by the difference
        (x) or half of it (y).

        Horizontally (y axis): the labels are aligned at the end of the part before the exponent, so the 0 stands
        under the 0 of 10^p and the exponents stick out towards the axis. Right-aligned labels (left axis) keep their
        right edge where the label with the widest exponent ends; left-aligned labels (right axis) start where the
        label with the widest part before the exponent starts.

        The tick objects are reused for other positions, so all of them are reset at every draw.
        """
        ticks = self.axis.get_major_ticks(len(labels))
        fontprops = ticks[0].label1.get_fontproperties()
        figure = self.axis.axes.figure
        powers = [lb for lb in labels if lb and lb != format_label(0)]
        shift = 0.
        if powers and format_label(0) in labels:
            shift = max(_label_box(lb, fontprops, figure)[2] for lb in set(powers)) \
                - _label_box(format_label(0), fontprops, figure)[2]
        onx = self.axis.axis_name == "x"
        # width of the exponent of each label and of the part before it, in points
        tail, head = {}, {}
        if not onx:
            for lb in set(lb for lb in labels if lb):
                head[lb] = _label_box(_base_label(lb), fontprops, figure)[0]
                tail[lb] = _label_box(lb, fontprops, figure)[0] - head[lb]
        dpi_scale = figure.dpi_scale_trans
        for tick, label in zip(ticks, labels):
            down = (shift if onx else shift / 2) if label == format_label(0) else 0.
            # label2 of the x axis (top) is aligned at its bottom: no shift needed there
            for text, transform, d in [(tick.label1, tick._get_text1_transform()[0], down),
                                       (tick.label2, tick._get_text2_transform()[0], 0. if onx else down)]:
                right = 0.
                if label in tail:
                    ha = text.get_horizontalalignment()
                    if ha == "right":
                        right = tail[label] - max(tail.values())
                    elif ha == "left":
                        right = max(head.values()) - head[label]
                text.set_transform(transform + mtransforms.ScaledTranslation(right / 72., -d / 72., dpi_scale))


class ArcSinhScale(mscale.ScaleBase):
    """Matplotlib scale ``"arcsinh"``: ``ax.set_xscale("arcsinh", cofactor=5000)``.

    Shows ``arcsinh(x / cofactor)``; the data is plotted in its own units. Ticks and labels: see the module
    description. The tick options (`minpower`, `label_gap`, `zero_gap`, `minor_spacing`) are those of `ArcSinhLocator`.

    Parameters
    ----------
    cofactor : float
        Cofactor of the arcsinh transform, in the units of the data.
    """
    name = "arcsinh"

    def __init__(self, *axis, cofactor, minpower=None, label_gap=.5, zero_gap=None, minor_spacing=2.5):
        super().__init__(*axis)                               # matplotlib < 3.11 passes the axis
        self.cofactor = cofactor
        self._locator_kw = dict(minpower=minpower, label_gap=label_gap, zero_gap=zero_gap,
                                minor_spacing=minor_spacing)

    def get_transform(self):
        return ArcSinhTransform(self.cofactor)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ArcSinhLocator(self.cofactor, **self._locator_kw))
        axis.set_minor_locator(ArcSinhLocator(self.cofactor, minor=True, **self._locator_kw))
        axis.set_major_formatter(ArcSinhFormatter(self.cofactor))
        axis.set_minor_formatter(mticker.NullFormatter())


mscale.register_scale(ArcSinhScale)


def set_arcsinh_ticks(ax, cofactor, axis="x", minpower=None, mark_cofactor=False, **kw):
    """Arcsinh ticks on an axis that shows already transformed data, ``arcsinh(x / cofactor)``.

    For data that is already transformed (datashader images, histograms of ``arcsinh(x / cofactor)``): only the
    ticks and labels change, the axis stays linear in the transformed values. For data in its own units, use the
    scale instead (``ax.set_xscale("arcsinh", cofactor=...)``). Ticks and labels are those of the scale (module
    description); they follow the axis limits and size at every draw, so this can be called before or after plotting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to label; the data on the axis must be ``arcsinh(x / cofactor)``.
    cofactor : float
        Cofactor of the arcsinh transform of the plotted data, in the units of the linear data.
    axis : {'x', 'y'}, default 'x'
        Axis to label.
    minpower : int, optional
        Fixed smallest labelled power (``|p| >= minpower``); by default chosen so that the labels fit.
    mark_cofactor : bool, default False
        Also draw short marks at the edge of the axis at +-`cofactor`.
    **kw
        Further tick options of `ArcSinhLocator` (`label_gap`, `zero_gap`, `minor_spacing`).
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(ArcSinhLocator(cofactor, transformed=True, minpower=minpower, **kw))
    target.set_minor_locator(ArcSinhLocator(cofactor, minor=True, transformed=True, minpower=minpower, **kw))
    target.set_major_formatter(ArcSinhFormatter(cofactor, transformed=True))
    target.set_minor_formatter(mticker.NullFormatter())
    if mark_cofactor:
        line = ax.axvline if axis == "x" else ax.axhline
        for v in [np.arcsinh(1), np.arcsinh(-1)]:
            line(v, 0, .05, linewidth=1, color="black")
