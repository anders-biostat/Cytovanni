"""BD FACSDiscover S8: spillover matrix, gating strategy from the file metadata, and image-based cell size/shape."""
import flowio
import flowutils
import flowkit as fk
import pandas as pd
import numpy as np
import json

from .utils import readfcs_metadata


def read_spill_S8(samplepath, safe=True):
    """Read the spectra (spillover matrix) of the BD spectral unmixing from an .fcs file of a FACSDiscover S8.

    Parameters
    ----------
    samplepath : str
        Path to the .fcs file.
    safe : bool, default True
        Check that the file comes from an S8 ($CYT is "FACSDiscover S8").

    Returns
    -------
    pandas.DataFrame
        Spectra of shape (channel, dye): index the detector channels of $SPILL, columns the probes listed in the
        'BDSPECTRAL PROBES' keyword (with the suffix "-A (SW-unmix)" removed). Column j is row j of the $SPILL
        matrix.

    Raises
    ------
    ValueError
        If `safe` and the file does not come from an S8.
    """
    metadata = flowio.FlowData(samplepath, only_text=True).text
    if safe:
        if not "cyt" in metadata or metadata["cyt"]!="FACSDiscover S8":
            raise ValueError("Are you sure the data came from an S8? Disable this check by passing 'safe=False'.")
    data, index = flowutils.compensate.get_spill(metadata["spill"])
    columns = [s.replace("-A (SW-unmix)","") for s in metadata['bdspectral probes'].split(",")]
    spill = pd.DataFrame(data[:len(columns)].T, index=index, columns=columns)
    return spill


class Gate():
    """One polygon gate of the gating strategy that the S8 software stores in the .fcs metadata.

    Parameters
    ----------
    g : dict
        One entry of ``["RecordingConfiguration"]["AnalysisModel"]["Gates"]`` of the 'BDCHORUSDATARECORD' JSON
        keyword, with at least two 'Parameters' (the gate axes).

    Attributes
    ----------
    id, id_gate, id_parent : str
        Population ID of the gated population, gate ID, and population ID of the parent ("root" for the top level).
    name : str
        Name of the gated population.
    key_x, key_y : str
        Channel names of the axes: "<fluorochrome>-<measurement>" for colour parameters,
        "<measurement id>-<measurement>" for scatter, else the measurement ID.
    vertices : numpy.ndarray
        Polygon vertices of shape (vertices, 2), as stored by the S8 software.
    name_parent, gate_order : str, tuple of str
        Set by `add_parent_name` and `add_order`.
    """
    def __init__(self, g):
        self.id_parent = g["ParentPopulationId"]
        if self.id_parent=="0-1":
            self.id_parent = "root"
        self.id = g["Children"][0]["PopulationId"]
        self.id_gate = g["GateId"]
        self.name = g["Children"][0]["Name"]
        self.vertices = g["Vertices"]
        
        def get_key(dct):
            if dct["ParameterKind"]=="Color":
                return dct["Fluorochrome"]+"-"+dct["Measurement"]
            elif dct["ParameterKind"]=="Scatter":
                return dct["MeasurementId"]+"-"+dct["Measurement"]
            else:
                return dct["MeasurementId"]
        self.key_x = get_key(g["Parameters"][0])
        self.key_y = get_key(g["Parameters"][1])
        
        self.vertices = pd.DataFrame(g["Vertices"])[["X","Y"]].to_numpy()
    
    def __repr__(self):
        ret = f"Gate '{self.name}' ({self.id}), child of '{self.id_parent if not hasattr(self, 'name_parent') else self.name_parent}'; on ({self.key_x}, {self.key_y})"
        return ret
    
    def add_parent_name(self, gatekey):
        """Set `name_parent` from `gatekey` (pandas.Series: population ID -> population name); "root" stays."""
        self.name_parent = gatekey.loc[self.id_parent] if self.id_parent!="root" else self.id_parent
    
    def add_order(self, connectkey):
        """Set `gate_order`, the gate path from "root" to the parent population (names), as flowkit expects it.

        Parameters
        ----------
        connectkey : pandas.Series
            Population name -> parent population name, for all gates. The path is followed for at most 10 levels.
        """
        order = [self.name_parent]
        while order[-1]!="root" and len(order)<10:
            order.append(connectkey.loc[order[-1]])
        self.gate_order = tuple(order[::-1])
    
    def to_fk(self):
        """Convert to a flowkit gate (uncompensated dimensions); needs `add_order` first.

        Returns
        -------
        gate : flowkit.gates.PolygonGate
        gate_path : tuple of str
            `gate_order`, for ``flowkit.GatingStrategy.add_gate``.
        """
        dim_x = fk.Dimension(self.key_x, compensation_ref='uncompensated')
        dim_y = fk.Dimension(self.key_y, compensation_ref='uncompensated')
        
        gate = fk.gates.PolygonGate( self.name, dimensions=[dim_x, dim_y], vertices=self.vertices )
        return gate, self.gate_order

def load_gates(g_list):
    """`Gate` objects for the entries of an S8 gate list that have parameters (axes); others are skipped.

    Parameters
    ----------
    g_list : list of dict
        ``["RecordingConfiguration"]["AnalysisModel"]["Gates"]`` of the 'BDCHORUSDATARECORD' JSON keyword.

    Returns
    -------
    list of Gate
    """
    gates = []
    for g in g_list:
        if len(g["Parameters"])>0:
            gates.append(Gate(g))
    return gates

def sort_gates(gates):
    """Reorder `gates` (in place, also returned) so that every parent gate comes before its children."""
    for it in range(len(gates)):
        for i in range(len(gates)):
            for j in range(i):
                if gates[i].id == gates[j].id_parent:
                    gates[j], gates[i] = gates[i], gates[j]
    return gates

def load_gates_fromfile(filepath):
    """Sorted `Gate` objects from the 'bdchorusdatarecord' keyword of an S8 .fcs file.

    Parent names and gate paths are not set (`load_S8_metadata_gatingstrategy` adds those).

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.

    Returns
    -------
    list of Gate
        Parents before children (see `sort_gates`).
    """
    metadct = readfcs_metadata(filepath)
    return sort_gates(load_gates(json.loads(metadct['bdchorusdatarecord'])["RecordingConfiguration"]["AnalysisModel"]["Gates"]))

def load_S8_metadata_gatingstrategy(filepath, verbose=False):
    """Read the gating strategy that the S8 software stores in an .fcs file into a flowkit GatingStrategy.

    The gates are taken from the JSON keyword 'BDCHORUSDATARECORD' (``RecordingConfiguration.AnalysisModel.Gates``);
    all gates with axes become polygon gates on uncompensated data.

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.
    verbose : bool, default False
        Print the gate hierarchy.

    Returns
    -------
    flowkit.GatingStrategy
        The gating strategy, with the population names of the S8 software as gate names.

    Examples
    --------
    >>> g_strat = load_S8_metadata_gatingstrategy(path)
    >>> res = g_strat.gate_sample(flowkit.Sample(path))
    """
    metadct = readfcs_metadata(filepath)
    jsn = json.loads(metadct['bdchorusdatarecord']) if 'bdchorusdatarecord' in metadct else json.loads(metadct['BDCHORUSDATARECORD'])
    gatestr = jsn["RecordingConfiguration"]["AnalysisModel"]["Gates"]
    
    gates = sort_gates(load_gates(gatestr))
    gatekey = pd.Series([g.name for g in gates], index=[g.id for g in gates])
    for g in gates: g.add_parent_name(gatekey)
    connectkey = pd.Series([g.name_parent for g in gates], index=[g.name for g in gates])
    for g in gates: g.add_order(connectkey)
    
    g_strat = fk.GatingStrategy()
    for g in gates:
        gate, path = g.to_fk()
        g_strat.add_gate(gate, gate_path=path)
    if verbose:
        print(g_strat.get_gate_hierarchy())
    
    return g_strat

S8_PIXEL_WIDTH = 60/104 # 104 pixels, approx 60um line width according to BD
S8_PIXEL_HEIGHT_RATIO = 1/1.3 # eccentricity of comp beads seems minimal with this ratio, no idea if correct or not?
S8_PIXEL_AREA = S8_PIXEL_WIDTH**2 * S8_PIXEL_HEIGHT_RATIO

def S8_size_to_surface(size):
    """Surface of a sphere with cross-section area `size` (4 x the cross-section, units squared as `size`)."""
    return 4*size

def S8_size_to_diameter(size):
    """Diameter of a circle with area `size`: A = pi/4 D^2 (units: square root of those of `size`)."""
    return np.sqrt(4/np.pi*size)

def S8_size_to_volume(size):
    """Volume of a sphere with cross-section area `size`: 4 / (3 sqrt(pi)) * A^(3/2)."""
    return 4/(3*np.sqrt(np.pi)) * size**(3/2)


def calculate_single_eccentricity_angle(mask, pixel_aspect_ratio=S8_PIXEL_HEIGHT_RATIO):
    """Eccentricity and major-axis angle of a cell mask, from its second moments, with pixel aspect correction.

    Parameters
    ----------
    mask : numpy.ndarray
        2-D boolean mask of shape (rows, columns).
    pixel_aspect_ratio : float, default S8_PIXEL_HEIGHT_RATIO
        Pixel height / width (row spacing / column spacing): 1.0 for square pixels, 1.2 for pixels 1.2 x taller than
        wide. Row coordinates are multiplied by it before the moments are computed.

    Returns
    -------
    eccentricity : float
        sqrt(1 - lambda_min / lambda_max) of the second-moment matrix, in [0, 1]: 0 for a circle, close to 1 for a
        line.
    angle : float
        Angle of the major axis in degrees, in [-90, 90]: 0 along the rows axis ("vertical", image rows running
        downwards), 90 along the columns axis. Positive when the major axis runs from low row and column indices to
        high ones. Not meaningful for eccentricity 0 (a circle gives 90).

    Notes
    -----
    Both are NaN if the mask touches the image boundary (cell cut off); an empty mask gives (0, 0).

    See Also
    --------
    calculate_batched_eccentricity_angle : the same for a list of masks.
    """
    # return nan if mask touches the boundary, cell cut off
    if mask[0,:].sum() + mask[-1,:].sum() + mask[:,0].sum() + mask[:,-1].sum() > 0:
        return np.nan, np.nan
    
    y, x = np.where(mask)
    
    if len(x) == 0:
        return 0.0, 0.0
    
    # Scale to physical units
    x_scaled = x
    y_scaled = y * pixel_aspect_ratio
    
    # Calculate moments in corrected space
    x_centered = x_scaled - x_scaled.mean()
    y_centered = y_scaled - y_scaled.mean()
    
    mu_20 = np.sum(x_centered**2)
    mu_02 = np.sum(y_centered**2)
    mu_11 = np.sum(x_centered * y_centered)
    
    term1 = (mu_20 + mu_02) / 2
    term2 = np.sqrt(4 * mu_11**2 + (mu_20 - mu_02)**2) / 2
    
    lambda_max = term1 + term2
    lambda_min = term1 - term2
    
    # Calculate eccentricity
    if lambda_max == 0:
        eccentricity = 0.0
    else:
        eccentricity = np.sqrt(1 - lambda_min / lambda_max)
    
    # Calculate angle
    angle_rad = 0.5 * np.arctan2(2 * mu_11, mu_20 - mu_02)
    angle_deg = np.degrees(angle_rad)
    
    # Convert to angle from vertical (y-axis)
    angle_from_vertical = 90 - angle_deg
    
    # Normalize to [-90, 90]
    if angle_from_vertical > 90:
        angle_from_vertical -= 180
    elif angle_from_vertical < -90:
        angle_from_vertical += 180
    
    return eccentricity, angle_from_vertical

def _pad_masks(mask_list):
    """Stack masks of equal width into one array of shape (masks, max height, width), zero-padded at the bottom."""
    max_h = max(m.shape[0] for m in mask_list)
    width = mask_list[0].shape[1]
    masks_padded = np.zeros((len(mask_list), max_h, width), dtype=bool)
    
    for i, m in enumerate(mask_list):
        masks_padded[i, :m.shape[0], :] = m

    return masks_padded

def calculate_batched_eccentricity_angle(mask_list, pixel_aspect_ratio=S8_PIXEL_HEIGHT_RATIO):
    """`calculate_single_eccentricity_angle` for a list of masks; gives the same results.

    Parameters
    ----------
    mask_list : list of numpy.ndarray
        2-D boolean masks of the same width; heights may differ.
    pixel_aspect_ratio : float, default S8_PIXEL_HEIGHT_RATIO
        Pixel height / width, see `calculate_single_eccentricity_angle`.

    Returns
    -------
    eccentricity, angle : numpy.ndarray
        Arrays of shape (masks,); NaN where a mask touches its own boundary (cell cut off).
    """
    heights = np.asarray([m.shape[0] for m in mask_list])
    masks = _pad_masks(mask_list)
    
    n_masks = masks.shape[0]
    eccentricities = np.full(n_masks, np.nan)
    angles = np.full(n_masks, np.nan)
    
    # Check boundaries; the bottom edge is each mask's own last row, not the last row of the padded array
    touches_boundary = (
        masks[:, 0, :].any(axis=1) | 
        masks[np.arange(n_masks), heights-1, :].any(axis=1) | 
        masks[:, :, 0].any(axis=1) | 
        masks[:, :, -1].any(axis=1)
    )
    
    # Process each mask (but vectorize the moment calculations)
    for i in range(n_masks):
        if touches_boundary[i]:
            continue
            
        y, x = np.where(masks[i])
        
        if len(x) == 0:
            eccentricities[i] = 0.0
            angles[i] = 0.0
            continue
        
        # Vectorized moment calculation
        x_scaled = x.astype(float)
        y_scaled = y.astype(float) * pixel_aspect_ratio
        
        cx = x_scaled.mean()
        cy = y_scaled.mean()
        
        x_c = x_scaled - cx
        y_c = y_scaled - cy
        
        # All vectorized operations on 1D arrays
        mu_20 = (x_c * x_c).sum()
        mu_02 = (y_c * y_c).sum()
        mu_11 = (x_c * y_c).sum()
        
        term1 = (mu_20 + mu_02) / 2
        term2 = np.sqrt(4 * mu_11**2 + (mu_20 - mu_02)**2) / 2
        
        lambda_max = term1 + term2
        lambda_min = term1 - term2
        
        if lambda_max > 0:
            eccentricities[i] = np.sqrt(1 - lambda_min / lambda_max)
        else:
            eccentricities[i] = 0.0
        
        angle_rad = 0.5 * np.arctan2(2 * mu_11, mu_20 - mu_02)
        angle_deg = 90 - np.degrees(angle_rad)
        
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
            
        angles[i] = angle_deg
    
    return eccentricities, angles

def add_S8_singlet_maskfeatures(ad, imgcont, pixel_aspect_ratio=S8_PIXEL_HEIGHT_RATIO):
    """Add size and shape features of singlet cells from the S8 image segmentation to ``ad.obs`` (in place).

    Adds 'cell_crosssection' (``seg_cell_totsize`` x `S8_PIXEL_AREA`, in um^2), 'cell_volume' (um^3),
    'cell_diameter' (um) and 'cell_surface' (um^2) of a sphere with that cross-section, and 'cell_eccentricity' and
    'cell_deformangle' (degrees) from the cell masks (see `calculate_single_eccentricity_angle`). All are NaN unless
    ``ad.obs['seg_cell_count'] == 1``; eccentricity and angle are also NaN for cells cut off at the image boundary.
    The physical units rest on approximate pixel dimensions (`S8_PIXEL_WIDTH` from the line width stated by BD,
    `S8_PIXEL_HEIGHT_RATIO` estimated from compensation beads).

    Parameters
    ----------
    ad : anndata.AnnData
        S8 events with ``ad.obs['seg_cell_totsize']`` (segmented cell area in pixels) and
        ``ad.obs['seg_cell_count']``.
    imgcont : object
        Image container, duck-typed and not part of this package: any object with a method
        ``load_masks_batched(index)`` that returns a list of 2-D boolean cell masks in the order of `index`
        (here ``ad.obs.index``).
    pixel_aspect_ratio : float, default S8_PIXEL_HEIGHT_RATIO
        Pixel height / width, see `calculate_single_eccentricity_angle`.
    """
    ad.obs["cell_crosssection"] = ad.obs["seg_cell_totsize"] * S8_PIXEL_AREA
    ad.obs.loc[ad.obs["seg_cell_count"]!=1, "cell_crosssection"] = np.nan # mask everything except singlets
    
    ad.obs["cell_volume"] = S8_size_to_volume(ad.obs["cell_crosssection"])
    ad.obs["cell_diameter"] = S8_size_to_diameter(ad.obs["cell_crosssection"])
    ad.obs["cell_surface"] = S8_size_to_surface(ad.obs["cell_crosssection"])

    masks = imgcont.load_masks_batched(ad.obs.index)
    ad.obs["cell_eccentricity"], ad.obs["cell_deformangle"] = calculate_batched_eccentricity_angle(masks, pixel_aspect_ratio=pixel_aspect_ratio)
    ad.obs.loc[ad.obs["seg_cell_count"]!=1, "cell_eccentricity"] = np.nan
    ad.obs.loc[ad.obs["seg_cell_count"]!=1, "cell_deformangle"] = np.nan
