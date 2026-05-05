import flowio
import flowutils
import flowkit as fk
import pandas as pd
import numpy as np
import json

from .utils import readfcs_metadata


def read_spill_S8(samplepath, safe=True):
    """ Read spillover matrix from .fcs file that was recorded on an S8 cytometer.
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
        self.name_parent = gatekey.loc[self.id_parent] if self.id_parent!="root" else self.id_parent
    
    def add_order(self, connectkey):
        order = [self.name_parent]
        while order[-1]!="root" and len(order)<10:
            order.append(connectkey.loc[order[-1]])
        self.gate_order = tuple(order[::-1])
    
    def to_fk(self):
        dim_x = fk.Dimension(self.key_x, compensation_ref='uncompensated')
        dim_y = fk.Dimension(self.key_y, compensation_ref='uncompensated')
        
        gate = fk.gates.PolygonGate( self.name, dimensions=[dim_x, dim_y], vertices=self.vertices )
        return gate, self.gate_order

def load_gates(g_list):
    gates = []
    for g in g_list:
        if len(g["Parameters"])>0:
            gates.append(Gate(g))
    return gates

def sort_gates(gates):
    for it in range(len(gates)):
        for i in range(len(gates)):
            for j in range(i):
                if gates[i].id == gates[j].id_parent:
                    gates[j], gates[i] = gates[i], gates[j]
    return gates

def load_gates_fromfile(filepath):
    metadct = readfcs_metadata(filepath)
    return sort_gates(load_gates(json.loads(metadct['bdchorusdatarecord'])["RecordingConfiguration"]["AnalysisModel"]["Gates"]))

def load_S8_metadata_gatingstrategy(filepath, verbose=False):
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
    """ S8 records cross-section, x4 for full surface
    """
    return 4*size

def S8_size_to_diameter(size):
    """ S8 records cross-section A, A=pi/4 D²
    """
    return np.sqrt(4/np.pi*size)

def S8_size_to_volume(size):
    """ S8 records cross-section A, sphere volume is 4/(3 sqrt(pi)) * A^(3/2)
    """
    return 4/(3*np.sqrt(np.pi)) * size**(3/2)


def calculate_single_eccentricity_angle(mask, pixel_aspect_ratio=S8_PIXEL_HEIGHT_RATIO):
    """
    Calculate eccentricity and major axis angle with pixel aspect ratio correction.
    
    Parameters:
    mask: 2D boolean array
    pixel_aspect_ratio: height/width of pixels (dy/dx)
                       1.0 = square pixels
                       1.2 = pixels are 1.2x taller than wide
    
    Returns:
    eccentricity: float in [0, 1), 0 = circle, 1 = line
    angle: angle in degrees, range [-90, 90]
           0° = vertical (aligned with y-axis)
           90° = horizontal (aligned with x-axis)
           positive = clockwise from vertical
    Both return np.nan if mask touches boundary
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
    max_h = max(m.shape[0] for m in mask_list)
    width = mask_list[0].shape[1]
    masks_padded = np.zeros((len(mask_list), max_h, width), dtype=bool)
    
    for i, m in enumerate(mask_list):
        masks_padded[i, :m.shape[0], :] = m

    return masks_padded

def calculate_batched_eccentricity_angle(mask_list, pixel_aspect_ratio=S8_PIXEL_HEIGHT_RATIO):
    """
    Fast vectorized version - processes coordinates, not full grids.
    
    Parameters:
    mask_list: list of boolean masks
    """
    masks = _pad_masks(mask_list)
    
    n_masks = masks.shape[0]
    eccentricities = np.full(n_masks, np.nan)
    angles = np.full(n_masks, np.nan)
    
    # Check boundaries
    touches_boundary = (
        masks[:, 0, :].any(axis=1) | 
        masks[:, -1, :].any(axis=1) | 
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
    ad.obs["cell_crosssection"] = ad.obs["seg_cell_totsize"] * S8_PIXEL_AREA
    ad.obs.loc[ad.obs["seg_cell_count"]!=1, "cell_crosssection"] = np.nan # mask everything except singlets
    
    ad.obs["cell_volume"] = S8_size_to_volume(ad.obs["cell_crosssection"])
    ad.obs["cell_diameter"] = S8_size_to_diameter(ad.obs["cell_crosssection"])
    ad.obs["cell_surface"] = S8_size_to_surface(ad.obs["cell_crosssection"])

    masks = imgcont.load_masks_batched(ad.obs.index)
    ad.obs["cell_eccentricity"], ad.obs["cell_deformangle"] = calculate_batched_eccentricity_angle(masks, pixel_aspect_ratio=pixel_aspect_ratio)
    ad.obs.loc[ad.obs["seg_cell_count"]!=1, "cell_eccentricity"] = np.nan
    ad.obs.loc[ad.obs["seg_cell_count"]!=1, "cell_deformangle"] = np.nan
