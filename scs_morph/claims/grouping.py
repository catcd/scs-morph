from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd


MISSING_GROUP_VALUES = {"", "nan", "na", "none", "null", "unknown", "unknown_subregion", "unknown_network"}


SUBCORTICAL_GROUPS = {
    "ventricle_csf": {"Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent", "3rd-Ventricle", "4th-Ventricle", "5th-Ventricle", "CSF", "Left-choroid-plexus", "Right-choroid-plexus"},
    "cerebellum": {"Left-Cerebellum-White-Matter", "Right-Cerebellum-White-Matter", "Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"},
    "thalamus": {"Left-Thalamus", "Right-Thalamus"},
    "basal_ganglia": {"Left-Caudate", "Right-Caudate", "Left-Putamen", "Right-Putamen", "Left-Pallidum", "Right-Pallidum", "Left-Accumbens-area", "Right-Accumbens-area"},
    "medial_temporal": {"Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala"},
    "ventral_diencephalon": {"Left-VentralDC", "Right-VentralDC"},
    "brainstem": {"Brain-Stem"},
    "vessel": {"Left-vessel", "Right-vessel"},
    "hypointensity": {"WM-hypointensities", "Left-WM-hypointensities", "Right-WM-hypointensities", "non-WM-hypointensities", "Left-non-WM-hypointensities", "Right-non-WM-hypointensities"},
    "optic_chiasm": {"Optic-Chiasm"},
    "corpus_callosum": {"CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior"},
    "global_brain_volume": {"BrainSegVol", "lhCortexVol", "rhCortexVol", "CortexVol", "lhCerebralWhiteMatterVol", "rhCerebralWhiteMatterVol", "CerebralWhiteMatterVol", "SubCortGrayVol", "TotalGrayVol", "SupraTentorialVol", "SupraTentorialVolNotVent", "MaskVol", "BrainSegVol-to-eTIV", "MaskVol-to-eTIV", "EstimatedTotalIntraCranialVol", "SegmentedTotalIntracranialVol"},
    "surface_qc": {"lhSurfaceHoles", "rhSurfaceHoles", "SurfaceHoles"},
}

CORTICAL_LOBES = {
    "temporal": {"bankssts", "entorhinal", "fusiform", "inferiortemporal", "middletemporal", "parahippocampal", "superiortemporal", "temporalpole", "transversetemporal"},
    "frontal": {"caudalmiddlefrontal", "lateralorbitofrontal", "medialorbitofrontal", "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "precentral", "rostralmiddlefrontal", "superiorfrontal", "frontalpole"},
    "parietal": {"inferiorparietal", "postcentral", "precuneus", "superiorparietal", "supramarginal"},
    "occipital": {"cuneus", "lateraloccipital", "lingual", "pericalcarine"},
    "cingulate": {"caudalanteriorcingulate", "isthmuscingulate", "posteriorcingulate", "rostralanteriorcingulate"},
    "insula": {"insula"},
}

SCHAEFER7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
SCHAEFER17_MAP = {
    "VisCent": "Vis", "VisPeri": "Vis", "SomMotA": "SomMot", "SomMotB": "SomMot",
    "DorsAttnA": "DorsAttn", "DorsAttnB": "DorsAttn", "SalVentAttnA": "SalVentAttn",
    "SalVentAttnB": "SalVentAttn", "LimbicA": "Limbic", "LimbicB": "Limbic",
    "ContA": "Cont", "ContB": "Cont", "ContC": "Cont", "DefaultA": "Default",
    "DefaultB": "Default", "DefaultC": "Default", "TempPar": "TempPar",
}
KONG17_MAP = {
    "DefaultA": "Default", "DefaultB": "Default", "DefaultC": "Default", "ContA": "Cont",
    "ContB": "Cont", "ContC": "Cont", "Language": "Language", "SalVenAttnA": "SalVentAttn",
    "SalVenAttnB": "SalVentAttn", "DorsAttnA": "DorsAttn", "DorsAttnB": "DorsAttn",
    "Aud": "Aud", "SomMotA": "SomMot", "SomMotB": "SomMot", "VisualA": "Vis",
    "VisualB": "Vis", "VisualC": "Vis",
}
YALE_GROUPS = {
    "TP": "temporal_pole", "T": "temporal", "O": "occipital", "AN": "anterior_region",
    "SM": "sensorimotor", "P": "parietal", "S": "superior_region", "M": "medial_region",
    "SF": "superior_frontal", "MF": "middle_frontal", "OP": "opercular", "TR": "triangular",
    "OR": "orbital", "FP": "frontal_pole", "FO": "fronto_orbital",
    "LOT": "lateral_occipitotemporal", "MOT": "medial_occipitotemporal", "PH": "parahippocampal",
    "L": "lateral_region", "CG": "cingulate", "H": "hippocampal_adjacent",
    "A": "amygdala_adjacent_or_anterior_temporal", "I": "insula", "CC": "corpus_callosum_or_cingulate_adjacent",
}


def _is_valid_group_value(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text.lower() not in MISSING_GROUP_VALUES


def _first_valid(*values) -> str | None:
    for value in values:
        if _is_valid_group_value(value):
            return str(value).strip()
    return None


def parse_hemisphere(roi_name) -> str | None:
    roi = str(roi_name)
    if roi.startswith(("lh_", "LH_")) or "_LH_" in roi or "_L_" in roi or roi.startswith("Left-"):
        return "LH"
    if roi.startswith(("rh_", "RH_")) or "_RH_" in roi or "_R_" in roi or roi.startswith("Right-"):
        return "RH"
    return None


def make_group_label(row_or_dict, feature_space=None) -> str:
    """Choose the best non-missing human group label for a grouped ROI row."""
    row = row_or_dict.to_dict() if hasattr(row_or_dict, "to_dict") else dict(row_or_dict or {})
    fs = str(feature_space or "").lower()

    if "yale" in fs:
        roi_group = _first_valid(row.get("roi_group"))
        if roi_group:
            return f"Yale code-group {roi_group}"
        return _first_valid(row.get("lobe")) or "unclassified regions"

    if "schaefer" in fs or "kong" in fs:
        return _first_valid(
            row.get("network"),
            row.get("network_7_like"),
            row.get("roi_group"),
            row.get("anatomical_group"),
            row.get("lobe"),
        ) or "unclassified regions"

    if "cortical_volume" in fs:
        return _first_valid(row.get("lobe"), row.get("roi_group"), row.get("anatomical_group")) or "unclassified regions"

    if "subcortical" in fs:
        return _first_valid(row.get("roi_group"), row.get("anatomical_group")) or "unclassified regions"

    return _first_valid(
        row.get("roi_group"),
        row.get("anatomical_group"),
        row.get("network"),
        row.get("network_7_like"),
        row.get("lobe"),
    ) or "unclassified regions"


def format_group_label(group_label, feature_space=None) -> str:
    """Format a safe group label for conservative claim text."""
    if not _is_valid_group_value(group_label):
        return "unclassified regions"
    label = str(group_label).strip()

    if label.startswith("Yale code-group "):
        group = label.removeprefix("Yale code-group ").replace("_", " ")
        return f"Yale code-group {group} regions"

    exact = {
        "temporal": "temporal regions",
        "parietal": "parietal regions",
        "frontal": "frontal regions",
        "occipital": "occipital regions",
        "cingulate": "cingulate regions",
        "insula": "insula regions",
        "medial_temporal": "medial temporal structures",
        "basal_ganglia": "basal ganglia structures",
        "ventricle_csf": "ventricular/CSF-related regions",
        "global_brain_volume": "global brain-volume measures",
        "corpus_callosum": "corpus callosum regions",
        "Default": "Default network parcels",
        "Cont": "Control network parcels",
        "DorsAttn": "Dorsal attention network parcels",
        "SalVentAttn": "Salience/ventral attention network parcels",
        "SomMot": "Somatomotor network parcels",
        "Vis": "Visual network parcels",
        "Limbic": "Limbic network parcels",
    }
    if label in exact:
        return exact[label]

    network_match = re.fullmatch(r"(Default|Cont|Limbic)([ABC])", label)
    if network_match:
        base, suffix = network_match.groups()
        base_text = {"Default": "Default", "Cont": "Control", "Limbic": "Limbic"}[base]
        return f"{base_text} {suffix} network parcels"

    text = label.replace("_", " ")
    if text.endswith(("measures", "structures", "parcels", "regions")):
        return text
    return f"{text} regions"


def broad_lobe_from_subregion(subregion) -> str:
    sub = str(subregion)
    groups = {
        "temporal": {"temporal", "temporal_pole", "superior_temporal", "parahippocampal", "auditory_superior_temporal", "temporo_occipital", "anterior_temporal", "auditory"},
        "parietal": {"inferior_parietal", "superior_parietal", "intraparietal_sulcus", "postcentral_parietal", "postcentral", "precuneus", "medial_parietal", "parieto_occipital"},
        "frontal": {"prefrontal", "frontal_pole", "medial_frontal", "orbitofrontal", "frontal_eye_field", "precentral", "opercular", "frontal_operculum"},
        "occipital": {"visual_occipital"},
        "cingulate": {"cingulate", "posterior_cingulate", "middle_cingulate", "retrosplenial", "precuneus_pcc"},
        "insula": {"insula", "operculo_insular"},
        "sensorimotor": {"secondary_somatosensory", "central_sensorimotor"},
        "medial": {"medial"},
    }
    for lobe, values in groups.items():
        if sub in values:
            return lobe
    return "unknown"


def _subregion(tokens: list[str]) -> str:
    text = "_".join(tokens)
    checks = [
        ({"pCunPCC"}, "precuneus_pcc"),
        ({"FrOperIns"}, "operculo_insular"),
        ({"Aud_ST"}, "auditory_superior_temporal"),
        ({"ExStr", "Striate", "StriCal", "ExStrInf", "ExStrSup"}, "visual_occipital"),
        ({"Aud"}, "auditory"), ({"S2"}, "secondary_somatosensory"), ({"Cent"}, "central_sensorimotor"),
        ({"TempOcc"}, "temporo_occipital"), ({"ParOcc"}, "parieto_occipital"), ({"SPL"}, "superior_parietal"),
        ({"Post", "PostC"}, "postcentral"), ({"FEF"}, "frontal_eye_field"), ({"ParOper"}, "parietal_operculum"),
        ({"Ins"}, "insula"), ({"FrOper"}, "frontal_operculum"),
        ({"ParMed"}, "medial_parietal"), ({"FrMed"}, "medial_frontal"), ({"IPL"}, "inferior_parietal"),
        ({"PFCl", "PFClv", "PFCld", "PFCmp", "PFCd", "PFCm", "PFCv", "PFCdPFCm"}, "prefrontal"),
        ({"OFC"}, "orbitofrontal"), ({"TempPole"}, "temporal_pole"), ({"Temp"}, "temporal"),
        ({"IPS"}, "intraparietal_sulcus"), ({"Cingm"}, "middle_cingulate"), ({"Cingp", "PCC"}, "posterior_cingulate"),
        ({"pCun"}, "precuneus"), ({"Rsp", "RSC"}, "retrosplenial"),
        ({"PHC"}, "parahippocampal"), ({"AntTemp"}, "anterior_temporal"), ({"PrCv", "PrC", "PrCd"}, "precentral"),
        ({"Oper"}, "opercular"), ({"ST"}, "auditory_superior_temporal"), ({"FPole"}, "frontal_pole"),
    ]
    for keys, label in checks:
        if any(key in text for key in keys):
            return label
    return "unknown_subregion"


def _base_group(roi, feature_space, group_source):
    hemi = parse_hemisphere(roi)
    return {"hemisphere": hemi, "roi_group": None, "anatomical_group": None, "network": None, "lobe": None, "group_source": group_source, "network_7_like": None}


def infer_roi_group(roi_name, feature_space=None) -> dict:
    roi = str(roi_name)
    fs = feature_space or ""
    out = _base_group(roi, fs, fs or "generic")
    if fs in {"fs8_subcortical_volume", "fastsurfer_subcortical_volume"}:
        group = next((g for g, names in SUBCORTICAL_GROUPS.items() if roi in names), "other_subcortical")
        out.update({"roi_group": group, "anatomical_group": group})
        return out
    if fs in {"fs8_cortical_volume", "fastsurfer_cortical_volume"}:
        base = re.sub(r"^(lh_|rh_)", "", roi)
        group = next((g for g, names in CORTICAL_LOBES.items() if base in names), "other_cortical")
        out.update({"roi_group": group, "anatomical_group": group, "lobe": group})
        return out
    if fs in {"fs8_thickness_schaefer200_7", "spm_thickness_schaefer200_7"}:
        parts = roi.split("_")
        network = next((p for p in parts if p in SCHAEFER7), None)
        sub = _subregion(parts[2:] if len(parts) > 2 else [])
        out.update({"roi_group": network, "anatomical_group": sub, "network": network, "network_7_like": network, "lobe": broad_lobe_from_subregion(sub)})
        return out
    if fs == "spm_thickness_schaefer200_17":
        parts = roi.split("_")
        network = next((p for p in parts if p in SCHAEFER17_MAP), None)
        sub = _subregion(parts[2:] if len(parts) > 2 else [])
        out.update({"roi_group": network, "anatomical_group": sub, "network": network, "network_7_like": SCHAEFER17_MAP.get(network), "lobe": broad_lobe_from_subregion(sub)})
        return out
    if fs == "fs8_thickness_kong200_17":
        parts = roi.split("_")
        network = next((p for p in parts if p in KONG17_MAP), None)
        sub = _subregion(parts[4:] if len(parts) > 4 else parts)
        out.update({"roi_group": network, "anatomical_group": sub, "network": network, "network_7_like": KONG17_MAP.get(network), "lobe": broad_lobe_from_subregion(sub)})
        return out
    if fs == "fs8_thickness_yale696":
        match = re.search(r"_[LR]_([A-Z]+)\d*_", roi)
        code = match.group(1) if match else None
        group = YALE_GROUPS.get(code or "", "unknown_yale")
        lobe = "other_yale"
        if code in {"TP", "T", "LOT", "MOT", "PH", "H", "A"}:
            lobe = "temporal"
        elif code == "O":
            lobe = "occipital"
        elif code == "P":
            lobe = "parietal"
        elif code in {"SF", "MF", "OP", "TR", "OR", "FP", "FO"}:
            lobe = "frontal"
        elif code == "SM":
            lobe = "sensorimotor"
        elif code in {"CG", "CC"}:
            lobe = "cingulate"
        elif code == "I":
            lobe = "insula"
        elif code == "M":
            lobe = "medial"
        out.update({"roi_group": group, "anatomical_group": group, "lobe": lobe})
        return out
    out.update({"roi_group": "unknown", "anatomical_group": "unknown", "lobe": "unknown"})
    return out


def annotate_roi_statistics(results_df: pd.DataFrame, feature_space) -> pd.DataFrame:
    rows = [infer_roi_group(roi, feature_space) for roi in results_df["roi"]]
    return pd.concat([results_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summarize_roi_groups(results_df: pd.DataFrame, support_regions, feature_space) -> pd.DataFrame:
    annotated = annotate_roi_statistics(results_df, feature_space) if "roi_group" not in results_df.columns else results_df.copy()
    annotated["is_support"] = annotated["roi"].isin(support_regions)
    group_cols = ["roi_group", "anatomical_group", "network", "network_7_like", "lobe"]
    rows = []
    for keys, group in annotated.groupby(group_cols, dropna=False):
        t = pd.to_numeric(group["t"], errors="coerce")
        coef = pd.to_numeric(group["coef"], errors="coerce")
        support = group[group["is_support"]]
        mean_coef = float(coef.mean()) if len(group) else np.nan
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_group_rois": int(len(group)),
            "n_support_rois": int(len(support)),
            "fraction_support": float(len(support) / len(group)) if len(group) else 0.0,
            "mean_t": float(t.mean()) if len(group) else np.nan,
            "mean_abs_t": float(t.abs().mean()) if len(group) else np.nan,
            "mean_coef": mean_coef,
            "direction": "higher" if mean_coef > 0 else "lower" if mean_coef < 0 else "altered",
            "support_rois": "; ".join(support["roi"].astype(str).tolist()),
        })
    return pd.DataFrame(rows).sort_values(["n_support_rois", "mean_abs_t"], ascending=False)


def make_region_summary(support_regions, feature_space, max_regions=5) -> str:
    regions = list(support_regions)
    if len(regions) <= max_regions:
        return ", ".join(regions) if regions else "no selected regions"
    groups = [infer_roi_group(roi, feature_space) for roi in regions]
    if "schaefer" in feature_space or "kong" in feature_space:
        labels = [g.get("network_7_like") or g.get("network") for g in groups]
        suffix = "network parcels"
    elif "cortical_volume" in feature_space:
        labels = [g.get("lobe") for g in groups]
        suffix = "regions"
    elif "subcortical" in feature_space:
        labels = [g.get("anatomical_group") for g in groups]
        suffix = "structures"
    elif "yale" in feature_space:
        labels = [g.get("anatomical_group") for g in groups]
        suffix = "Yale code-groups"
    else:
        labels = [g.get("roi_group") for g in groups]
        suffix = "regions"
    common = [label for label, _ in Counter([x for x in labels if x]).most_common(3)]
    if not common:
        return ", ".join(regions[:max_regions])
    if len(common) == 1:
        text = common[0]
    elif len(common) == 2:
        text = f"{common[0]} and {common[1]}"
    else:
        text = f"{common[0]}, {common[1]}, and {common[2]}"
    return f"{text} {suffix}"
