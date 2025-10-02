"""Script for scoring peaks in MS data using spectral libraries."""

import numpy as np
import pandas as pd

from alphadia_search_rs import SpecLibFlat, DIAData, PeakGroupSelection


class PeakScoring:
    """Peak scoring class for MS2 extraction based on MS1 features."""

    def __init__(
        self,
        dia_data,
        precursor_df,
        fragment_df,
        config_dict,
        rt_column="rt_library",
        mobility_column="mobility_library",
        precursor_mz_column="mz_library",
        fragment_mz_column="mz_library",
        fwhm_rt=5.0,
        fwhm_mobility=0.012,
    ):
        """Select candidates for MS2 extraction based on MS1 features.

        Parameters
        ----------
        dia_data : alphadia.data.bruker.TimsTOFDIA
            dia data object
        precursors_flat : pandas.DataFrame
            flattened precursor dataframe
        rt_column : str, optional
            name of the rt column in the precursor dataframe, by default 'rt_library'
        mobility_column : str, optional
            name of the mobility column in the precursor dataframe, by default 'mobility_library'
        precursor_mz_column : str, optional
            name of the precursor mz column in the precursor dataframe, by default 'mz_library'
        fragment_mz_column : str, optional
            name of the fragment mz column in the fragment dataframe, by default 'mz_library'

        Returns
        -------
        pandas.DataFrame
            dataframe containing the extracted candidates
        """
        self.prepare_dia_data(dia_data)
        self.prepare_spec_lib(
            precursor_df,
            fragment_df,
            rt_column,
            precursor_mz_column,
            fragment_mz_column,
            mobility_column,
        )

        self.config_dict = config_dict
        self.precursor_df = precursor_df

    def prepare_dia_data(self, dia_data):
        """Prepare DIA data for peak scoring."""
        cycle_len = dia_data.cycle.shape[1]
        delta_scan_idx = np.tile(
            np.arange(cycle_len), int(len(dia_data.spectrum_df) / cycle_len + 1)
        )
        cycle_idx = np.repeat(
            np.arange(int(len(dia_data.spectrum_df) / cycle_len + 1)), cycle_len
        )

        dia_data.spectrum_df["delta_scan_idx"] = delta_scan_idx[
            : len(dia_data.spectrum_df)
        ]
        dia_data.spectrum_df["cycle_idx"] = cycle_idx[: len(dia_data.spectrum_df)]

        self.cycle_len = cycle_len

        self.dia_data = DIAData.from_arrays(
            dia_data.spectrum_df["delta_scan_idx"].values,
            dia_data.spectrum_df["isolation_lower_mz"].values.astype(np.float32),
            dia_data.spectrum_df["isolation_upper_mz"].values.astype(np.float32),
            dia_data.spectrum_df["peak_start_idx"].values,
            dia_data.spectrum_df["peak_stop_idx"].values,
            dia_data.spectrum_df["cycle_idx"].values,
            dia_data.spectrum_df["rt"].values.astype(np.float32) * 60,
            dia_data.peak_df["mz"].values.astype(np.float32),
            dia_data.peak_df["intensity"].values.astype(np.float32),
        )

    def prepare_spec_lib(
        self,
        precursor_df,
        fragment_df,
        rt_column,
        precursor_mz_column,
        fragment_mz_column,
        mobility_column,
    ):
        """Prepare spectral library for peak scoring."""
        self.speclib = SpecLibFlat.from_arrays(
            precursor_df["precursor_idx"].values.astype(np.uint64),
            precursor_df[precursor_mz_column].values.astype(np.float32),
            precursor_df[rt_column].values.astype(np.float32),
            precursor_df["flat_frag_start_idx"].values.astype(np.uint64),
            precursor_df["flat_frag_stop_idx"].values.astype(np.uint64),
            fragment_df[fragment_mz_column].values.astype(np.float32),
            fragment_df["intensity"].values.astype(np.float32),
            fragment_df["cardinality"].values.astype(np.uint8),
            fragment_df["charge"].values.astype(np.uint8),
            fragment_df["loss_type"].values.astype(np.uint8),
            fragment_df["number"].values.astype(np.uint8),
            fragment_df["position"].values.astype(np.uint8),
            fragment_df["type"].values.astype(np.uint8),
        )

    def parse_candidates(self, candidates):
        """Parse candidates and convert to DataFrame."""
        result = candidates.to_arrays()

        precursor_idx = result[0]
        rank = result[1]
        score = result[2]
        scan_center = result[3]
        scan_start = result[4]
        scan_stop = result[5]
        frame_center = result[6]
        frame_start = result[7]
        frame_stop = result[8]

        candidates_df = pd.DataFrame(
            {
                "precursor_idx": precursor_idx,
                "rank": rank,
                "score": score,
                "scan_center": scan_center,
                "scan_start": scan_start,
                "scan_stop": scan_stop,
                "frame_center": frame_center,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
            }
        )

        candidates_df = candidates_df.merge(
            self.precursor_df[["precursor_idx", "elution_group_idx", "decoy"]],
            on="precursor_idx",
            how="left",
        )

        candidates_df["frame_start"] = candidates_df["frame_start"] * self.cycle_len
        candidates_df["frame_stop"] = candidates_df["frame_stop"] * self.cycle_len
        candidates_df["frame_center"] = candidates_df["frame_center"] * self.cycle_len

        candidates_df["scan_start"] = 0
        candidates_df["scan_stop"] = 1
        candidates_df["scan_center"] = 0

        return candidates_df

    def __call__(self):
        """Execute peak scoring pipeline."""
        fwhm_rt = self.config_dict.get("fwhm_rt", 3.0)
        kernel_size = self.config_dict.get("kernel_size", 15)
        peak_length = self.config_dict.get("peak_length", 5)

        peak_group_selection = PeakGroupSelection(fwhm_rt, kernel_size, peak_length)

        mass_tolerance = self.config_dict["ms2_tolerance"]
        rt_tolerance = self.config_dict["rt_tolerance"]
        candidates = peak_group_selection.search(
            self.dia_data, self.speclib, mass_tolerance, rt_tolerance
        )

        return self.parse_candidates(candidates)
