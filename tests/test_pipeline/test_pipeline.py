import pytest
import pandas as pd
import numpy as np
from mgait.pipeline.multimobility_pipeline import MultimobilityPipeline
from mgait.CAD.cad import Cadence
from mgait.GSD.GSD2 import HickeyGSD
from mgait.ICD.ICD2 import McCamleyIC
from mgait.SL.SL1 import WeinbergSL
from mgait.WS.walking_speed import Ws
from mgait.pipeline.utils._stride_filtering import StrideFiltering
from mgait.pipeline.utils._wb_assembly import WbAssembly
from mgait.aggregation._generic_aggregator import GenericAggregator
from mgait.pipeline.utils._thresholds import get_thresholds

# Minimal example GaitDatasetT fixture
@pytest.fixture
def example_datapoint():
    class DummyDataset:
        participant_metadata = {"height_m": 1.75}
        recording_metadata = {"device": "wrist"}
        sampling_rate_hz = 100.0  # typical for wrist IMU
        group_label = "test"

        # Generate a larger signal (e.g., 50 samples)
        n_samples = 50
        t = np.linspace(0, 1, n_samples)

        # Example signals: simple sinusoids + gravity offset
        data_ss = pd.DataFrame({
            "acc_is": 0.2 * np.sin(2 * np.pi * 1 * t) + 0.0,   # IS axis
            "acc_ml": 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.0, # ML axis
            "acc_pa": 9.8 + 0.1 * np.sin(2 * np.pi * 1.2 * t) # PA axis (gravity included)
        })

    return DummyDataset()


# Predefined algorithm instances for tests
@pytest.fixture
def example_pipeline_algorithms():
    return dict(
        gait_sequence_detection=HickeyGSD(),  # <-- Use the alternative GSD here
        initial_contact_detection=McCamleyIC(),
        cadence_calculation=Cadence(),
        stride_length_calculation=WeinbergSL(),
        walking_speed_calculation=Ws(),
        stride_selection=StrideFiltering(),
        wba=WbAssembly(),
        dmo_thresholds=get_thresholds(),
        dmo_aggregation=GenericAggregator(**GenericAggregator.PredefinedParameters.single_day),
    )


class TestFullMultimobilityPipeline:
    def test_full_pipeline_run(self, example_datapoint, example_pipeline_algorithms):
        pipeline = MultimobilityPipeline(**example_pipeline_algorithms)
        result = pipeline.run(example_datapoint)

        # Basic assertions on output structure
        assert hasattr(result, "per_stride_parameters_")
        assert hasattr(result, "per_wb_parameters_")
        assert hasattr(result, "aggregated_parameters_")

        # Optional: test that per-WB table has expected columns
        expected_cols = [
            "stride_duration_s", "cadence_spm", "stride_length_m", "walking_speed_mps"
        ]
        for col in expected_cols:
            assert col in result.per_wb_parameters_.columns
