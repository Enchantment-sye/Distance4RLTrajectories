from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reachability_metrics.experiments.paper_reproduction import (
    DATASETS,
    TABLE2_EXPECTED,
    TABLE6_EXPECTED,
    PaperReproductionConfig,
    run_paper_reproduction,
)


LEGACY = Path("/share/shangyy/codes/metra/outputs")


def test_paper_reproduction_from_legacy_outputs(tmp_path) -> None:
    if not LEGACY.exists():
        pytest.skip("legacy METRA outputs are not available")
    result = run_paper_reproduction(
        PaperReproductionConfig(
            legacy_outputs_dir=str(LEGACY),
            output_dir=str(tmp_path),
            include_figures=False,
            verify_paper_values=True,
        )
    )
    assert result["verification_errors"] == []
    assert (tmp_path / "paper_source_manifest.json").exists()
    assert (tmp_path / "tables" / "table1_reconstruction_notes.md").exists()

    table2 = pd.read_csv(tmp_path / "tables" / "table2_relabel_spearman.csv").set_index("method")
    for method, values in TABLE2_EXPECTED.items():
        for dataset, expected in zip(DATASETS, values):
            assert table2.loc[method, dataset] == pytest.approx(expected, abs=1.5e-4)

    table6 = pd.read_csv(tmp_path / "tables" / "table6_successor_h10_auroc.csv").set_index("method")
    for method, values in TABLE6_EXPECTED.items():
        for dataset, expected in zip(DATASETS, values):
            assert table6.loc[method, dataset] == pytest.approx(expected, abs=1.5e-4)


def test_paper_reproduction_copies_figure2(tmp_path) -> None:
    if not (LEGACY / "paper_toy_pointmaze_summary_v1/figures/toy_main_figure.png").exists():
        pytest.skip("legacy paper figures are not available")
    result = run_paper_reproduction(
        PaperReproductionConfig(
            legacy_outputs_dir=str(LEGACY),
            output_dir=str(tmp_path),
            include_figures=True,
            verify_paper_values=False,
        )
    )
    assert len(result["figures"]) == 2
    for figure in result["figures"]:
        assert Path(figure).exists()
        assert Path(figure).stat().st_size > 0

