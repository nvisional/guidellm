import json
from pathlib import Path

import pytest
import yaml

from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.outputs.serialized import (
    GenerativeBenchmarkerSerialized,
)
from guidellm.benchmark.schemas import (
    GenerativeBenchmarksReport,
)
from guidellm.schemas.benchmark import (
    BenchmarkScenario,
    JSONBenchmarkOutputArgs,
    YAMLBenchmarkOutputArgs,
)


@pytest.fixture
def minimal_report() -> GenerativeBenchmarksReport:
    """
    Create a GenerativeBenchmarksReport for testing serialization.
    """
    config = BenchmarkScenario.model_validate(
        {
            "spec": {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000/v1",
                    "model": "test-model",
                },
                "data": [{"kind": "huggingface", "source": "test_data.jsonl"}],
                "profile": {"kind": "constant", "rate": 10.0},
            },
        }
    )
    return GenerativeBenchmarksReport(config=config)


@pytest.fixture
def file_path_report(tmp_path: Path) -> GenerativeBenchmarksReport:
    """
    Create a report whose benchmark args contain nested Path values.

    ## WRITTEN BY AI ##
    """
    args = BenchmarkScenario.model_validate(
        {
            "spec": {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000/v1",
                    "model": "test-model",
                },
                "profile": {"kind": "sweep"},
                "data": [
                    {
                        "kind": "json_file",
                        "path": tmp_path / "data.jsonl",
                        "load_kwargs": {"split": "train"},
                    }
                ],
                "tokenizer": {"kind": "huggingface_auto", "model": "test-model"},
                "data_column_mapper": {"kind": "generative_column_mapper"},
                "data_preprocessors": [],
                "data_finalizer": {"kind": "generative"},
                "data_loader": {"kind": "pytorch"},
            },
        }
    )
    return GenerativeBenchmarksReport(config=args)


class TestResolveFormatType:
    """
    Tests that resolve() propagates the format key correctly.

    ## WRITTEN BY AI ##
    """

    def test_resolve_yaml_alias_sets_format_type(self, tmp_path: Path):
        """
        Resolving with YAMLBenchmarkOutputArgs should produce a handler
        with format_type='yaml'.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerOutput.resolve(
            YAMLBenchmarkOutputArgs(path=tmp_path / "benchmarks.yaml")
        )
        assert isinstance(handler, GenerativeBenchmarkerSerialized)
        assert handler.format_type == "yaml"

    def test_resolve_json_alias_sets_format_type(self, tmp_path: Path):
        """
        Resolving with JSONBenchmarkOutputArgs should produce a handler
        with format_type='json'.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerOutput.resolve(
            JSONBenchmarkOutputArgs(path=tmp_path / "benchmarks.json")
        )
        assert isinstance(handler, GenerativeBenchmarkerSerialized)
        assert handler.format_type == "json"

    def test_resolve_explicit_yaml_filename(self, tmp_path: Path):
        """
        Resolving with an explicit path should produce a handler
        whose output_path matches.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerOutput.resolve(
            YAMLBenchmarkOutputArgs(path=tmp_path / "results.yaml")
        )
        assert isinstance(handler, GenerativeBenchmarkerSerialized)
        assert handler.output_path == tmp_path / "results.yaml"


class TestFinalizeFormats:
    """
    Tests that finalize() writes the correct file format.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.asyncio
    async def test_finalize_yaml_writes_yaml_file(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        A handler with format_type='yaml' should write a .yaml file
        containing valid YAML content.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="yaml"
        )
        result_path = await handler.finalize(minimal_report)

        assert result_path.suffix == ".yaml"
        assert result_path.name == "benchmarks.yaml"
        assert result_path.exists()

        content = result_path.read_text()
        # JSON is valid YAML, so verify the output is not JSON
        # to confirm YAML was actually written
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert "config" in parsed

    @pytest.mark.asyncio
    async def test_finalize_json_writes_json_file(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        A handler with format_type='json' should write a .json file
        containing valid JSON content.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="json"
        )
        result_path = await handler.finalize(minimal_report)

        assert result_path.suffix == ".json"
        assert result_path.name == "benchmarks.json"
        assert result_path.exists()

        content = result_path.read_text()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
        assert "config" in parsed

    @pytest.mark.asyncio
    async def test_finalize_json_serializes_nested_paths(
        self, tmp_path: Path, file_path_report: GenerativeBenchmarksReport
    ):
        """
        JSON report serialization should convert nested Path values to strings.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="json"
        )
        result_path = await handler.finalize(file_path_report)

        parsed = json.loads(result_path.read_text())
        assert parsed["config"]["spec"]["data"][0]["path"] == str(
            tmp_path / "data.jsonl"
        )

    @pytest.mark.asyncio
    async def test_finalize_yaml_serializes_nested_paths(
        self, tmp_path: Path, file_path_report: GenerativeBenchmarksReport
    ):
        """
        YAML report serialization should emit nested Path values as safe strings.

        ## WRITTEN BY AI ##
        """
        handler = GenerativeBenchmarkerSerialized(
            output_path=tmp_path, format_type="yaml"
        )
        result_path = await handler.finalize(file_path_report)

        parsed = yaml.safe_load(result_path.read_text())
        assert parsed["config"]["spec"]["data"][0]["path"] == str(
            tmp_path / "data.jsonl"
        )

    @pytest.mark.asyncio
    async def test_finalize_explicit_path_respects_extension(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        When output_path is an explicit file (not a directory), finalize
        should use that path directly regardless of format_type.

        ## WRITTEN BY AI ##
        """
        explicit_path = tmp_path / "my_report.yaml"
        handler = GenerativeBenchmarkerSerialized(output_path=explicit_path)
        result_path = await handler.finalize(minimal_report)

        assert result_path == explicit_path
        assert result_path.exists()

        content = result_path.read_text()
        # JSON is valid YAML, so verify the output is not JSON
        # to confirm YAML was actually written
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert "config" in parsed

    @pytest.mark.asyncio
    async def test_finalize_extensionless_path_uses_format_type_json(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        kind=json with a path that has no .json/.yaml/.yml suffix should still
        write JSON to that path instead of raising after the run (#1286).

        ## WRITTEN BY AI ##
        """
        explicit_path = tmp_path / "results"
        handler = GenerativeBenchmarkerSerialized(
            output_path=explicit_path, format_type="json"
        )
        result_path = await handler.finalize(minimal_report)

        assert result_path == explicit_path
        assert result_path.exists()
        parsed = json.loads(result_path.read_text())
        assert isinstance(parsed, dict)
        assert "config" in parsed

    @pytest.mark.asyncio
    async def test_finalize_extensionless_path_uses_format_type_yaml(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        kind=yaml with an unrecognized suffix should write YAML to that path.

        ## WRITTEN BY AI ##
        """
        explicit_path = tmp_path / "results.out"
        handler = GenerativeBenchmarkerSerialized(
            output_path=explicit_path, format_type="yaml"
        )
        result_path = await handler.finalize(minimal_report)

        assert result_path == explicit_path
        content = result_path.read_text()
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)
        assert "config" in parsed

    @pytest.mark.asyncio
    async def test_finalize_nonexistent_dir_path_uses_format_type(
        self, tmp_path: Path, minimal_report: GenerativeBenchmarksReport
    ):
        """
        A trailing-slash dir that does not exist yet becomes a suffix-less path;
        use format_type so the run still writes output (#1286).

        ## WRITTEN BY AI ##
        """
        # Mimic pathlib dropping a trailing slash on a not-yet-created directory.
        explicit_path = tmp_path / "new_results"
        assert not explicit_path.exists()
        handler = GenerativeBenchmarkerSerialized(
            output_path=explicit_path, format_type="json"
        )
        result_path = await handler.finalize(minimal_report)

        assert result_path == explicit_path
        assert json.loads(result_path.read_text())["config"]



@pytest.mark.regression
def test_report_round_trips_profile_conclusions(minimal_report):
    """
    Persist and restore what each profile concluded after its run.

    ## WRITTEN BY AI ##
    """
    report = minimal_report
    report.conclusions = [
        {"best_passing_streams": 26, "stop_reason": "converged"},
        {"best_passing_streams": 12, "stop_reason": "max_probes_exhausted"},
    ]
    restored = GenerativeBenchmarksReport.model_validate(report.model_dump())

    assert restored.conclusions == [
        {"best_passing_streams": 26, "stop_reason": "converged"},
        {"best_passing_streams": 12, "stop_reason": "max_probes_exhausted"},
    ]


@pytest.mark.regression
def test_report_without_conclusions_still_validates(minimal_report):
    """
    Restore a report written before the conclusions field existed.

    ## WRITTEN BY AI ##
    """
    payload = minimal_report.model_dump()
    del payload["conclusions"]

    assert GenerativeBenchmarksReport.model_validate(payload).conclusions == []
