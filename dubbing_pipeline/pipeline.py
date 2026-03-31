"""Pipeline coordinator and runtime context for the dubbing workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

from .models import JobManifest, StageName, StageStatus
from .state import load_manifest, save_manifest, stage_lookup, update_stage_status


STAGE_DIRECTORIES = {
    StageName.SOURCE_SEPARATION: "01_source",
    StageName.TRANSCRIPTION: "02_transcript",
    StageName.TRANSCRIPT_REVIEW: "02_transcript_review",
    StageName.TRANSLATION: "03_translation",
    StageName.TRANSLATION_REVIEW: "03_translation_review",
    StageName.VOICE_PREP: "04_voice_prep",
    StageName.SYNTHESIS: "04_synthesis",
    StageName.ALIGNMENT: "05_alignment",
    StageName.FINAL_MIX: "06_mix",
}


@dataclass
class StageResult:
    """Outcome returned by a pipeline stage."""

    stage_name: StageName
    status: StageStatus
    message: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Runtime context shared across stage modules."""

    manifest: JobManifest
    manifest_path: Path
    api_keys: Dict[str, str]

    def save(self) -> Path:
        return save_manifest(self.manifest)

    @property
    def job_dir(self) -> Path:
        return self.manifest.job_dir

    @property
    def settings(self):
        return self.manifest.settings

    @property
    def source_media(self):
        return self.manifest.source_media

    def stage_dir(self, stage_name: StageName) -> Path:
        directory_name = STAGE_DIRECTORIES[stage_name]
        path = self.job_dir / directory_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_status(self, stage_name: StageName) -> StageStatus:
        return stage_lookup(self.manifest)[stage_name].status

    def stage_artifacts(self, stage_name: StageName) -> Dict[str, str]:
        return stage_lookup(self.manifest)[stage_name].artifacts

    def stage_artifact_path(self, stage_name: StageName, key: str) -> Optional[Path]:
        value = self.stage_artifacts(stage_name).get(key)
        return Path(value) if value else None

    def set_stage_result(self, result: StageResult) -> None:
        self.manifest = update_stage_status(
            self.manifest,
            result.stage_name,
            result.status,
            result.message,
            result.artifacts,
        )
        self.save()


def build_context(job_dir: Path, api_keys: Dict[str, str]) -> PipelineContext:
    manifest = load_manifest(job_dir)
    return PipelineContext(
        manifest=manifest,
        manifest_path=job_dir / "manifest.json",
        api_keys=api_keys,
    )


class PipelineRunner:
    """Runs the configured stages in order and persists manifest state."""

    def __init__(self, context: PipelineContext, stages: Sequence["PipelineStage"]) -> None:
        self.context = context
        self.stages = list(stages)

    def run(self, *, resume: bool = True) -> JobManifest:
        for stage in self.stages:
            if resume and self.context.stage_status(stage.stage_name) == StageStatus.COMPLETED:
                continue

            self.context.manifest = update_stage_status(
                self.context.manifest,
                stage.stage_name,
                StageStatus.IN_PROGRESS,
                "Running stage.",
                self.context.stage_artifacts(stage.stage_name),
            )
            self.context.save()

            result = stage.run(self.context)
            self.context.set_stage_result(result)

            if result.status in {StageStatus.NEEDS_REVIEW, StageStatus.FAILED}:
                break

        return self.context.manifest


def load_runner(job_dir: Path, api_keys: Dict[str, str]) -> PipelineRunner:
    from .stages import build_default_stages

    return PipelineRunner(build_context(job_dir, api_keys), build_default_stages())
