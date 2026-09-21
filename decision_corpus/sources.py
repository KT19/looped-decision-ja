from dataclasses import dataclass
from pathlib import Path

JGLUE_VERSION = "v1.3.0"
JGLUE_REPO = "https://github.com/yahoojapan/JGLUE.git"
JCOLA_REPO = "https://github.com/osekilab/JCoLA.git"

DEFAULT_SYNTHETIC_PATH = Path("data/synthetic_decisions/judged.jsonl")
DEFAULT_EXTERNAL_DIR = Path("data/external")
DEFAULT_OUTPUT_DIR = Path("data/decision_corpus")


@dataclass(frozen=True)
class SourcePaths:
    jglue_root: Path
    jcola_root: Path
    jnli_train: Path
    jnli_valid: Path
    jcommonsenseqa_train: Path
    jcommonsenseqa_valid: Path
    jsts_train: Path
    jsts_valid: Path
    jcola_valid_in: Path
    jcola_valid_out: Path

    @classmethod
    def from_external_dir(cls, external_dir: Path) -> "SourcePaths":
        jglue_root = external_dir / "JGLUE-v1.3.0"
        jcola_root = external_dir / "JCoLA"
        jglue_datasets = jglue_root / "datasets"
        jcola_data = jcola_root / "data" / "jcola-v1.0"

        return cls(
            jglue_root=jglue_root,
            jcola_root=jcola_root,
            jnli_train=jglue_datasets / "jnli-v1.3" / "train-v1.3.json",
            jnli_valid=jglue_datasets / "jnli-v1.3" / "valid-v1.3.json",
            jcommonsenseqa_train=jglue_datasets / "jcommonsenseqa-v1.3" / "train-v1.3.json",
            jcommonsenseqa_valid=jglue_datasets / "jcommonsenseqa-v1.3" / "valid-v1.3.json",
            jsts_train=jglue_datasets / "jsts-v1.3" / "train-v1.3.json",
            jsts_valid=jglue_datasets / "jsts-v1.3" / "valid-v1.3.json",
            jcola_valid_in=jcola_data / "in_domain_valid-v1.0.json",
            jcola_valid_out=jcola_data / "out_of_domain_valid-v1.0.json",
        )

    def dataset_files(self) -> list[Path]:
        return [
            self.jnli_train,
            self.jnli_valid,
            self.jcommonsenseqa_train,
            self.jcommonsenseqa_valid,
            self.jsts_train,
            self.jsts_valid,
            self.jcola_valid_in,
            self.jcola_valid_out,
        ]
