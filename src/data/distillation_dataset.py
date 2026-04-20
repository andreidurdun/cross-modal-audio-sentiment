from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class AudioTeacherDistillationDataset(Dataset):
    """Pairs audio-student inputs with aligned teacher embedding samples."""

    def __init__(self, student_dataset: Dataset, teacher_dataset: Dataset):
        if len(student_dataset) != len(teacher_dataset):
            raise ValueError(
                "Student and teacher datasets must have the same number of samples. "
                f"Got {len(student_dataset)} and {len(teacher_dataset)}."
            )

        self.student_dataset = student_dataset
        self.teacher_dataset = teacher_dataset

    def __len__(self) -> int:
        return len(self.student_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        student_sample = self.student_dataset[idx]
        teacher_sample = self.teacher_dataset[idx]

        student_file_id = str(student_sample.get("file_id", ""))
        teacher_file_id = str(teacher_sample.get("file_id", ""))
        if student_file_id and teacher_file_id and student_file_id != teacher_file_id:
            raise ValueError(
                "Student and teacher samples are misaligned at index "
                f"{idx}: {student_file_id} != {teacher_file_id}"
            )

        merged = dict(student_sample)
        for key, value in teacher_sample.items():
            if key in {"labels", "file_id"}:
                continue
            merged[key] = value
        if teacher_file_id and "file_id" not in merged:
            merged["file_id"] = teacher_file_id
        return merged