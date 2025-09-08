"""Custom input generators for ONNX export configs."""

from typing import Optional, Tuple
from transformers.processing_utils import ProcessorMixin
from PIL import Image
from optimum.utils import (
    DEFAULT_DUMMY_SHAPES,
    DTYPE_MAPPER,
    DummyTextInputGenerator,
    NormalizedTextConfig,
)


class Gemma3DummyInputGenerator(DummyTextInputGenerator):
    """Dummy input generator for Gemma3."""

    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "pixel_values",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        padding_side: str = "right",
        preprocessors: list | None = None,
        **kwargs,
    ):
        super().__init__(
            task,
            normalized_config,
            batch_size,
            sequence_length,
            num_choices,
            random_batch_size_range,
            random_sequence_length_range,
            random_num_choices_range,
            padding_side,
            **kwargs,
        )
        self.preprocessors = preprocessors or []

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        if self.task in [
            "image-text-to-text",
            "image-text-to-text-with-past",
            "feature-extraction",
            "feature-extraction-with-past",
        ]:
            image = Image.new("RGB", (896, 896), color=128)
            single_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Example"},
                    ],
                }
            ]
        elif self.task in ["text-generation", "text-generation-with-past"]:
            single_message = [
                {"role": "user", "content": [{"type": "text", "text": "Example"}]}
            ]
        else:
            message = (
                f"The task {self.task} is not supported by the {type(self).__name__}."
            )
            raise ValueError(message)

        messages = [single_message] * self.batch_size
        processor = next(
            (
                processor
                for processor in self.preprocessors
                if isinstance(processor, ProcessorMixin)
            ),
            None,
        )
        if processor is None:
            message = (
                "Gemma3 requires an AutoProcessor. Please provide `preprocessors=[AutoProcessor."
                'from_pretrained(model_name_or_path, padding_side="left")]` to this ONNX config.'
            )
            raise ValueError(message)

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors=framework,
            return_dict=True,
            add_generation_prompt=True,
            # Gemma3 uses left padding.
            padding_side="left",
        )
        if input_name not in inputs:
            message = (
                f"The requested input name '{input_name}' not found in "
                "the output from the processor.apply_chat_template."
            )
            raise ValueError(message)

        tensor = inputs[input_name]

        dtype_converter = getattr(DTYPE_MAPPER, framework)
        return tensor.to(dtype_converter(int_dtype))
