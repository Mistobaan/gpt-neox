"""
plausibility check for the usage of neox_args in the megatron codebase
"""
import pytest
import re
from ..common import get_root_directory


@pytest.mark.cpu
def test_neoxargs_usage():
    """ "
    checks for code pieces of the pattern "args.*" and verifies that such used arg is defined in NeoXArgs
    """
    from megatron.neox_arguments import NeoXArgs

    declared_all = True
    neox_args_attributes = set(NeoXArgs.__dataclass_fields__.keys())

    # we exlude a number of properties (implemented with the @property decorator) or functions that we know exists
    exclude = set(
        [
            "params_dtype",
            "deepspeed_config",
            "get",
            "pop",
            "get_deepspeed_main_args",
            'optimizer["params"]',
            "attention_config[layer_number]",
            "adlr_autoresume_object",
            "update_value",
            "all_config",
            "tensorboard_writer",
            "tokenizer",
            "train_batch_size]",
            "items",
            "configure_distributed_args",
            "build_tokenizer",
            "attention_config[i]",
            "print",
        ]
    )

    # test file by file
    for filename in (get_root_directory() / "megatron").glob("**/*.py"):
        if filename.name in ["text_generation_utils.py", "train_tokenizer.py"]:
            continue

        # load file
        with open(filename, "r") as f:
            file_contents = f.read()

        # find args matches
        matches = list(
            re.findall(r"(?<=args\.).{2,}?(?=[\s\n(){}+-/*;:,=])", file_contents)
        )
        if len(matches) == 0:
            continue

        # compare
        for match in matches:
            if match not in neox_args_attributes and match not in exclude:
                print(
                    f"(arguments used not found in neox args): {filename.name}: {match}",
                    flush=True,
                )
                declared_all = False

    assert declared_all, "all arguments used in code defined in NeoXArgs"
