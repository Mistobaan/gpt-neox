import os
import argparse

from megatron.neox_arguments import NeoXArgs


def parse_arguments():
    group = argparse.ArgumentParser(
        description="GPT-NeoX Configuration", allow_abbrev=False
    )

    group.add_argument(
        "--conf_dir",
        "-d",
        type=str,
        default=None,
        help="Directory to prefix to all configuration file paths",
    )

    group.add_argument(
        "conf_file",
        type=str,
        nargs="+",
        help="Configuration file path. Multiple files can be provided and will be merged.",
    )

    return group.parse_args()


def main(args):
    # load config files
    conf_files = args.conf_file
    if args.conf_dir:
        conf_files = [os.path.join(args.conf_dir, f)
                      for f in conf_files]

    # enables us to pass in `small` instead of `small.yml`
    conf_files = [(cf if cf.endswith(".yml") else cf + ".yml")
                  for cf in conf_files]

    # determine overwrite values
    overwrite_values = dict()
    for k, v in vars(args).items():
        if k not in ["conf_dir", "conf_file"] and v is not None:
            overwrite_values[k] = v

    import json
    import ipdb
    ipdb.set_trace()

    # load args
    neox_args = NeoXArgs.from_ymls(
        paths_to_yml_files=conf_files, overwrite_values=overwrite_values
    )
    # args = neox_args.get_deepspeed_main_args()

    config = neox_args.get_parent_class_value_dict(
        *neox_args.__class__.__bases__, only_non_defaults=True
    )
    print(json.dumps(config, indent=None, separators=(",", ":")))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
