from model_training.cli import build_parser


def test_subcommands_present():
    p = build_parser()
    choices = p._subparsers._group_actions[0].choices.keys()
    assert {"ingest", "train-ep", "train-wp", "train-qbr", "validate", "figures"} <= set(choices)
