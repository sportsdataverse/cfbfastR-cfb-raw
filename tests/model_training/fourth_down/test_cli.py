from model_training.fourth_down.cli import build_parser


def test_train_fd_subcommand_present():
    p = build_parser()
    choices = p._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    assert "train-fd" in choices


def test_train_fd_accepts_final_dir_and_out():
    p = build_parser()
    args = p.parse_args(["train-fd", "--final-dir", "/tmp/final", "--out", "/tmp/fd.ubj"])
    assert args.final_dir == "/tmp/final"
    assert args.out == "/tmp/fd.ubj"
    assert args.seasons is None


def test_train_fd_accepts_seasons():
    p = build_parser()
    args = p.parse_args(
        ["train-fd", "--final-dir", "cfb/json/final", "--out", "fd.ubj", "--seasons", "2018", "2019", "2020"]
    )
    assert args.seasons == [2018, 2019, 2020]
