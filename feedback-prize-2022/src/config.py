import torch


class DeBERTaBaseConfig:
    seed: int = 42
    exp_name: str = "deberta-base"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-base"
    max_seq_len: int = 512
    num_epoch: int = 5
    batch_size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaBase256Config:
    seed: int = 42
    exp_name: str = "deberta-base-256"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-base"
    max_seq_len: int = 256
    num_epoch: int = 5
    batch_size = 24

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaBase768Config:
    seed: int = 42
    exp_name: str = "deberta-base-768"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-base"
    max_seq_len: int = 768
    num_epoch: int = 5
    batch_size = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaLargeConfig:
    seed: int = 42
    exp_name: str = "deberta-large"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-large"
    max_seq_len: int = 512
    num_epoch: int = 5
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaLarge256Config:
    seed: int = 42
    exp_name: str = "deberta-large-256"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-large"
    max_seq_len: int = 256
    num_epoch: int = 5
    batch_size = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaV3BaseConfig:
    seed: int = 42
    exp_name: str = "deberta-v3-base"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-v3-base"
    max_seq_len: int = 512
    num_epoch: int = 5
    batch_size = 24

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaV3Base256Config:
    seed: int = 42
    exp_name: str = "deberta-v3-base-256"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-v3-base"
    max_seq_len: int = 256
    num_epoch: int = 5
    batch_size = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaV3Base768Config:
    seed: int = 42
    exp_name: str = "deberta-v3-base-768"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-v3-base"
    max_seq_len: int = 768
    num_epoch: int = 5
    batch_size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaV3LargeConfig:
    seed: int = 42
    exp_name: str = "deberta-v3-large"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-v3-large"
    max_seq_len: int = 512
    num_epoch: int = 5
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaV3Large256Config:
    seed: int = 42
    exp_name: str = "deberta-v3-large-256"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "microsoft/deberta-v3-large"
    max_seq_len: int = 256
    num_epoch: int = 5
    batch_size = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoBERTaBaseConfig:
    seed: int = 42
    exp_name: str = "roberta-base"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "roberta-base"
    max_seq_len: int = 512
    num_epoch: int = 5
    batch_size = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoBERTaBase256Config:
    seed: int = 42
    exp_name: str = "roberta-base-256"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "roberta-base"
    max_seq_len: int = 256
    num_epoch: int = 5
    batch_size = 24

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoBERTaBase768Config:
    seed: int = 42
    exp_name: str = "roberta-base-768"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "roberta-base"
    max_seq_len: int = 768
    num_epoch: int = 5
    batch_size = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoBERTaLargeConfig:
    seed: int = 42
    exp_name: str = "roberta-large"

    num_fold: int = 4
    num_seed: int = 3

    model_path: str = "roberta-large"
    max_seq_len: int = 512
    num_epoch: int = 5
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
