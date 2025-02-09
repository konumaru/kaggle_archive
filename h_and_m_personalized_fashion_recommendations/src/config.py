class Config:
    exp_num: str = "00104"
    seed: int = 42
    max_seq_len: int = 16
    # NOTE: article_embedding_size = 512にしたら過学習した
    article_embedding_size: int = 128
