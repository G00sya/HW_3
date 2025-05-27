config = dict(
    d_ff=1024,
    blocks_count=4,  # in Encoder and Decoder
    heads_count=10,  # Because we use d_model=300 in navec, and it must be divisible by heads_count
    dropout_rate=0.1,
    epochs=30,
    train_batch_size=16,
    test_batch_size=32,
    data_split_ratio=0.99,
    use_pretrained_embedding=True,
    d_model=300,  # in case of don't using pretrained embedding
)
