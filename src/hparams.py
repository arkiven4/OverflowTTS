r"""
hparams.py

Hyper Parameters for the experiment
"""
import os
from argparse import Namespace

import torch

from src.utilities.data import Normalise
from src.utilities.text import symbols
from src.utilities.text.cmudict import CMUDict


def create_hparams(generate_parameters=False):
    """
    Model hyperparemters
    Args:
        generate_paramters: default False
            Only used when you run data_properties.py

    Returns:
        hparams (Namespace)
    """
    data_parameters_filename = "data_parameters.pt"

    if not generate_parameters:
        if not os.path.exists(data_parameters_filename):
            raise FileNotFoundError(
                "Data Normalizing file not found! " + 'Run "python generate_data_properties.py" first'
            )

        data_properties = torch.load(data_parameters_filename)
        mean = data_properties["data_mean"].item()
        std = data_properties["data_std"].item()
        init_transition_prob = data_properties["init_transition_prob"]
        go_token_init_value = data_properties["go_token_init_value"]
        normaliser = Normalise(mean, std)
    else:
        # Must be while generating data properties
        normaliser = None
        init_transition_prob = None
        go_token_init_value = None
        mean = None
        std = None

    hparams = Namespace(
        ################################
        # Experiment Parameters        #
        ################################
        run_name="OverFlow",
        gpus=[0, 1],
        max_epochs=50000,
        val_check_interval=100,
        save_model_checkpoint=500,
        gradient_checkpoint=False,
        seed=1234,
        checkpoint_dir="checkpoints",
        tensorboard_log_dir="tb_logs",
        gradient_accumulation_steps=1,
        precision=32,
        # Placeholder to use it later while loading model
        logger=None,
        run_tests=False,
        warm_start=False,
        ignore_layers=["model.embedding.weight"],
        ################################
        # Data Parameters             #
        ################################
        batch_size=4,
        load_mel_from_disk=False,
        training_files="filelists/paper2_train_filelist.txt",
        validation_files="filelists/paper2_val_filelist.txt",
        text_cleaners=["english_cleaners", "universal_cleaners", "universal_cleaners", "japanese_cleaners"],
        phonetise=False,
        cmu_phonetiser=CMUDict("src/phonetised_files/cmudict-0.7b.txt"),
        spk_embeds_path="/run/media/viblab/Markov2/Pras/Thesis/Database/dataset_name/spk_embeds/",
        emo_embeds_path="/run/media/viblab/Markov2/Pras/Thesis/Database/dataset_name/emo_embeds/",
        database_name_index=8,
        gin_channels=128,
        emoin_channels=128,
        lin_channels=4,
        num_workers=20,
        n_lang=10,
        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        ################################
        # Data Properties              #
        ################################
        normaliser=normaliser,
        go_token_init_value=go_token_init_value,
        init_transition_probability=init_transition_prob,
        init_mean=0.0,
        init_std=1.0,
        data_mean=0,
        data_std=0,
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=508,
        ################################
        # Encoder parameters           #
        ################################
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        state_per_phone=2,
        ################################
        # HMM Parameters               #
        ################################
        n_frames_per_step=1,  # AR Order
        train_go=True,
        variance_floor=0.001,
        data_dropout=0,
        data_dropout_while_eval=True,
        data_dropout_while_sampling=False,
        predict_means=True,
        max_sampling_time=1000,
        deterministic_transition=True,
        duration_quantile_threshold=0.5,
        ################################
        # Prenet parameters            #
        ################################
        prenet_n_layers=3,
        prenet_dim=256,
        prenet_dropout=0.5,
        prenet_dropout_while_eval=True,
        ################################
        # Decoder RNN parameters       #
        ################################
        post_prenet_rnn_dim=1024,
        ################################
        # Decoder Parameters           #
        ################################
        parameternetwork=[1024],
        ################################
        # Decoder Flow Parameters      #
        ################################
        flow_hidden_channels=150,
        kernel_size_dec=5,
        dilation_rate=1,
        n_blocks_dec=12,
        n_block_layers=4,
        p_dropout_dec=0.05,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate=5e-4,
        weight_decay=1e-6,
        grad_clip_thresh=40000.0,
        stochastic_weight_avg=False,
    )

    return hparams
