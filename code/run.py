import logging
import torch
import os
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from mutils import set_logger, setup_seed
from models import BGAN, VAE, data_type, ldir, isA
import warnings
warnings.filterwarnings("ignore")
# pl.utilities.seed.seed_everything(2021)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'


def main():
    data_dir = "/home/xwm/crisper/"
    config = {
        "hidden_dim": 128,
        "lr": 0.00002,
        "beta1": 0.5,
        "batch_size": 128,
        "z_dim": 128,
        "c_dim": 1,
        "gf_dim": 16,
        "df_dim": 16,
        "class_num": 2,
    }

    # logger = TensorBoardLogger(os.path.join(data_dir, "gan_code/tblogs"), name='my_model')
    checkpoint_callback = ModelCheckpoint(
        monitor="ae/train_loss",
        dirpath=os.path.join(
            data_dir, "/home/xwm/crisper/gan_code/checkpoints" + str(data_type))
    )
    model = VAE(config, data_dir)

    trainer = pl.Trainer(
        max_epochs=17,
        gpus=1,
        # replace_sampler_ddp=False,
        # logger=logger,
        # val_percent_check=0,
        callbacks=[checkpoint_callback],
        # resume_from_checkpoint="/home/xwm/cdproject/Crispr/gan_code/checkpoints/epoch=12-step=59409.ckpt"
    )

    trainer.fit(model)
    trainer.test()


def main_gan():
    data_dir = "/home/xwm/crisper/"
    if data_type == 1:
        config = {
            "hidden_dim": 128,
            "beta1": 0.9,
            "z_dim": 128,
            "c_dim": 1,
            "df_dim": 16,
            "class_num": 2,
            "lr_ratio": 0.066145,
            "lambda1": 1.0067,
            # "lambda1": 1,
            "lambda2": 4.7769,
            # "lambda2": 1,
            "lr": 0.000014473,
            "batch_size": 128,
        }
    if data_type == 2:
        config = {
            "hidden_dim": 128,
            "beta1": 0.9,
            "z_dim": 128,
            "c_dim": 1,
            "df_dim": 16,
            "class_num": 2,
            "lr_ratio": 0.052714,
            "lambda1": 1.0475,
            # "lambda1": 1,
            "lambda2": 1.571,
            # "lambda2": 1,
            "lr": 1.986e-05,
            "batch_size": 128,
        }

    logger = TensorBoardLogger(os.path.join(
        data_dir, "gan_code/tblogs_predict_" + str(data_type)), name='my_model')
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="ptl/prc_value",
    #     dirpath=os.path.join(data_dir, "gan_code/checkpoints_predict_2")
    # )
    checkpoint_callback = ModelCheckpoint(monitor='ptl_roc_value', mode="max", save_top_k=10, dirpath=os.path.join(
        data_dir, "gan_code/checkpoints_predict_" + str(data_type)), filename='{epoch:02d}-{ptl_prc_value:.2f}-{ptl_roc_value:.2f}')
    model = BGAN(config, data_dir)

    if data_type == 2:
        resume = r"/home/xwm/result/ray_results_2/tune_asha/train_tune_1b71b_00027_27_batch_size=128,lambda1=5.8717,lambda2=1.5207,lr=0.0004356,lr_ratio=0.014875_2021-08-17_11-45-33/checkpoints/epoch=29-step=15989.ckpt"

    trainer = pl.Trainer(
        max_epochs=200,
        gpus=1,
        # replace_sampler_ddp=False,
        progress_bar_refresh_rate=0,
        logger=logger,
        # val_percent_check=0,
        callbacks=[checkpoint_callback],
        # resume_from_checkpoint=resume
    )

    trainer.fit(model)


def train_tune(config, data_dir=None, num_epochs=20, num_gpus=1):
    # config.update(ori_config)
    model = BGAN(config, data_dir)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    # "loss": "ptl/val_loss",
                    "roc_value": "ptl_roc_value",
                    "prc_value": "ptl_prc_value",
                },
                on="validation_end")
        ])
    trainer.fit(model)


def gan_tune(num_samples=250, num_epochs=30, gpus_per_trial=0.3):
    data_dir = "/home/xwm/crisper/"
    config = {
        "hidden_dim": 128,
        "beta1": 0.5,
        "z_dim": 128,
        "c_dim": 1,
        "df_dim": 16,
        "class_num": 2,
        "lr_ratio": tune.loguniform(1e-2, 1),
        "lambda1": tune.loguniform(1, 30),
        "lambda2": tune.loguniform(1, 10),
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([128, 256]),
    }
    # bayesopt = HyperOptSearch(config, metric="prc_value", mode="max")
    # re_search_alg = Repeater(bayesopt, repeat=5)
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        metric='roc_value',
        mode='max')

    reporter = CLIReporter(
        parameter_columns=["lambda1", "lr",
                           "batch_size", 'lr_ratio', 'lambda2'],
        metric_columns=["roc_value", "prc_value", "training_iteration"])
    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            data_dir=data_dir,
            num_epochs=num_epochs,
        ),
        local_dir=ldir,
        resources_per_trial={
            "cpu": 2,
            "gpu": gpus_per_trial
        },
        metric="roc_value",
        mode="max",
        config=config,
        num_samples=num_samples,
        # scheduler=scheduler,
        progress_reporter=reporter,
        # search_alg=re_search_alg,
        name="tune_asha")


if __name__ == '__main__':
    # main()
    # main_gan()
    # ray.init(num_cpus=8, num_gpus=2)
    ray.init()
    gan_tune()
