# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
        use_orig_params=False,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.use_orig_params = use_orig_params


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=fsdp_config.use_orig_params,
    )


# class FSDPCheckpoint:
#     @staticmethod
#     def fsdp_save_ckpt(
#         ckpt_dir, 
#         train_steps, 
#         model, 
#         ema_model, 
#         optimizer, 
#         scheduler, 
#         data_status,
#         logger, 
#         fsdp_config,
#         save_optimizer: bool = False,   # 新增：是否保存优化器
#         save_scheduler: bool = False,   # 新增：是否保存 scheduler
#     ):
#         """
#         保存 checkpoint（默认只保存 model 与 EMA，跳过 optimizer/scheduler 以提速）

#         Args:
#             ckpt_dir: 目录
#             train_steps: 步数
#             model: FSDP 模型
#             ema_model: FSDP EMA 模型（可为 None）
#             optimizer: 优化器（当 save_optimizer=True 时使用）
#             scheduler: 调度器（当 save_scheduler=True 时使用）
#             data_status: rank0 收集的 data_status（可为 None）
#             logger: 日志器
#             fsdp_config: FSDPConfig
#             save_optimizer: 是否保存优化器（默认 False）
#             save_scheduler: 是否保存调度器（默认 False）
#         """
#         save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
#         os.makedirs(save_path, exist_ok=True)
#         if logger is not None:
#             logger.info(f"Saving checkpoint to {save_path}.")

#         # ===== 保存 EMA（full consolidate 到 rank0，offload 到 CPU） =====
#         if ema_model is not None:
#             with FSDP.state_dict_type(
#                 ema_model,
#                 StateDictType.FULL_STATE_DICT,
#                 FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
#             ):
#                 ema_state_dict = ema_model.state_dict()
#                 if dist.get_rank() == 0:
#                     save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))

#         # ===== 保存模型（full consolidate 到 rank0，offload 到 CPU） =====
#         with FSDP.state_dict_type(
#             model,
#             StateDictType.FULL_STATE_DICT,
#             FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
#         ):
#             model_state_dict = model.state_dict()
#             if dist.get_rank() == 0:
#                 save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))

#         # =====（可选）保存优化器（按 shard/策略写多个小文件）=====
#         if save_optimizer:
#             with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
#                 if fsdp_config.sharding_strategy == "FULL_SHARD":
#                     shard_index = dist.get_rank()
#                     total_shards = dist.get_world_size()
#                     torch.save(
#                         optimizer.state_dict(),
#                         os.path.join(save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"),
#                     )
#                 elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
#                     shard_index = dist.get_rank() % fsdp_config.num_shard
#                     total_shards = fsdp_config.num_shard
#                     if dist.get_rank() < fsdp_config.num_shard:
#                         torch.save(
#                             optimizer.state_dict(),
#                             os.path.join(save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"),
#                         )
#                 else:
#                     raise NotImplementedError

#         # =====（可选）保存 scheduler（很小，只在 rank0 存一份）=====
#         if save_scheduler and dist.get_rank() == 0 and scheduler is not None:
#             torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

#         # ===== rank0 额外保存 data_status（小对象）=====
#         if dist.get_rank() == 0 and data_status is not None:
#             torch.save(data_status, os.path.join(save_path, "data_status.pt"))

#         dist.barrier()
#         return

#     @staticmethod
#     def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
#         if resume_from is not None and os.path.exists(resume_from):
#             if logger is not None:
#                 logger.info(f"Loading checkpoint from {resume_from}.")
#             if resume_from_ema:
#                 model_state_dict_path = os.path.join(resume_from, "ema.safetensors")
#             else:
#                 model_state_dict_path = os.path.join(resume_from, "model.safetensors")
#             model_state_dict = load_file(model_state_dict_path, device="cpu")
#             # NOTE：位置编码是固定正弦，直接丢弃以适配分辨率变化
#             model_state_dict.pop('latent_pos_embed.pos_embed', None)
#             model_state_dict.pop('vit_pos_embed.pos_embed', None)
#             msg = model.load_state_dict(model_state_dict, strict=False)
#             if logger is not None:
#                 logger.info(msg)
#             del model_state_dict

#             if ema_model is not None:
#                 ema_state_dict_path = os.path.join(resume_from, "ema.safetensors")
#                 if not os.path.exists(ema_state_dict_path):
#                     if logger is not None:
#                         logger.info(f"replicaing ema model from {model_state_dict_path}.")
#                     ema_state_dict_path = model_state_dict_path
#                 ema_state_dict = load_file(ema_state_dict_path, device="cpu")
#                 ema_state_dict.pop('latent_pos_embed.pos_embed', None)
#                 ema_state_dict.pop('vit_pos_embed.pos_embed', None)
#                 msg = ema_model.load_state_dict(ema_state_dict, strict=False)
#                 if logger is not None:
#                     logger.info(msg)
#                 del ema_state_dict
#         else:
#             if logger is not None:
#                 logger.info(f"Training from scratch.")
#         return model, ema_model

#     @staticmethod
#     def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
#         """
#         尝试加载优化器/调度器/数据进度。
#         若对应文件不存在（比如你选择不保存优化器/调度器），会自动跳过并返回 train_steps=0/data_status=None。
#         """
#         if resume_from is not None and os.path.exists(resume_from):
#             # 计算 shard 索引
#             if fsdp_config.sharding_strategy == "FULL_SHARD":
#                 shard_index = dist.get_rank()
#                 total_shards = dist.get_world_size()
#             elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
#                 shard_index = dist.get_rank() % fsdp_config.num_shard
#                 total_shards = fsdp_config.num_shard
#             else:
#                 raise NotImplementedError

#             # optimizer（可缺省）
#             opt_path = os.path.join(resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt")
#             if os.path.exists(opt_path):
#                 opt_state = torch.load(opt_path, map_location="cpu", weights_only=True)
#                 optimizer.load_state_dict(opt_state)
#                 del opt_state

#             # scheduler（可缺省）
#             sch_path = os.path.join(resume_from, "scheduler.pt")
#             if os.path.exists(sch_path):
#                 sch_state = torch.load(sch_path, weights_only=True, map_location="cpu")
#                 scheduler.load_state_dict(sch_state)
#                 del sch_state

#             # 训练步数：若没有 optimizer/scheduler，也用文件夹名推断
#             try:
#                 train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
#             except Exception:
#                 train_steps = 0

#             # data_status（可缺省）
#             data_status_path = os.path.join(resume_from, "data_status.pt")
#             if os.path.exists(data_status_path):
#                 data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
#                 local_rank = dist.get_rank()
#                 data_status = data_status[local_rank] if isinstance(data_status, (list, tuple)) and local_rank < len(data_status) else data_status
#             else:
#                 data_status = None
#         else:
#             train_steps = 0
#             data_status = None

#         return optimizer, scheduler, train_steps, data_status



class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed')
            model_state_dict.pop('vit_pos_embed.pos_embed')
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed')
                ema_state_dict.pop('vit_pos_embed.pos_embed')
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            # 确保参数形状匹配
            if ema_handle.flat_param.shape == new_handle.flat_param.shape:
                ema_params.append(ema_handle.flat_param.data)
                new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))
            else:
                # 如果形状不匹配，记录警告但继续执行
                import logging
                logging.warning(f"Parameter shape mismatch: EMA {ema_handle.flat_param.shape} vs Model {new_handle.flat_param.shape}")

    if ema_params and new_params:
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
