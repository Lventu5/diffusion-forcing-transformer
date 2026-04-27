from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.distributed_utils import is_rank_zero
from .dfot_video import DFoTVideo


class DFoTVideoBinaryEval(DFoTVideo):
    """DFoTVideo variant with binary-conditioning validation diagnostics.

    This class is intended only for the synthetic binary conditioning test and
    keeps extra validation logging/ablations out of standard DFoT runs.
    """

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        if self.trainer.state.fn == "FIT":
            self._eval_denoising(batch, batch_idx, namespace=namespace)

        if self.trainer.sanity_checking and not self.cfg.logging.sanity_generation:
            return

        log_attn = is_rank_zero and self.logger and batch_idx == 0
        if log_attn:
            for m in self.diffusion_model.modules():
                if m.__class__.__name__ == "CrossAttnBlock":
                    m.store_attn_weights = True
                    m.first_attn_weights = None

        all_videos = self._sample_all_videos(batch, batch_idx, namespace)

        if log_attn:
            self._log_cross_attn_map(namespace)
            for m in self.diffusion_model.modules():
                if m.__class__.__name__ == "CrossAttnBlock":
                    m.store_attn_weights = False
                    m.first_attn_weights = None
                    m.last_attn_weights = None

        self._update_metrics(all_videos)
        self._update_semantic_metrics(all_videos)
        self._log_validation_prediction_stats(batch, batch_idx, all_videos, namespace)
        self._log_videos(all_videos, namespace)

    def _log_validation_prediction_stats(
        self, batch, batch_idx: int, all_videos: Dict[str, Tensor], namespace: str
    ) -> None:
        if "prediction" not in all_videos or "gt" not in all_videos:
            return

        pred = all_videos["prediction"]
        gt = all_videos["gt"]

        pred_mse = F.mse_loss(pred, gt)
        pred_mae = F.l1_loss(pred, gt)
        self.log(
            f"{namespace}/prediction_mse",
            pred_mse,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{namespace}/prediction_mae",
            pred_mae,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        _, conditions, *_ = batch
        if conditions is None or conditions.ndim < 3 or conditions.shape[-1] < 1:
            return

        frame_labels = conditions[..., -1].reshape(-1)
        is_white = frame_labels > 0.5
        is_black = ~is_white
        if not (bool(is_white.any()) and bool(is_black.any())):
            return

        frame_intensity = pred.mean(dim=(2, 3, 4)).reshape(-1)
        white_mean = frame_intensity[is_white].mean()
        black_mean = frame_intensity[is_black].mean()
        intensity_gap = white_mean - black_mean
        cond_acc = ((frame_intensity > 0.5) == is_white).float().mean()

        self.log(
            f"{namespace}/cond_white_pred_mean",
            white_mean,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{namespace}/cond_black_pred_mean",
            black_mean,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{namespace}/cond_pred_mean_gap",
            intensity_gap,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{namespace}/cond_pred_threshold_acc",
            cond_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.is_latent_diffusion:
            return
        if not bool(self.logging.get("condition_ablations", False)):
            return
        max_batches = int(self.logging.get("condition_ablation_max_batches", 1))
        if batch_idx >= max_batches:
            return

        xs, conditions, *_ = batch
        if conditions is None:
            return

        batch_size = conditions.shape[0]
        if batch_size < 1:
            return

        current_rng_state = None
        if self.generator is not None:
            current_rng_state = self.generator.get_state()

        def _predict_with(conds: Tensor, base_state: Optional[Tensor]) -> Tensor:
            if self.generator is not None and base_state is not None:
                self.generator.set_state(base_state)
            pred_local = self._predict_videos(xs, conditions=conds)
            pred_local = self._unnormalize_x(pred_local).detach()
            pred_local[:, : self.n_context_frames] = gt[:, : self.n_context_frames]
            return pred_local

        base_state = None
        if self.generator is not None:
            base_state = self.generator.get_state().clone()

        if batch_size > 1:
            shuffled_conditions = conditions.roll(shifts=1, dims=0)
            pred_shuffled = _predict_with(shuffled_conditions, base_state)
            mse_shuffled = F.mse_loss(pred_shuffled, gt)
            mae_shuffled = F.l1_loss(pred_shuffled, gt)
            self.log(
                f"{namespace}/ablation_mse_shuffled",
                mse_shuffled,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{namespace}/ablation_mae_shuffled",
                mae_shuffled,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{namespace}/ablation_mse_gap_shuffled_vs_true",
                mse_shuffled - pred_mse,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        zero_conditions = torch.zeros_like(conditions)
        pred_zero = _predict_with(zero_conditions, base_state)
        mse_zero = F.mse_loss(pred_zero, gt)
        mae_zero = F.l1_loss(pred_zero, gt)
        self.log(
            f"{namespace}/ablation_mse_zero",
            mse_zero,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{namespace}/ablation_mae_zero",
            mae_zero,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{namespace}/ablation_mse_gap_zero_vs_true",
            mse_zero - pred_mse,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        flipped_conditions = conditions.clone()
        flipped_conditions[..., -1] = 1.0 - flipped_conditions[..., -1]
        pred_flipped = _predict_with(flipped_conditions, base_state)
        flip_l1 = (pred_flipped - pred).abs().mean()
        self.log(
            f"{namespace}/ablation_counterfactual_l1",
            flip_l1,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        true_intensity = pred.mean(dim=(2, 3, 4))
        flip_intensity = pred_flipped.mean(dim=(2, 3, 4))
        label_delta = flipped_conditions[..., -1] - conditions[..., -1]
        pred_delta = flip_intensity - true_intensity
        sign_mask = label_delta != 0
        if bool(sign_mask.any()):
            direction_acc = (
                (torch.sign(pred_delta[sign_mask]) == torch.sign(label_delta[sign_mask]))
                .float()
                .mean()
            )
            self.log(
                f"{namespace}/ablation_counterfactual_direction_acc",
                direction_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if self.generator is not None and current_rng_state is not None:
            self.generator.set_state(current_rng_state)
