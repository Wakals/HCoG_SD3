import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.gpt.PE import ImageEvaluate, get_chain
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from torch.cuda.amp import autocast
from torch.nn import functional as F

from ..geometry.gaussian_base import BasicPointCloud, Camera

from lang_sam import LangSAM
from PIL import Image
    

@threestudio.register("hcog-system")
class GaussianSplatting(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        
        segment_threshold: float = 0.95
        
        init_steps: int = 300
        expand_steps: int = 900
        final_optim_steps: int = 1200
        
        seg_steps: int = 100
        optim_steps: int = 400
        

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False
        
        chain = get_chain(self.cfg.prompt_processor.prompt)
        self.original_prompt = self.cfg.prompt_processor.prompt
        self.text_prompts = chain["sub_prompts"]
        self.instances = chain["instances"]
        self.stratification_order = chain["stratification_order"]
        self.body = chain["body"]
        
        self.layer = 0
        self.order_idx = 0
        self.seg_num = 0
        
        self.stage_num = 0
        self.cur_step = 0
        
        self.init_steps = self.cfg.init_steps
        self.expand_steps = self.cfg.expand_steps
        self.final_optim_steps = self.cfg.final_optim_steps
        
        self.seg_steps = self.cfg.seg_steps
        self.optim_steps = self.cfg.optim_steps
        
        self.segment_threshold = self.cfg.segment_threshold

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        
        new_cfg = self.cfg.prompt_processor
        new_cfg.prompt = self.text_prompts[0]
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(new_cfg)
        self.prompt_utils = self.prompt_processor()
        
        self.fixed_noise = None

        self.sam = LangSAM()

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()
        
    def training_step(self, batch, batch_idx):
        # stage 0: original training
        # stage 1: sam segmentation
        # stage 2: part optimization
        # stage 3: extend
        # stage 4: segment for Label Elimination
        # stage 5: Label Elimination
        # stage 6: final optimization
        if self.stage_num == 0:
            self.cur_step += 1
            if self.cur_step == self.init_steps:
                self.cur_step = 0
                self.stage_num = 1
            return self.training_step_original(batch, batch_idx)
        
        if self.stage_num == 1:
            if self.cur_step == 0:
                self.geometry.reset_segment_p()
                
                out = self(batch)
                image = out["comp_rgb"].detach().cpu().clone()
                image = image
                img = (image[0] * 255).clamp(0, 255).byte()
                np_image = img.numpy()
                image_pil = Image.fromarray(np_image)
                check_pth = f'./check/check_attribute.png'
                import os
                if not os.path.exists('./check'):
                    os.makedirs('./check')
                image_pil.save(check_pth)
                cur_order = self.stratification_order[self.order_idx]
                while True:
                    if self.order_idx >= len(self.stratification_order):
                        self.cur_step = 0
                        self.stage_num = 6
                        return self.training_step(batch, batch_idx)
                    cur_order = self.stratification_order[self.order_idx]
                    if cur_order == "EXTEND":
                        self.cur_step = 0
                        self.stage_num = 3
                        self.order_idx += 1
                        self.layer += 1
                        return self.training_step(batch, batch_idx)
                    for instance in self.instances:
                        attribute = self.instances[instance]
                        if instance in cur_order.split(" "):
                            attr = attribute
                            part = instance
                            break
                    print(f"the attr is {attr}, the part is {part}")
                    res = ImageEvaluate(attr, part, check_pth)
                    if res[0] == "y" or res[0] == "Y":
                        self.order_idx += 1
                    else:
                        break
                
                # select every gaussian kernels
                self.geometry.selected_gaussians = torch.ones(self.geometry._xyz.shape[0], dtype=torch.bool)
                self.geometry.selected_gaussians = self.geometry.selected_gaussians.to(self.geometry._xyz.device)
                def hook_fn(grad):
                    mask_shape = [grad.shape[0]] + [1] * (grad.dim() - 1)
                    mask = self.geometry.selected_gaussians.to(grad.dtype).view(mask_shape)  
                    return grad * mask
                self.geometry._xyz.register_hook(hook_fn)
                self.geometry._features_dc.register_hook(hook_fn)
                self.geometry._features_rest.register_hook(hook_fn)
                self.geometry._scaling.register_hook(hook_fn)
                self.geometry._rotation.register_hook(hook_fn)
                self.geometry._opacity.register_hook(hook_fn)
                self.geometry._segment_p.register_hook(hook_fn)
                
            self.cur_step += 1
            cur_order = self.stratification_order[self.order_idx]
            for instance in self.instances:
                attribute = self.instances[instance]
                if instance in cur_order.split(" "):
                    attr = attribute
                    part = instance
                    break
            if self.cur_step == self.seg_steps:
                self.cur_step = 0
                self.stage_num = 2
            return self.training_step_sam(batch, batch_idx, part)
        
        if self.stage_num == 2:
            if self.cur_step == 0:
                self.geometry.selected_gaussians = (self.geometry.get_segment_p > self.segment_threshold).to(torch.bool)
                self.geometry.selected_gaussians = self.geometry.selected_gaussians[:, 0]
                
                def hook_fn(grad):
                    mask_shape = [grad.shape[0]] + [1] * (grad.dim() - 1)
                    mask = self.geometry.selected_gaussians.to(grad.dtype).view(mask_shape)  
                    return grad * mask
                
                self.geometry._xyz.register_hook(hook_fn)
                self.geometry._features_dc.register_hook(hook_fn)
                self.geometry._features_rest.register_hook(hook_fn)
                self.geometry._scaling.register_hook(hook_fn)
                self.geometry._rotation.register_hook(hook_fn)
                self.geometry._opacity.register_hook(hook_fn)
                self.geometry._segment_p.register_hook(hook_fn)
                
                new_cfg = self.cfg.prompt_processor
                new_cfg.prompt = self.stratification_order[self.order_idx]
                self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(new_cfg)
                self.prompt_utils = self.prompt_processor()
                
            self.cur_step += 1
            if self.cur_step == self.optim_steps:
                self.order_idx += 1
                if self.order_idx >= len(self.stratification_order):
                    self.cur_step = 0
                    self.stage_num = 6
                    return self.training_step(batch, batch_idx)
                cur_order = self.stratification_order[self.order_idx]
                if cur_order == "EXTEND":
                    self.order_idx += 1
                    self.layer += 1
                    self.cur_step = 0
                    self.stage_num = 3
                    return self.training_step(batch, batch_idx)
                self.cur_step = 0
                self.stage_num = 1
            return self.training_step_sam_2(batch, batch_idx)
        
        if self.stage_num == 3:
            if self.cur_step == 0:
                init_num_points = self.geometry._xyz.shape[0]
                # print(f'init_num_points: {init_num_points}')
                self.geometry.densify_double()
                self.geometry.selected_gaussians = torch.cat([torch.zeros(init_num_points, dtype=torch.bool), torch.ones(self.geometry._xyz.shape[0] - init_num_points, dtype=torch.bool)])
                self.geometry.selected_gaussians = self.geometry.selected_gaussians.to(self.geometry._xyz.device)
                
                def hook_fn(grad):
                    mask_shape = [grad.shape[0]] + [1] * (grad.dim() - 1)
                    mask = self.geometry.selected_gaussians.to(grad.dtype).view(mask_shape)  
                    return grad * mask 
                    
                self.geometry._xyz.register_hook(hook_fn)
                self.geometry._features_dc.register_hook(hook_fn)
                self.geometry._features_rest.register_hook(hook_fn)
                self.geometry._scaling.register_hook(hook_fn)
                self.geometry._rotation.register_hook(hook_fn)
                self.geometry._opacity.register_hook(hook_fn)
                self.geometry._segment_p.register_hook(hook_fn)
                
                new_cfg = self.cfg.prompt_processor
                new_cfg.prompt = self.text_prompts[self.layer]
                self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(new_cfg)
                self.prompt_utils = self.prompt_processor()
                
            self.cur_step += 1
            if self.cur_step == self.expand_steps:
                self.cur_step = 0
                self.stage_num = 4
            return self.training_step_original(batch, batch_idx)
        
        if self.stage_num == 4:
            if self.cur_step == 0:
                self.geometry.reset_segment_p()
            self.cur_step += 1
            cur_order_idx = self.order_idx
            orders = []
            while cur_order_idx < len(self.stratification_order) and self.stratification_order[cur_order_idx] != "EXTEND":
                for instance in self.instances:
                    attribute = self.instances[instance]
                    if instance in self.stratification_order[cur_order_idx].split(" "):
                        attr = attribute
                        part = instance
                        break
                orders.append(part)
                cur_order_idx += 1
            # random choose one order
            assert len(orders) > 0
            cur_order = orders[np.random.randint(0, len(orders))]
            if self.cur_step == self.seg_steps:
                self.cur_step = 0
                self.stage_num = 5
            return self.training_step_sam(batch, batch_idx, cur_order)
        
        if self.stage_num == 5:
            origin_labels = ~self.geometry.selected_gaussians
            seg_labels = (self.geometry.get_segment_p > self.segment_threshold).to(torch.bool)[:, 0]
            # align origin_labels and seg_labels
            dif_len = seg_labels.shape[0] - origin_labels.shape[0]
            if dif_len > 0:
                origin_labels = torch.cat([origin_labels, torch.zeros(dif_len, dtype=torch.bool, device=origin_labels.device)])
            elif dif_len < 0:
                seg_labels = torch.cat([seg_labels, torch.zeros(-dif_len, dtype=torch.bool, device=seg_labels.device)])
            labels = origin_labels | seg_labels
            self.geometry.prune_points(~labels)
            
            self.cur_step = 0
            self.stage_num = 1
            return self.training_step(batch, batch_idx)
        
        if self.stage_num == 6:
            if self.cur_step == 0:
                new_cfg = self.cfg.prompt_processor
                new_cfg.prompt = self.original_prompt
                self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(new_cfg)
                self.prompt_utils = self.prompt_processor()
                
                # select every gaussian kernels
                self.geometry.selected_gaussians = torch.ones(self.geometry._xyz.shape[0], dtype=torch.bool)
                self.geometry.selected_gaussians = self.geometry.selected_gaussians.to(self.geometry._xyz.device)
                def hook_fn(grad):
                    mask_shape = [grad.shape[0]] + [1] * (grad.dim() - 1)
                    mask = self.geometry.selected_gaussians.to(grad.dtype).view(mask_shape)  
                    return grad * mask
                self.geometry._xyz.register_hook(hook_fn)
                self.geometry._features_dc.register_hook(hook_fn)
                self.geometry._features_rest.register_hook(hook_fn)
                self.geometry._scaling.register_hook(hook_fn)
                self.geometry._rotation.register_hook(hook_fn)
                self.geometry._opacity.register_hook(hook_fn)
                self.geometry._segment_p.register_hook(hook_fn)
                
            self.cur_step += 1
                
            return self.training_step_original(batch, batch_idx)
            

    def training_step_original(self, batch, batch_idx):
        opt = self.optimizers()
        out = self(batch)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        # import pdb; pdb.set_trace()
        viewspace_point_tensor = out["viewspace_points"]
        guidance_out = self.guidance(
            guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False, noise=self.fixed_noise
        )

        loss_sds = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_sds += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        xyz_mean = None
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_scales"] > 0.0:
            scale_sum = torch.sum(self.geometry.get_scaling)
            self.log(f"train/scales", scale_sum)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                out["comp_rgb"].permute(0, 3, 1, 2)
            )
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        ):
            loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
                + tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            )
            self.log(f"train/loss_depth_tv", loss_depth_tv)
            loss += loss_depth_tv

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_sds.backward(retain_graph=True)
        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        if loss > 0:
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss_sds}

    def training_step_sam(self, batch, batch_idx, cur_order):
        opt = self.optimizers()
        
        out = self(batch)
        rendered_segment = out["segment"]
        
        image = out["comp_rgb"].detach().cpu().clone()
        image = image
        img = (image[0] * 255).clamp(0, 255).byte()
        np_image = img.numpy()
        image_pil = Image.fromarray(np_image)
        seg_text_prompt = cur_order
        results = self.sam.predict([image_pil], [seg_text_prompt])
        masks = results[0]["masks"]
        masks = torch.Tensor(masks)
        if len(masks.shape) > 1:
            mask = masks[0].float().to("cuda") 
            mask = mask.clamp(0, 1)
            
            # save mask as grey png image
            if self.global_step % 50 == 0:
                mask_img = (mask * 255).clamp(0, 255).byte()
                np_mask = mask_img.cpu().numpy()
                mask_pil = Image.fromarray(np_mask)
                check_pth = f'./check/check_mask_{self.global_step}.png'
                mask_pil.save(check_pth)

            segment_img = rendered_segment[0][:, :, 0]
            
            # save segment_img as grey png image
            if self.global_step % 50 == 0:
                segment_img_ = (segment_img * 255).clamp(0, 255).byte()
                np_segment = segment_img_.cpu().numpy()
                segment_pil = Image.fromarray(np_segment)
                check_pth = f'./check/check_segment_{self.global_step}.png'
                segment_pil.save(check_pth)
            
            segment_loss = F.binary_cross_entropy_with_logits(segment_img, mask)

            self.log("train/segment_loss", segment_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
            # self.manual_backward(segment_loss)
            segment_loss.backward(retain_graph=True)
            opt.step()
            opt.zero_grad(set_to_none=True)
            # print the lr of the optimizer
            # for param_group in opt.param_groups:
            #     print(f"the lr is {param_group['lr']}")
            
            
    def training_step_sam_2(self, batch, batch_idx):
        
        opt = self.optimizers()
        out = self(batch)
        
        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        # import pdb; pdb.set_trace()
        viewspace_point_tensor = out["viewspace_points"]
        guidance_out = self.guidance(
            guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False, noise=self.fixed_noise
        )

        loss_sds = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_sds += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        xyz_mean = None
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_scales"] > 0.0:
            scale_sum = torch.sum(self.geometry.get_scaling)
            self.log(f"train/scales", scale_sum)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                out["comp_rgb"].permute(0, 3, 1, 2)
            )
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        ):
            loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
                + tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            )
            self.log(f"train/loss_depth_tv", loss_depth_tv)
            loss += loss_depth_tv

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_sds.backward(retain_graph=True)
        iteration = self.global_step
        if loss > 0:
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss_sds}
    

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.global_step,
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.global_step,
        )

    def on_load_checkpoint(self, ckpt_dict) -> None:
        num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        super().on_load_checkpoint(ckpt_dict)
