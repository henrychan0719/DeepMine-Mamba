"""DeepMine Mamba — core architecture for document image binarization.

This release contains only the network definition, plus a sliding-window
inference helper for full-page images. The training pipeline (data
augmentation, losses, schedules) is intentionally not included.

Dependencies:
    torch, timm, einops, mamba-ssm   (mamba-ssm requires a CUDA build)

Expected input:
    RGB tensor of shape (B, 3, H, W), normalized with ImageNet statistics
    mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)

Output:
    A dict of tensors. "soft_bin" is the differentiable binarization map
    used at inference time: text (foreground) ~ 1, background ~ 0.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from einops import rearrange
import timm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# =========================================================
# Feature gating / refinement modules
# =========================================================
class LocalVarianceSuppressor(nn.Module):
    """Suppresses noisy regions using local variance as a cue."""

    def __init__(self, channels, kernel=9):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel, stride=1, padding=kernel // 2)
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        x_f = x.float()
        mu_f = self.pool(x_f)
        var = self.pool(x_f ** 2) - mu_f ** 2
        var = var.clamp(min=0)
        var_map = var.mean(dim=1, keepdim=True).to(x.dtype)
        return self.gate(torch.cat([x, var_map], dim=1))


class PoliceGate(nn.Module):
    """Predicts a per-pixel confidence map P and softly re-weights features."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        P = self.gate(x)
        x_clean = x * (0.5 + 0.5 * P)
        return x_clean, P


class LocalContrastRecovery(nn.Module):
    """Re-injects local contrast (x - local_mean) to recover faint strokes."""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        local_mean = self.pool(x)
        diff = x - local_mean
        return self.proj(torch.cat([x, diff], dim=1))


class SobelEdgePrior(nn.Module):
    """Fixed Sobel gradients fused with the RGB input as a low-level edge prior."""

    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.repeat(in_ch, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(in_ch, 1, 1, 1))
        self.net = nn.Sequential(
            nn.Conv2d(in_ch * 3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        fused_input = torch.cat([x, grad_x.abs(), grad_y.abs()], dim=1)
        return self.net(fused_input)


class PureDilutionDetector(nn.Module):
    """Estimates where global mixing has diluted the original local signal."""

    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, m):
        signal_loss = torch.abs(x - m)
        feat = torch.cat([x, signal_loss], dim=1)
        I = torch.sigmoid(self.net(feat))
        return I


class DeepMineAntiDilution(nn.Module):
    """Blends the input back into the mixed features where dilution is detected."""

    def __init__(self, channels, max_gate=0.7):
        super().__init__()
        self.detector = PureDilutionDetector(channels)
        self.max_gate = max_gate

    def forward(self, x, m, P_stroke=None):
        I = self.detector(x, m)
        if P_stroke is not None:
            dynamic_gate = self.max_gate * P_stroke.detach()
        else:
            dynamic_gate = self.max_gate
        I = I * dynamic_gate
        y = m + I * (x - m)
        return y, I


class PreWeakTextRescue(nn.Module):
    """Amplifies weak-text responses before the Mamba block."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, 1, 1)
        )

    def forward(self, x):
        W = torch.sigmoid(self.net(x))
        return x + x * W


class IntensityGuidedWeakText(nn.Module):
    """Recovers weak text guided by the current prediction confidence."""

    def __init__(self, channels):
        super().__init__()
        self.weak_finder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU()
        )

    def forward(self, x, m, confidence_thresh=0.35):
        W = torch.sigmoid(self.weak_finder(x))
        m_prob = torch.sigmoid(m.mean(dim=1, keepdim=True))
        miss_factor = 1.0 - m_prob
        active_mask = (m_prob > confidence_thresh).float()
        weak_feat = self.refine(x * W * miss_factor * active_mask)
        return weak_feat, W


class StructureRefinement(nn.Module):
    """Residual double-conv refinement."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


# =========================================================
# DeepMine Mamba core
# =========================================================
class PixelDirectionalAttention(nn.Module):
    """Pixel-wise softmax fusion of the four directional scan results."""

    def __init__(self, channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(channels * 4, 4, kernel_size=1)

    def forward(self, h_up, h_down, h_left, h_right):
        attn_weights = F.softmax(
            self.attention_conv(torch.cat([h_up, h_down, h_left, h_right], dim=1)), dim=1)
        w_up, w_down, w_left, w_right = torch.split(attn_weights, 1, dim=1)
        return (w_up * h_up) + (w_down * h_down) + (w_left * h_left) + (w_right * h_right)


class DeepMineMambaCore(nn.Module):
    """Single-direction selective-scan (S6) core, shared across the four scan directions."""

    def __init__(self, d_model, d_state=16, dt_rank="auto"):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward_direction(self, x_2d):
        x_2d = x_2d.float()
        B, C, H, W = x_2d.shape
        x = self.in_proj(rearrange(x_2d, 'b c h w -> b c (h w)'))
        dt_raw, B_raw, C_raw = torch.split(
            self.x_proj(x.transpose(1, 2)),
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        A = -torch.exp(self.A_log.float().clamp(max=10))
        out_seq = selective_scan_fn(
            x,
            self.dt_proj(dt_raw).transpose(1, 2).contiguous(),
            A,
            B_raw.transpose(1, 2).contiguous(),
            C_raw.transpose(1, 2).contiguous(),
            self.D.float(),
            z=None, delta_bias=None, delta_softplus=True, return_last_state=False
        )
        out = rearrange(out_seq, 'b c (h w) -> b c h w', h=H, w=W)
        return out.to(x_2d.dtype)


class DeepMineMambaBlock(nn.Module):
    """Stroke-prior-gated, four-directional Mamba block with anti-dilution."""

    def __init__(self, channels, lambda_bg=0.1, max_gate=0.7):
        super().__init__()
        self.lambda_bg = lambda_bg
        self.stroke_prior = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2), nn.GELU(),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.mamba_core = DeepMineMambaCore(d_model=channels)
        self.fusion = PixelDirectionalAttention(channels)
        self.anti_dilution = DeepMineAntiDilution(channels, max_gate=max_gate)

    def forward(self, x, return_importance=False):
        P_stroke = self.stroke_prior(x)
        x_fg = x * P_stroke + self.lambda_bg * x * (1.0 - P_stroke)
        h_right = self.mamba_core.forward_direction(x_fg)
        h_left = torch.flip(self.mamba_core.forward_direction(torch.flip(x_fg, [3])), [3])
        x_T = torch.transpose(x_fg, 2, 3)
        h_down = torch.transpose(self.mamba_core.forward_direction(x_T), 2, 3)
        h_up = torch.transpose(
            torch.flip(self.mamba_core.forward_direction(torch.flip(x_T, [3])), [3]), 2, 3)
        m = self.fusion(h_up, h_down, h_left, h_right)
        refined, I = self.anti_dilution(x, m, P_stroke=P_stroke)
        return (refined, I) if return_importance else refined


# =========================================================
# Main architecture
# =========================================================
class ConvNeXtDeepMineMamba(nn.Module):
    """ConvNeXt-Tiny encoder + U-Net decoder + DeepMine Mamba refinement.

    Args:
        max_gate:        upper bound of the anti-dilution gate.
        fixed_threshold: binarization threshold applied to clean_prob.
        th_sharpness:    sharpness of the differentiable thresholding.
        pretrained_backbone: load ImageNet weights for the encoder
                             (set False when loading a full checkpoint).
    """

    def __init__(self, return_importance=False, max_gate=0.7,
                 fixed_threshold=0.5, th_sharpness=20.0,
                 pretrained_backbone=True):
        super().__init__()
        self.return_importance = return_importance
        self.fixed_threshold = fixed_threshold
        self.th_sharpness = th_sharpness

        self.encoder = timm.create_model(
            'convnext_tiny', pretrained=pretrained_backbone, features_only=True)
        self.edge_extractor = SobelEdgePrior(in_ch=3, out_ch=32)

        self.up4 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(768, 384, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384), nn.GELU()
        )
        self.up3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(384, 192, 3, 1, 1, bias=False),
            nn.BatchNorm2d(192), nn.GELU()
        )
        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 96, 3, 1, 1, bias=False),
            nn.BatchNorm2d(96), nn.GELU()
        )

        self.contrast_recovery = LocalContrastRecovery(96)
        self.pre_weak = PreWeakTextRescue(96)
        self.deepmine_block_low = DeepMineMambaBlock(channels=96, lambda_bg=0.1, max_gate=max_gate)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
        )

        self.detail_fuse = nn.Sequential(
            nn.Conv2d(64 + 32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.GELU()
        )

        self.structure_refine = StructureRefinement(64)
        self.var_suppressor = LocalVarianceSuppressor(64)
        self.police_gate = PoliceGate(64)

        # Not used in forward(); kept so that existing checkpoints load with strict=True.
        self.high_refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
        )

        self.post_weak = IntensityGuidedWeakText(channels=64)
        self.lambda_w = 0.25

        self.fg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 1, 1)
        )
        self.bg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 1, 1)
        )

        self.skel_head = nn.Conv2d(96, 1, kernel_size=1)

    def forward(self, img):
        edge_prior = self.edge_extractor(img)
        f1, f2, f3, f4 = self.encoder(img)

        d4 = self.dec4(torch.cat([self.up4(f4), f3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), f2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), f1], dim=1))

        d2_contrast = self.contrast_recovery(d2)
        d2_pre = self.pre_weak(d2_contrast)
        d2_mamba, importance = self.deepmine_block_low(d2_pre, return_importance=True)

        skel_logits = self.skel_head(d2_mamba)

        d1 = self.up1(d2_mamba)
        x_crisp = self.detail_fuse(torch.cat([d1, edge_prior], dim=1))
        x_refined = self.structure_refine(x_crisp)

        x_suppressed = self.var_suppressor(x_refined)
        m, p_police = self.police_gate(x_suppressed)

        m_high = m
        weak_feat, weak_map = self.post_weak(x_refined, m_high)
        f = m_high + self.lambda_w * weak_feat

        fg_logit = self.fg_head(f)
        bg_logit = self.bg_head(f)

        fg_prob = torch.sigmoid(fg_logit)
        bg_prob = torch.sigmoid(bg_logit)

        clean_prob = (fg_prob * (1.0 - bg_prob)).clamp(1e-6, 1.0 - 1e-6)

        threshold_map = torch.full_like(clean_prob, self.fixed_threshold)

        # Differentiable binarization map (soft threshold).
        soft_bin = torch.sigmoid(
            (clean_prob - self.fixed_threshold) * self.th_sharpness
        )

        return {
            "fg_logit": fg_logit,
            "bg_logit": bg_logit,
            "fg_prob": fg_prob,
            "bg_prob": bg_prob,
            "threshold_map": threshold_map,
            "clean_prob": clean_prob,
            "soft_bin": soft_bin,
            "skel_logits": skel_logits,
            "importance": importance,
            "police": p_police,
            "weak_map": weak_map,
        }


# =========================================================
# Sliding-window inference (full-page images)
# =========================================================
def _normalized_white_value(device, dtype, batch_size=1):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
    white = (torch.ones((batch_size, 3, 1, 1), device=device, dtype=dtype) - mean) / std
    return white


def _extract_model_soft_output(output):
    if isinstance(output, dict):
        return output["soft_bin"]
    if isinstance(output, tuple):
        return torch.sigmoid(output[0])
    return torch.sigmoid(output)


def sliding_window_inference(model, img_tensor, crop_size=512, stride=256):
    """Run the model over a full image with overlapping windows and average the results."""
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    pred_map = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    count_map = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)

    y_positions = list(range(0, max(H - crop_size + 1, 1), stride))
    x_positions = list(range(0, max(W - crop_size + 1, 1), stride))
    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]
    if y_positions[-1] != max(H - crop_size, 0):
        y_positions.append(max(H - crop_size, 0))
    if x_positions[-1] != max(W - crop_size, 0):
        x_positions.append(max(W - crop_size, 0))

    with torch.inference_mode():
        for y in y_positions:
            for x in x_positions:
                y2, x2 = min(y + crop_size, H), min(x + crop_size, W)
                patch = img_tensor[:, :, y:y2, x:x2]
                pad_h = crop_size - patch.shape[2]
                pad_w = crop_size - patch.shape[3]
                if pad_h > 0 or pad_w > 0:
                    padded = torch.empty((B, C, crop_size, crop_size),
                                         device=device, dtype=patch.dtype)
                    white_value = _normalized_white_value(
                        device=device, dtype=patch.dtype, batch_size=B)
                    padded[:] = white_value
                    padded[:, :, :patch.shape[2], :patch.shape[3]] = patch
                    patch = padded
                with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    patch_output = model(patch)
                    patch_soft = _extract_model_soft_output(patch_output).detach()
                pred_map[:, :, y:y2, x:x2] += patch_soft[:, :, :y2 - y, :x2 - x].float()
                count_map[:, :, y:y2, x:x2] += 1.0
                del patch, patch_output, patch_soft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pred_map / (count_map + 1e-8)


if __name__ == "__main__":
    # Quick shape check (mamba-ssm's selective scan requires a CUDA GPU).
    assert torch.cuda.is_available(), "This model requires CUDA (mamba-ssm selective scan)."
    model = ConvNeXtDeepMineMamba(pretrained_backbone=False).cuda().eval()
    # model.load_state_dict(torch.load("deepmine_mamba_best.pth", map_location="cuda"))
    dummy = torch.randn(1, 3, 512, 512, device="cuda")
    with torch.inference_mode():
        out = model(dummy)
    for k, v in out.items():
        print(f"{k:>14s}: {tuple(v.shape)}")
