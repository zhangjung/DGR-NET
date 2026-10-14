"""Model definitions: preprocessing blocks, dual EfficientNet-B2 encoder, simplified decoder,
fusion model, and loss functions (FocalLoss, UNetMultiTaskLoss)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2

from utils import logger, rgb_to_hsv_tensor


# =============================================================================
# Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# Preprocessing modules
# =============================================================================
class SVFeatureTransformation(nn.Module):
    """SV-channel feature transformation module."""

    def __init__(self, input_channels=2, output_channels=3):
        super(SVFeatureTransformation, self).__init__()
        self.sv_transform = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, output_channels),
            nn.Sigmoid()
        )
        logger.info(f"SV feature transformation: {input_channels} -> {output_channels} channels")

    def forward(self, sv_input):
        return self.sv_transform(sv_input)


class DepthFeatureTransformation(nn.Module):
    """Depth feature transformation module."""

    def __init__(self, input_channels=1, output_channels=3):
        super(DepthFeatureTransformation, self).__init__()
        self.depth_transform = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, output_channels),
            nn.Sigmoid()
        )
        logger.info(f"Depth feature transformation: {input_channels} -> {output_channels} channels")

    def forward(self, depth_input):
        return self.depth_transform(depth_input)


class RGBSVFusionModule(nn.Module):
    """RGB-SV fusion module."""

    def __init__(self, input_channels=6, output_channels=3):
        super(RGBSVFusionModule, self).__init__()
        self.rgb_sv_fusion = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, output_channels),
            nn.Tanh()
        )
        logger.info(f"RGB-SV fusion: {input_channels} -> {output_channels} channels")

    def forward(self, rgb_sv_input):
        return self.rgb_sv_fusion(rgb_sv_input)


# =============================================================================
# Dual EfficientNet-B2 encoder
# =============================================================================
class DualEfficientNetB2Encoder(nn.Module):
    """
    Dual EfficientNet-B2 encoder.
    - The RGB-SV branch uses one EfficientNet-B2
    - The depth branch uses another EfficientNet-B2
    - Intermediate features are extracted stage-by-stage and fused by element-wise addition
    """

    def __init__(self, pretrained=True):
        super(DualEfficientNetB2Encoder, self).__init__()

        # RGB-SV encoder
        self.rgb_sv_encoder = efficientnet_b2(pretrained=pretrained)
        self.rgb_sv_encoder.classifier = nn.Identity()

        # Depth encoder
        self.depth_encoder = efficientnet_b2(pretrained=pretrained)
        self.depth_encoder.classifier = nn.Identity()

        # Register hooks to capture intermediate-stage features
        self.rgb_sv_features = {}
        self.depth_features = {}
        self.fused_features = {}

        self._register_hooks()

        logger.info("Dual EfficientNet-B2 encoder initialized")
        logger.info("  RGB-SV encoder: EfficientNet-B2")
        logger.info("  Depth encoder: EfficientNet-B2")
        logger.info("  Fusion strategy: stage-wise addition")


    def _register_hooks(self):
        """Register forward hooks to capture per-stage features."""

        def get_rgb_sv_activation(name):
            def hook(model, input, output):
                self.rgb_sv_features[name] = output

            return hook

        def get_depth_activation(name):
            def hook(model, input, output):
                self.depth_features[name] = output

            return hook

        # RGB-SV encoder hooks
        self.rgb_sv_encoder.features[1].register_forward_hook(get_rgb_sv_activation('stage1'))
        self.rgb_sv_encoder.features[2].register_forward_hook(get_rgb_sv_activation('stage2'))
        self.rgb_sv_encoder.features[3].register_forward_hook(get_rgb_sv_activation('stage3'))
        self.rgb_sv_encoder.features[4].register_forward_hook(get_rgb_sv_activation('stage4'))
        self.rgb_sv_encoder.features[5].register_forward_hook(get_rgb_sv_activation('stage5'))
        self.rgb_sv_encoder.features[6].register_forward_hook(get_rgb_sv_activation('stage6'))
        self.rgb_sv_encoder.features[7].register_forward_hook(get_rgb_sv_activation('stage7'))

        # Depth encoder hooks
        self.depth_encoder.features[1].register_forward_hook(get_depth_activation('stage1'))
        self.depth_encoder.features[2].register_forward_hook(get_depth_activation('stage2'))
        self.depth_encoder.features[3].register_forward_hook(get_depth_activation('stage3'))
        self.depth_encoder.features[4].register_forward_hook(get_depth_activation('stage4'))
        self.depth_encoder.features[5].register_forward_hook(get_depth_activation('stage5'))
        self.depth_encoder.features[6].register_forward_hook(get_depth_activation('stage6'))
        self.depth_encoder.features[7].register_forward_hook(get_depth_activation('stage7'))

    def forward(self, rgb_sv_input, depth_input):
        """Forward pass."""
        self.rgb_sv_features = {}
        self.depth_features = {}
        self.fused_features = {}

        # Run both encoders to extract features
        rgb_sv_final = self.rgb_sv_encoder(rgb_sv_input)
        depth_final = self.depth_encoder(depth_input)

        # Stage-wise additive fusion
        stage_names = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7']
        for stage_name in stage_names:
            rgb_sv_feat = self.rgb_sv_features.get(stage_name)
            depth_feat = self.depth_features.get(stage_name)

            if rgb_sv_feat is not None and depth_feat is not None:
                fused_feat = rgb_sv_feat + depth_feat
                self.fused_features[stage_name] = fused_feat

        return rgb_sv_final, depth_final, self.fused_features


# =============================================================================
# Simplified decoder (consumes raw skip-connection features; no channel compression)
# =============================================================================
class SimplifiedDecoder(nn.Module):
    """Simplified decoder that consumes raw skip-connection features without channel compression."""

    def __init__(self, output_channels=1):
        super(SimplifiedDecoder, self).__init__()

        self.output_channels = output_channels

        # Stage 7 -> Stage 6 (352 -> 208 + 208skip = 416)
        self.decode7 = nn.Sequential(
            nn.ConvTranspose2d(352, 208, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(208),
            nn.ReLU(inplace=True)
        )
        self.fuse6 = nn.Sequential(
            nn.Conv2d(208 + 208, 208, kernel_size=3, padding=1),  # 416 -> 208
            nn.BatchNorm2d(208),
            nn.ReLU(inplace=True)
        )

        # Stage 6 -> Stage 5 (208 -> 120 + 120skip = 240)
        self.decode6 = nn.Sequential(
            nn.ConvTranspose2d(208, 120, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True)
        )
        self.fuse5 = nn.Sequential(
            nn.Conv2d(120 + 120, 120, kernel_size=3, padding=1),  # 240 -> 120
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True)
        )

        # Stage 5 -> Stage 4 (120 -> 88 + 88skip = 176)
        self.decode5 = nn.Sequential(
            nn.ConvTranspose2d(120, 88, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(88),
            nn.ReLU(inplace=True)
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(88 + 88, 88, kernel_size=3, padding=1),  # 176 -> 88
            nn.BatchNorm2d(88),
            nn.ReLU(inplace=True)
        )

        # Stage 4 -> Stage 3 (88 -> 48 + 48skip = 96)
        self.decode4 = nn.Sequential(
            nn.ConvTranspose2d(88, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, kernel_size=3, padding=1),  # 96 -> 48
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Stage 3 -> Stage 2 (48 -> 24 + 24skip = 48)
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(24 + 24, 24, kernel_size=3, padding=1),  # 48 -> 24
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # Stage 2 -> Stage 1 (24 -> 16 + 16skip = 32)
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(24, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, kernel_size=3, padding=1),  # 32 -> 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.final_decode = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        logger.info(f"Simplified decoder initialized (direct skip connections, full channels), output channels: {output_channels}")
        logger.info(f"  Channel flow (no compression):")
        logger.info(f"    Stage7->6: 208+208=416 -> 208")
        logger.info(f"    Stage6->5: 120+120=240 -> 120")
        logger.info(f"    Stage5->4: 88+88=176 -> 88")
        logger.info(f"    Stage4->3: 48+48=96 -> 48")
        logger.info(f"    Stage3->2: 24+24=48 -> 24")
        logger.info(f"    Stage2->1: 16+16=32 -> 16")

    def forward(self, fused_features):
        """Forward pass."""
        stage7 = fused_features.get('stage7')
        stage6 = fused_features.get('stage6')
        stage5 = fused_features.get('stage5')
        stage4 = fused_features.get('stage4')
        stage3 = fused_features.get('stage3')
        stage2 = fused_features.get('stage2')
        stage1 = fused_features.get('stage1')

        # Stage 7 -> Stage 6
        x = self.decode7(stage7)
        if x.size()[2:] != stage6.size()[2:]:
            x = F.interpolate(x, size=stage6.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, stage6], dim=1)  # 208 + 208 = 416
        x = self.fuse6(x)  # 416 -> 208

        # Stage 6 -> Stage 5
        x = self.decode6(x)
        if x.size()[2:] != stage5.size()[2:]:
            x = F.interpolate(x, size=stage5.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, stage5], dim=1)  # 120 + 120 = 240
        x = self.fuse5(x)  # 240 -> 120

        # Stage 5 -> Stage 4
        x = self.decode5(x)
        if x.size()[2:] != stage4.size()[2:]:
            x = F.interpolate(x, size=stage4.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, stage4], dim=1)  # 88 + 88 = 176
        x = self.fuse4(x)  # 176 -> 88

        # Stage 4 -> Stage 3
        x = self.decode4(x)
        if x.size()[2:] != stage3.size()[2:]:
            x = F.interpolate(x, size=stage3.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, stage3], dim=1)  # 48 + 48 = 96
        x = self.fuse3(x)  # 96 -> 48

        # Stage 3 -> Stage 2
        x = self.decode3(x)
        if x.size()[2:] != stage2.size()[2:]:
            x = F.interpolate(x, size=stage2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, stage2], dim=1)  # 24 + 24 = 48
        x = self.fuse2(x)  # 48 -> 24

        # Stage 2 -> Stage 1
        x = self.decode2(x)
        if x.size()[2:] != stage1.size()[2:]:
            x = F.interpolate(x, size=stage1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, stage1], dim=1)  # 16 + 16 = 32
        x = self.fuse1(x)  # 32 -> 16

        # Final output
        reconstructed_depth = self.final_decode(x)

        return reconstructed_depth


# =============================================================================
# Main model: dual-encoder stage-wise fusion + final-stage concatenation classifier (simplified)
# =============================================================================
class DualEncoderLayerFusionModel(nn.Module):
    """Dual-encoder stage-wise fusion model (simplified, no bottleneck compression)."""

    def __init__(self, pretrained=True, num_classes=1):
        super(DualEncoderLayerFusionModel, self).__init__()

        # Preprocessing modules
        self.sv_feature_transform = SVFeatureTransformation(input_channels=2, output_channels=3)
        self.depth_feature_transform = DepthFeatureTransformation(input_channels=1, output_channels=3)
        self.rgb_sv_fusion = RGBSVFusionModule(input_channels=6, output_channels=3)

        # Dual encoder
        self.dual_encoder = DualEfficientNetB2Encoder(pretrained=pretrained)

        # Decoder (simplified)
        self.decoder = SimplifiedDecoder(output_channels=1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1408 + 1408, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

        logger.info("Dual-encoder stage-wise fusion model initialized (simplified, no bottleneck compression)")

    def forward(self, rgb_input, depth_input):
        """Forward pass."""
        # ========== RGB-SV processing (no FFT, uses raw RGB directly) ==========
        # 1. Extract SV channels from the raw RGB input
        sv_channels = rgb_to_hsv_tensor(rgb_input)
        transformed_sv = self.sv_feature_transform(sv_channels)

        # 2. Fuse the raw RGB with the transformed SV
        rgb_sv_concatenated = torch.cat([rgb_input, transformed_sv], dim=1)
        rgb_sv_fused = self.rgb_sv_fusion(rgb_sv_concatenated)

        # ========== Depth processing ==========
        transformed_depth = self.depth_feature_transform(depth_input)

        # ========== Extract features via the dual encoder ==========
        rgb_sv_features, depth_features, fused_features = self.dual_encoder(
            rgb_sv_fused, transformed_depth
        )

        # ========== Classify by concatenating final-stage features ==========
        concat_features = torch.cat([rgb_sv_features, depth_features], dim=1)
        classification_output = self.classifier(concat_features)

        # ========== Reconstruct the depth map via the decoder ==========
        reconstructed_depth = self.decoder(fused_features)

        return {
            'classification': classification_output,
            'reconstructed_depth': reconstructed_depth,
            'original_depth': depth_input,
            'rgb_sv_fused': rgb_sv_fused,
            'rgb_sv_features': rgb_sv_features,
            'depth_features': depth_features,
            'fused_features': fused_features
        }

    def get_features(self, rgb_input, depth_input):
        """Return the concatenated feature vector."""
        with torch.no_grad():
            outputs = self.forward(rgb_input, depth_input)
            return torch.cat([outputs['rgb_sv_features'], outputs['depth_features']], dim=1)


# =============================================================================
# Multi-task loss
# =============================================================================
class UNetMultiTaskLoss(nn.Module):
    """Multi-task loss function."""

    def __init__(self, classification_weight=1.0, reconstruction_weight=0.5,
                 use_focal_loss=True, focal_alpha=0.75, focal_gamma=2.0):
        super(UNetMultiTaskLoss, self).__init__()

        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"Using Focal Loss - alpha:{focal_alpha}, gamma:{focal_gamma}")
        else:
            self.classification_loss = nn.BCEWithLogitsLoss()
            logger.info("Using BCE Loss")

        self.reconstruction_loss = nn.MSELoss()

        logger.info(f"Multi-task loss - classification weight: {classification_weight}, reconstruction weight: {reconstruction_weight}")

    def forward(self, model_outputs, labels):
        class_loss = self.classification_loss(
            model_outputs['classification'],
            labels.view(-1, 1)
        )

        recon_loss = self.reconstruction_loss(
            model_outputs['reconstructed_depth'],
            model_outputs['original_depth']
        )

        total_loss = (self.classification_weight * class_loss +
                      self.reconstruction_weight * recon_loss)

        return total_loss, {
            'total_loss': total_loss,
            'classification_loss': class_loss,
            'reconstruction_loss': recon_loss
        }
