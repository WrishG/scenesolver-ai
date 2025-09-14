import torch
import torch.nn as nn
import clip

class CLIPMultiClassClassifier(nn.Module):
    """
    This single class works for both multi-class and binary models.
    """
    def __init__(self, num_classes: int, freeze_clip: bool = True):
        super().__init__()
        # Load the model initially to the CPU. The loading script will move it.
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")

        self.visual_encoder = self.clip_model.visual

        if freeze_clip:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # --- FINAL FIX ---
        # No more manual data type casting. We pass the input tensor directly.
        # The CLIP model will handle its own mixed-precision logic internally.
        feats = self.visual_encoder(x)
        # --- END OF FIX ---
        
        # The classifier head runs in full precision for stability.
        logits = self.classifier(feats.float())
        return logits