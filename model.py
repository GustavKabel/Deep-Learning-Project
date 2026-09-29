import segmentation_models_pytorch as smp
import torch

def get_unet_resnet50(in_channels=13, num_classes=6):
    """
    Initializes a U-Net with a ResNet50 encoder.
    """
    model = smp.Unet(
        encoder_name="resnet50",        # Choose ResNet50 as the backbone
        encoder_weights="imagenet",     # Use pre-trained weights for a head start
        in_channels=in_channels,        # Override 3-channel default for our 13 bands
        classes=num_classes,            # Output 6 channels (logits for our 6 classes)
    )
    return model

if __name__ == "__main__":
    # Quick test to ensure the dimensions work before training
    dummy_input = torch.randn(2, 13, 512, 512) # Batch of 2, 13 bands, 512x512
    test_model = get_unet_resnet50()
    output = test_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Batch, Classes, Height, Width)")