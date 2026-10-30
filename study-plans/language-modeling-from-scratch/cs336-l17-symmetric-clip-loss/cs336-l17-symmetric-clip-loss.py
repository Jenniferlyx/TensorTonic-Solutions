import torch

def symmetric_clip_loss(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, temperature: int | float,
) -> dict:
    """
    Returns a dict: loss and two directional accuracies (scalars), similarities (tensor).
    """
    images = F.normalize(image_embeddings, dim=1)
    texts = F.normalize(text_embeddings, dim=1)
    similarities = (images @ texts.T) / temperature
    def directional_loss(logits):
        shifted = logits - logits.max(dim=1, keepdim=True).values
        log_denominator = torch.log(
            torch.exp(shifted).sum(dim=1)
        )
        return (log_denominator - shifted.diagonal()).mean()

    image_to_text_loss = directional_loss(similarities)
    text_to_image_loss = directional_loss(similarities.T)
    loss = (image_to_text_loss + text_to_image_loss) / 2
    targets = torch.arange(
        image_embeddings.shape[0],
        device=image_embeddings.device
    )
    image_to_text_accuracy = (
        similarities.argmax(dim=1) == targets
    ).to(image_embeddings.dtype).mean()
    text_to_image_accuracy = (
        similarities.argmax(dim=0) == targets
    ).to(image_embeddings.dtype).mean()

    return {
        "loss": loss,
        "similarities": similarities,
        "image_to_text_accuracy": image_to_text_accuracy,
        "text_to_image_accuracy": text_to_image_accuracy,
    }
