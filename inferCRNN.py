import torch
from modelCRNN import CRNN
from dataloaders import OCRVocab
from torchvision import transforms
from PIL import Image


# Path to checkpoint
ckpt_path = "checkpoints/model_best1.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

# Rebuild vocab
vocab = OCRVocab()
vocab.chars = ckpt["vocab"]
vocab.c2i = {c: i+2 for i,c in enumerate(vocab.chars)}
vocab.i2c = {i+2: c for i,c in enumerate(vocab.chars)}

# Rebuild model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_h=32, num_classes=len(vocab.chars)+2).to(device)  # +2 for PAD,UNK
model.load_state_dict(ckpt["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),   # match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def ctc_greedy_decode(logits, vocab):
    # logits: [T, 1, C]
    pred = logits.softmax(2).argmax(2)   # [T, 1]
    pred = pred.squeeze(1).tolist()      # [T]

    result = []
    prev = None
    for idx in pred:
        if idx != vocab.PAD and idx != prev:   # collapse repeats, skip blank
            result.append(idx)
        prev = idx
    return vocab.decode(result)


img_path = "/data/Vitou/testset/image.png"

img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)   # [1,1,32,128]

with torch.no_grad():
    logits = model(img)   # [T, 1, C]
    text = ctc_greedy_decode(logits, vocab)

print("Predicted text:", text)