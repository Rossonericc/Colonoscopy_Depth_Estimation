import torchvision
import Image
resnet = torchvision.models.resnet18(pretrained=True)
image = Image.open("test.png")
image_tensor = preprocess(image)

output = resnet.conv1(image_tensor.unsqueeze(0))

