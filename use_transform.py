from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter("logs")
img = Image.open("dataset/practice_data/train/ants_image/0013035.jpg")
print(img)

# totensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)


# Normalize的使用
print(img_tensor[0][0][0])
trans_norm =transforms.Normalize([6,3,2],[9,3,5])
img_norm = trans_norm(img_tensor)
# output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

# Resize的使用
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL -> Resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> ToTensor -> img_resize Tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)

# Compose - resize - 2
trans_resize_2=transforms.Resize(512)
# PIL -> PIL -> Tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)


# RandomCrop
trans_random = transforms.RandomCrop(500,1000)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()

