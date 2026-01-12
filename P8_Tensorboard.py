# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("logs")

# # writer.add_image()
# # writer.add_scalar()

# for i in range(100):
#     writer.add_scalar("y=2x", 2*i, i)
    
# writer.close()


from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "..\\dataset\\practice_data\\train\\ants_image/6240329_72c01e663e.jpg"
img_pil = Image.open(image_path)
img_pil.show()
print("PIL size (W, H):", img_pil.size)
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, 1, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()