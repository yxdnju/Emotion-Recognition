import torch

# 模型文件路径
model_path = 'G:/Experiment/Experiment3/OutCome/Save_Model/best_model.pth'

# 加载模型参数
# 如果您的环境支持CUDA，且模型是在CUDA上训练的，可以直接加载
# 如果环境不支持CUDA，需要将模型映射到CPU上
model_parameters = torch.load(model_path, map_location=torch.device('cpu'))

# 打印模型的参数
# 这里只打印参数的名称和尺寸，不打印具体数值，以避免输出过长
for name, param in model_parameters.items():
    print(f"Layer: {name} | Size: {param.size()}")

# 注意：如果您想查看具体的参数数值，可以去掉对数值的限制，但请注意输出可能会很长
