def predict(model_path, image_path):
    from torchvision import transforms
    from torch import nn
    import torch
    from PIL import Image

    index_to_name = {0:'Covid', 1:'Normal', 2:'Viral Pnuemonia'}
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model_eval = torch.load(model_path, map_location=device)

    # open method used to open different extension image file
    im = Image.open(image_path) 
    transform = transforms.Compose([
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                               ])

    image = transform(im).unsqueeze(0)
    with torch.no_grad():
        image = image.to(device)
        model_eval = model_eval.to(device)
        model_eval.eval()
        outputs = model_eval(image)
        score, predicted = torch.max(outputs.data, 1)
    print(index_to_name[predicted.tolist()[0]])
    return index_to_name[predicted.tolist()[0]]
