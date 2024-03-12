import os
from torch.utils.data import Dataset
from tqdm import tqdm

class PairedDataset(Dataset):
    def __init__(self, examples, image_field, text_field) -> None:
        super(PairedDataset, self).__init__()
        self.examples = examples
        self.image_field = image_field
        self.text_field = text_field

    def __getitem__(self, index):
        example = self.examples[index]
        image = self.image_field.flickr_process(example['image_path'])
        text = self.text_field.process(example['caption'])[0]
        id = int(example['image_id'])
        return id, image, text

    def __len__(self):
        return len(self.examples)


class Flickr(PairedDataset):
    def __init__(self, image_folder, annotation_path, image_field, text_field) -> None:
        self.image_field = image_field
        self.text_field = text_field

        
        images_name_list = os.listdir(image_folder)

        self.examples = []
        it = 0
        with tqdm(desc="Prepare data", unit="it", total=len(images_name_list) * 5) as pbar:
            with open(annotation_path, "r", encoding='utf-8') as f:
                for line in f:
                    content = line.split('\t')
                    img_name = content[0].split("#")[0]
                    cap = content[1].replace("\n", "")
                    if img_name in images_name_list:
                        self.examples.append({
                            'image_path':os.path.join(image_folder, img_name),
                            'image_id': img_name.split(".")[0],
                            'caption':cap})
                    else:
                        it += 1
                    pbar.update()
            if it > 0: print("漏加载了{}个图像文本对".format(it))
    @property
    def splits(self):
        train = PairedDataset(self.examples[:-2000], self.image_field, self.text_field)
        val = PairedDataset(self.examples[-2000:-1000], self.image_field, self.text_field)
        test = PairedDataset(self.examples[-1000:], self.image_field, self.text_field)
        return train, val, test
    

    



# if __name__ == '__main__':

    # transform = Compose([
    #         Resize(224, interpolation=BICUBIC),
    #         CenterCrop(224),
    #         lambda image: image.convert("RGB"),
    #         ToTensor(),
    #         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    #         ])

    # image_field = ImageField(transform=transform)
    # text_field = TextField()
    # dataset = Flickr('/raid/ggw/datasets/flickr30k/flickr30k-images', '/raid/ggw/datasets/flickr30k/results_20130124.token', image_field, text_field)
    # train, val, test = dataset.split()
    # dataloader_train = DataLoader(train, batch_size=10, shuffle=True, num_workers=0)
    # for it, (img_id, images, texts_gt) in enumerate(dataloader_train):
    #     print(img_id)
