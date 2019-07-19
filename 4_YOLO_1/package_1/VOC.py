# print('test')
import torch
import torch.utils.data as data
import os
import xml.etree.ElementTree as ET
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

IMAGE_FOLDER="JPEGImages"
LABEL_FOLDER="Annotations"
IMG_EXTENSIONS=".jpg"


def detection_collate(batch):
    ''' 
    puts each data field into a tensor with other dimentional batch size

    Args:
    batch[0]: image
    batch[1]: label
    batch[2]: size
    '''
    targets=[]
    imgs=[]
    sizes=[]

    for sample in  (batch):
        img = torch.from_numpy(sample[0]).float()
        imgs.append(img)

        # for drawing box
        # if using batch it should keep original image size.
        sizes.append(sample[2])

        np_label = np.zeros((7, 7, 6), dtype=np.float32)
        for object in sample[1]:
            objectness = 1
            classes = object[0]
            x_ratio = object[1]
            y_ratio = object[2]
            w_ratio = object[3]
            h_ratio = object[4]

            # can be acuqire grid (x,y) index when divide (1/S) of x_ratio
            scale_factor = (1 / 7)
            grid_x_index = int(x_ratio // scale_factor)
            grid_y_index = int(y_ratio // scale_factor)
            x_offset = (x_ratio / scale_factor) - grid_x_index
            y_offset = (y_ratio / scale_factor) - grid_y_index

            # insert object row in specific label tensor index as (x,y)
            # object row follow as
            # [objectness, class, x offset, y offset, width ratio, height ratio]
            np_label[grid_x_index][grid_y_index] = np.array([objectness, x_offset, y_offset, w_ratio, h_ratio, classes])
        label = torch.from_numpy(np_label)
        targets.append(label)

    return torch.stack(imgs, 0), torch.stack(targets, 0), sizes

class VOC(torch.utils.data.Dataset):
    def __init__(self,root='./dataset/pascal_voc_100',
                    classes_name_path_file='./voc.names',
                    resize_factor=480,
                    transform=None):
        print("the class is created")
        self.classes_path=classes_name_path_file
        self.root=root
        self.resize_factor=resize_factor
        self.transform=transform

        with open(os.path.abspath(classes_name_path_file)) as f:
            self.classes= f.read().splitlines()

        # print(self.classes)
        # Step (1) check data set avaibility
        self._check_exist()
        # Step (2) parse XML Data and get bound boxes data in the form of data dictionary
        self.dict_data=self.VOC_Parse()
        print("Step (2): parsing the XMLs files are done.")
        # self.yolo_genrate(self.dict_data)
        # print(len(self.dict_data))
        # Step (3) Images are converted to yolo format and scaled to 1.
        
        # the yolo_format is a dictionary with the key--> name of images and its bunding boxes information:
        # bunding box -->(id,x,y,w,h'\n')
        # for example if we have one object in image it will be:
        # '2007_000549': '80.4560.5480.9070.9\n'} --> 8,0.456,0.548,0.907,0.9,\n
        # if it has more than one objest:(for example 2 objects)
        # '2007_000629': '190.5340.5830.9320.435\n150.1940.4870.040.104\n'
        #
        self.yolo_format=self.yolo_genrate(self.dict_data)
        print("Step (3) Images are converted to yolo format and scaled to 1.")
        # print(len(self.yolo_format))


        # The out put of following function is below dictonary for each image
        #{
        # './dataset/pascal_voc_100/JPEGImages/2007_001585.jpg': 
        # [[15.0, 0.525, 0.545, 0.262, 0.56], [18.0, 0.431, 0.6, 0.838, 0.371], [5.0, 0.13, 0.402, 0.028, 0.076]]
        # }
        self.data=self.ConvertYoloFormatFromStringtoFloatDict()
        print("Step (4) Yolo Foramat Dataset is cratred. There is a Dic {Image_absolute_path: its bound boxes info (float)}")
        # print(self.data)

        # img,target=self.__getitem__(9)
        # print(target)
        print('torch.utils.data.Dataset object from the Pascal VOC Data in Yolo Format is Created.')






        # keys= list(self.yolo_format.keys())
        # result=[]
        # # print(key[0].split("_")[-1])
        # keys=sorted(keys, key=lambda key: int(key.split("_")[-1]))
        # # print(key)
        # for key in keys:
        #     contents= list(self.yolo_format[key].split('\n'))
        #     contents=contents[:-1]    
        #     # print(contents)
        #     target=[]
        #     for i in range(len(contents)):
        #         temp=contents[i]
        #         # print((temp))
        #         temp=temp[:-1]
        #         temp=temp.split(" ")
        #         # print((temp))
        #         for j in range(len(temp)):
        #             temp[j]=float(temp[j])
        #             # print(float(temp[j]))
        #         target.append(temp)
        #     result.append({os.path.join(self.root,IMAGE_FOLDER , "".
        #     join([key, IMG_EXTENSIONS])): target})
        #     print(result)

        


        # with open(self.dict_data , 'r') as file:
        #     cls_list=file.read().splitlines()
        # print(cls_list)

    

    def _check_exist(self):
        print("Image_Folder: {}".format(os.path.join(self.root,IMAGE_FOLDER)))
        print("Label_Folder: {}".format(os.path.join(self.root,LABEL_FOLDER)))
        _,_,images_name=next(os.walk(os.path.join(self.root,IMAGE_FOLDER))) 
        if len(images_name) > 0:
            print("Step (1): The dataset is available.")
            print("The dataset has {} images.".format(len(images_name)))
        else:
            print("Step (1---> Error): The dataset is not available..")
        
    def VOC_Parse(self):
        (dir_path,dir_name,files_names)=next(os.walk(os.path.join(self.root,LABEL_FOLDER))) 
        data={}

        for filename in files_names:
            xml= open(os.path.join(dir_path,filename),'r')

            tree= ET.parse(xml)
            root=tree.getroot()
            xml_size= root.find("size")
            size={
                "width":xml_size.find("width").text,
                "height":xml_size.find("height").text,
                "depth":xml_size.find("depth").text
            }

            objects= root.findall("object")

            # if len (objects) == 0:
            #     return False , "Number of object is Zero!"

            obj={ 
                "num_obj": len(objects)
            }

            obj_index=0

            for _object in objects:
                tmp = { "name": _object.find("name").text }
                # print(tmp['name'])
                xml_bndbox=_object.find("bndbox")
                bndbox={
                    "xmin":xml_bndbox.find("xmin").text,
                    "ymin":xml_bndbox.find("ymin").text,
                    "xmax":xml_bndbox.find("xmax").text,
                    "ymax":xml_bndbox.find("ymax").text
                }
                tmp["bndbox"]=bndbox

                obj[str(obj_index)]=tmp
                obj_index +=1

            annotation ={
                "size":size,
                "objects":obj
            }

            if obj_index !=0:
                data[filename.split(".")[0]]=annotation
            
        return data

    def yolo_genrate(self,data):
        (dir_path,dir_name,files_names)=next(os.walk(os.path.join(self.root,LABEL_FOLDER))) 

        result={}

        for key in data:
            # print(key)
            img_width= int(data[key]["size"]["width"])
            img_height= int(data[key]["size"]["height"])
            contents=""

            for idx in range (0 , int(data[key]["objects"]["num_obj"])):
                xmin=(data[key]["objects"][str(idx)]["bndbox"]["xmin"])
                ymin=(data[key]["objects"][str(idx)]["bndbox"]["ymin"])
                xmax=(data[key]["objects"][str(idx)]["bndbox"]["xmax"])
                ymax=(data[key]["objects"][str(idx)]["bndbox"]["ymax"])
                # b is the dimention of bund box based on pixels (x0,x1,y0,y1)
                b=(float(xmin), float(xmax),float(ymin),float(ymax))
                # print(b)
                ## bb is the scaled vertion of b depends on image size --> it is between 0 and 1
                bb=self.CoordinatedToYolo((img_width,img_height),b)
                # print(bb)
                id=self.classes.index(data[key]["objects"][str(idx)]["name"])
                bndbox="".join(["".join([str(e)," "])for e in bb])
                contents="".join([contents,str(id)," ",bndbox[:],'\n'])
                # print([str(id),"",bndbox[:-1],'\n'])
                # print(contents)
                
            result[key]=contents
            # print(result)
        return result
           
                


            # img_width= int(data[key])

    def CoordinatedToYolo(self,size,box):
        dw= 1. / size[0]
        dh= 1. / size[1]
        # center of boundbox
        x= (box[0]+box[1]) / 2.0
        y= (box[2]+box[3]) / 2.0
        # dimention of the bounding box --> w & h
        w= box[1]-box[0]
        h= box[3]-box[2]

        x= x * dw
        w= w * dw
        y= y * dh
        h= h * dh

        return (round(x,3) , round(y,3) , round(w,3), round(h,3)) 
    
    def ConvertYoloFormatFromStringtoFloatDict(self):
        keys= list(self.yolo_format.keys())
        result=[]
        # print(key[0].split("_")[-1])
        keys=sorted(keys, key=lambda key: int(key.split("_")[-1]))
        # print(key)
        for key in keys:
            contents= list(self.yolo_format[key].split('\n'))
            contents=contents[:-1]    
            # print(contents)
            target=[]
            for i in range(len(contents)):
                temp=contents[i]
                # print((temp))
                temp=temp[:-1]
                temp=temp.split(" ")
                # print((temp))
                for j in range(len(temp)):
                    temp[j]=float(temp[j])
                    # print(float(temp[j]))
                target.append(temp)
            result.append({os.path.join(self.root,IMAGE_FOLDER , "".
            join([key, IMG_EXTENSIONS])): target})
        return result
            # print(result)

    def __len__(self):
        return len(self.dict_data)

    def __getitem__(self,index):
        '''
        Args:
        index (int)

        Return:
            tuple: Tuple (image , target)
            target--> [[class, x,y,w,h]] 
        '''

        key= list(self.data[index].keys())[0]
        img=Image.open(key).convert('RGB')
        current_shape=img.size
        img= img.resize((self.resize_factor,self.resize_factor))
        # print(img.size)
        img = np.array(img, dtype=float)
        img=np.transpose(img, (2, 0, 1))
        target=self.data[index][key]
        # print(target)

        # if self.transform is not None:
        #     img , aug_target= self.transform([img, target])
        #     print(aug_target)
        # return img , target[0][0]
        return img , target, current_shape


                


        





        


if __name__ == "__main__":
    dataset=VOC()
    print('torch.utils.data.Dataset object from the Pascal VOC Data in Yolo Format is Created.')
    trainloader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=15, 
                                          shuffle=True, 
                                          collate_fn=dataset.detection_collate)
    print(type(trainloader))

    idx=iter(trainloader)
    images,targets=next(idx)

    # tt.detection_collate(tt.__getitem__)

