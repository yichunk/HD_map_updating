from PIL import Image, ImageDraw, ImageFilter
import json
import random

def random_paste_one_object(foreground, background, backgroubd_ann):
    img_w, img_h = background.size
    obj_to_background_ratio = random.uniform(0.01, 0.19)
    offset_x = round(random.uniform(0.2, 0.8) * img_w)
    offset_y = round(random.uniform(0.2, 0.8) * img_h)
    
    new_obj_w = round(obj_to_background_ratio * img_w)
    new_obj_h = round(obj_to_background_ratio * img_h)
    #print(new_obj_w)
    #print(new_obj_h)
    foreground = foreground.resize((new_obj_w, new_obj_h))
    background.paste(foreground, (offset_x, offset_y))

    
if __name__ == "__main__":
    im1 = Image.open('MTSD_samples/North_America/f82itz69069z_LC7xM6UIA.jpg')
    im2 = Image.open('MTSD_samples/7_5_6.jpg')


    ann1 = None
    with open('MTSD_samples/North_America/f82itz69069z_LC7xM6UIA.json') as json_file:
        ann1 = json.load(json_file) #annotation_data is a python dict

    print(ann1['objects'])
    #num_objects_max = len(annotation_data['objects'])

    random_paste_one_object(im2, im1, ann1)
    im1.save('output.jpg', quality=95)
