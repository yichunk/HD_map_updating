from PIL import Image, ImageDraw, ImageFilter
import json
import random

id_to_sign_labe = [ '',      #0
                    'regulatory--stop--g1', #1 
                    'regulatory--yield--g1', #2
                    'regulatory--no-entry--g1', #3 
                    'regulatory--no-parking--g2', #4
                    'regulatory--maximum-speed-limit-15--g3', #5
                    'regulatory--turn-right--g3', #6
                    'regulatory--go-straight--g3', #7
                    'regulatory--turn-left--g2', #8
                    'regulatory--no-right-turn--g1', #9
                    'regulatory--no-straight-through--g1', #10
                    'regulatory--no-left-turn--g1', #11
                    'warning--pedestrians-crossing--g4' #12
                   ]

def save_obj(obj, file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def random_paste_one_object(foreground, background, backgroubd_ann, object_label):
    img_w, img_h = background.size
    obj_to_background_ratio = random.uniform(0.01, 0.05)
    offset_x = round(random.uniform(0.2, 0.8) * img_w)
    offset_y = round(random.uniform(0.2, 0.8) * img_h)
    
    new_obj_w = round(obj_to_background_ratio * img_w)
    new_obj_h = round(obj_to_background_ratio * img_h)
    #print(new_obj_w)
    #print(new_obj_h)
    foreground = foreground.resize((new_obj_w, new_obj_h))
    background.paste(foreground, (offset_x, offset_y))

    new_obj = backgroubd_ann['objects'][-1].copy()
    new_obj['bbox']['xmin'] = offset_x
    new_obj['bbox']['ymin'] = offset_y
    new_obj['bbox']['xmax'] = offset_x + new_obj_w
    new_obj['bbox']['ymax'] = offset_y + new_obj_h
    new_obj['label'] = object_label
    backgroubd_ann['objects'].append(new_obj)

    
if __name__ == "__main__":
    im1 = Image.open('MTSD_samples/North_America/f82itz69069z_LC7xM6UIA.jpg')
    im2 = Image.open('MTSD_samples/7_5_6.jpg') #image_id_class_id_num_object


    ann1 = None
    with open('MTSD_samples/North_America/f82itz69069z_LC7xM6UIA.json') as json_file:
        ann1 = json.load(json_file) #annotation_data is a python dict

    random_paste_one_object(im2, im1, ann1, id_to_sign_labe[5])

    
    with open('output.json', 'w') as outfile:
        json.dump(ann1, outfile)
    
    im1.save('output.jpg', quality=95)
