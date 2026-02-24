import os, glob
import sys
import pandas as pd
from datetime import date
import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import time
from sklearn.metrics import pairwise_distances
from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

date = date.today().strftime("%Y%m%d")
start_time = time.perf_counter()

        
def run_pipeline(im, input_folder, output_folder):
    image_name = im.replace(f'{input_folder}/', "").replace(".jpeg","")

    print(f"Processing: {image_name}")

    #Read in image
    im = cv.imread(im)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)


    # Segment using text prompt
    inputs = processor(images=im, text="stem", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]


    #Making a mask with all leaves
    masks = results["masks"]  # list or tensor of shape (N, H, W)

    # # Stack and combine (logical OR across objects)
    # if isinstance(masks, list):
    #     masks_tensor = torch.stack(masks)
    # else:
    #     masks_tensor = masks

    # combined_mask = masks_tensor.any(dim=0).cpu().numpy().astype(np.uint8) * 255

    # image_np = np.array(im)
    # masked_im = image_np * combined_mask[..., None]


    #filter out incorrect masks
    filtered_mask_ids = []
    for idx, mask in enumerate(masks):
        if np.array(mask.sum() <=70000):
            filtered_mask_ids.append(idx)
        
    masks = masks[filtered_mask_ids]
    result_boxes = results["boxes"]
    result_boxes = result_boxes[filtered_mask_ids]


    #Extract stem lengths and coordinates to draw later
    stem_lengths = []
    boxes = []
    for box in result_boxes:
        x0,y0,x1,y1 = np.array(box).astype(int)
        #cv.rectangle(masked_image, (x0, y0), (x1, y1), (255, 0, 0), 20)
        coords = [x0,y0,x1,y1]
        h = x1-x0
        stem_lengths.append(h)
        boxes.append(coords)


    ###Extract apples:
    #Masking 
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #blur = cv.blur(gray, (45,45))
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    thresh = cv.bitwise_not(thresh)

    #Find contours
    contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    #remove everything other than apple
    filtered_contours = [cnt for cnt in contours if 650000 < cv.contourArea(cnt) < 8000000]

    output_cp = im

    #Initialize data frame
    object_info = []

    #Draw apple contours
    cv.drawContours(output_cp, filtered_contours, -1, (0, 255, 0), 25)

    for apple_idx, contour in enumerate(filtered_contours):
        M = cv.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv.circle(output_cp, (cx, cy), 45, (0, 255, 255), 25) 
        
        x, y, w, h = cv.boundingRect(contour)
        #Extract section that overlaps stem
        sx0,sy0,sx1,sy1 = x, y, x+w//2, y+h

        #Find stem mask within roi
        for idx, mask in enumerate(masks):
            mask_roi = np.array(mask[sy0:sy1, sx0:sx1])
            if(np.any(mask_roi==1)): stem_id = idx
                

        if stem_id is not None:
            stem_length = stem_lengths[stem_id]
        
            coords = boxes[stem_id]
            bx0,by0,bx1,by1 = coords
        
            #Draw bounding box for stem
            cv.rectangle(output_cp, (bx0,by0),(bx1, by1), (255,0,0), 20)

            pass

        #Draw width
        cv.line(output_cp, (x + w // 2, y), (x + w // 2, y + h), (255, 0, 255), 25)
        
        
        width = h    
        
        object_info.append({
        'Object_ID': apple_idx + 1,
        'apple_width': width,        # Apple width
        'stem_length': stem_length,        # Stem length
        })


        stem_id = None
        stem_length = None


    output_path = os.path.join(output_folder, f'processed_{image_name}.jpeg')
    cv.imwrite(output_path, cv.cvtColor(output_cp, cv.COLOR_BGR2RGB))
    
    
    df_unit = pd.DataFrame(object_info)
    df_unit.insert(0, 'image', image_name)

    print(f"Done processing {image_name}")

    return(df_unit)



def main():
    # Initialize options
    # args = options()
    # folder,save= input("enter foldername directory and output directory: ").split()
    # input_folder=os.path.abspath(folder)
    # output_folder=os.path.abspath(save)

    result = f'FruitAnalysisResults_{date}.csv'

    parser = argparse.ArgumentParser(description="Enter input and output directory.")
    parser.add_argument("indir", type=str, help="Input directory with no final /")
    parser.add_argument("outdir", type=str, help="Output directory with no final /")
    args = parser.parse_args()
    input_folder = os.path.abspath(args.indir)
    output_folder= os.path.abspath(args.outdir)
    os.makedirs(output_folder, exist_ok=True)



    ext="jpeg"
    files = glob.glob(os.path.join(input_folder, f'*.{ext}'))
    
    with ThreadPoolExecutor() as executor:
        df = pd.concat(executor.map(run_pipeline, files, repeat(input_folder), repeat(output_folder)))    
        

    # Save to CSV in output folder
    df_output_path = os.path.join(output_folder, result)
    df.to_csv(df_output_path, index=False)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Done running all images in {elapsed_time:.4f} seconds")
        
if __name__ == "__main__":
    main() 