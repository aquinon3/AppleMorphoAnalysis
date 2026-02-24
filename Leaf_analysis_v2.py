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

    image_name = im.replace(f'{input_folder}/', "").replace(".jpg","")

    print(f"Processing: {image_name}")
    

    #Read in image
    im = cv.imread(im)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)


    # Segment using text prompt
    inputs = processor(images=im, text="leaf_with_stem", return_tensors="pt").to(device)

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

    # Stack and combine (logical OR across objects)
    if isinstance(masks, list):
        masks_tensor = torch.stack(masks)
    else:
        masks_tensor = masks

    combined_mask = masks_tensor.any(dim=0).cpu().numpy().astype(np.uint8) * 255

    image_np = np.array(im)
    masked_im = image_np * combined_mask[..., None]


    #Initialize dataframe
    object_info = []

    masked_image = masked_im.copy()

    for idx, leaf in enumerate(results['masks']):
        leaf = np.array(leaf, np.uint8) * 255
        
        # Find right, bottom-most point to get tip of petiole
        points = cv.findNonZero(leaf).reshape(-1,2)

        #Get convex hull of whole object
        cxh = cv.convexHull(points)
        
        ####Get start of petiole:
        #First, get contour, select biggest
        contours, _ = cv.findContours(leaf, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bigleaf = max(contours, key=cv.contourArea)

        #Find convex hull of contour, get deepest depression
        
        defects = cv.convexityDefects(bigleaf, cv.convexHull(bigleaf,  returnPoints=False))

        #select the defects with the largest depth
        petiole_pt1_id = np.argsort(defects[:,0,3])[::-1][0]

        #Convert defect info into coords
        #Get difference between deepest defect and all others
        defects_xy = []

        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            far = tuple(bigleaf[f][0])
            defects_xy.append(far)

        defects_xy = np.array(defects_xy)

        #Get pairwise distances between all points, subset column of deepest defect
        defect_distances = pairwise_distances(defects_xy, metric='euclidean')[:,petiole_pt1_id]
        #Subset closest non-zero defect
        p2 = defects_xy[np.argsort(defect_distances)[1]]
        #Extract pt1 coords
        p1 = defects_xy[petiole_pt1_id]
        #Now get coordinates for mid point
        (px, py) = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

        
        ###Get leaf tip and petiole end
        # Convert to (N, 2)
        points = cxh.reshape(-1, 2)

        # Pairwise vector differences (N, N, 2)
        diffs = points[:, None, :] - points[None, :, :]

        # Pairwise Euclidean distances (N, N)
        distances = np.linalg.norm(diffs, axis=2)

        # Ignore self-distances
        np.fill_diagonal(distances, -1)

        # Index of maximum distance
        flat_idx = np.argmax(distances)
        i, j = np.unravel_index(flat_idx, distances.shape)

        # Results
        max_distance = distances[i, j]
        (x1,y1) = points[i]
        (x2,y2) = points[j]
        
        d1 = int(np.linalg.norm(np.array([x1,y1]) - np.array([px,py])))
        d2 = int(np.linalg.norm(np.array([x2,y2]) - np.array([px,py])))
        
        if d1 > d2:
            blade_length = d1
            petiole_length = d2
            (lx, ly) = (x1,y1)
            (sx, sy) = (x2,y2)
        else:
            blade_length = d2
            petiole_length = d1
            (lx, ly) = (x2,y2)
            (sx, sy) = (x1,y1)
        
        
        #Draw petiole start
        cv.circle(masked_image,(px,py),10,[255, 0, 0],15)
        cv.circle(masked_image,(x1,y1),10,[0, 0, 255],15)
        cv.circle(masked_image,(x2,y2),10,[0, 0, 255],15)

        #Height from leaf tip to petiole start
        cv.line(masked_image, (lx, ly), (px,py), (0, 0, 255), 20)

        #Height from petiole start to petiole end
        cv.line(masked_image, (px,py), (sx,sy), (0, 255, 0), 20)
        
        
        object_info.append({
            'Leaf': idx + 1,
            'blade_length': blade_length,    
            'petiole_length': petiole_length
        })

    output_path = os.path.join(output_folder, f'processed_{image_name}.jpg')
    masked_image = cv.cvtColor(masked_image, cv.COLOR_BGR2RGB)
    cv.imwrite(output_path, masked_image)
    
    
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

    result = f'LeafAnalysisResults_{date}.csv'

    parser = argparse.ArgumentParser(description="Enter input and output directory.")
    parser.add_argument("indir", type=str, help="Input directory with no final /")
    parser.add_argument("outdir", type=str, help="Output directory with no final /")
    args = parser.parse_args()
    input_folder = os.path.abspath(args.indir)
    output_folder= os.path.abspath(args.outdir)
    os.makedirs(output_folder, exist_ok=True)


    ext="jpg"
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
