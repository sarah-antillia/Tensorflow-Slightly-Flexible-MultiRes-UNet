# Copyright 2023 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# spit_to_train_test.py
# 2023/05/22 Antillia.com Toshiyuki Arai
# 
import os
import shutil
import glob
import traceback

def split_to_images_masks(input_dir, category, output_dir):
  

  datasets= ["/train/" + category, "/test/" + category]
  for dataset in datasets:
    output_images_dir = output_dir + dataset + "/images/"

    output_masks_dir  = output_dir + dataset + "/masks/"
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)
  
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    image_files = glob.glob(input_dir + dataset + "/*).jpg")    
    mask_files  = glob.glob(input_dir + dataset + "/*_mask.jpg")
    for image_file in image_files:
      shutil.copy2(image_file, output_images_dir)
      print("Copied {} to {}".format(image_file, output_images_dir))
    for mask_file in mask_files:
      shutil.copy2(mask_file, output_masks_dir)
      print("Copied {} to {}".format(mask_file, output_masks_dir))

if __name__ == "__main__":
  try:
    input_dir  = "./BUSI_augmented_master_256x256/"
    output_dir = "./Breast-Cancer"
    if not os.path.exists(input_dir):
      raise Exception("Not found input_dir " + input_dir)

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    category = "malignant"
    print("=== output_dir {}".format(output_dir))

    split_to_images_masks(input_dir, category, output_dir)
    
  except:
    traceback.print_exc()
