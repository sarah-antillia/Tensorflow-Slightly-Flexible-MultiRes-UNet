# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2023/05/22 Toshiyuki Arai
# create_mini_test.py

import os
import glob
import random
import shutil
import traceback

def create_mini_test():
  image_datapath = "./Breast-Cancer/test/malignant/images"
  orginal_datapath = "./Dataset_BUSI_with_GT/malignant"

  mini_test= "./mini_test"
  if os.path.exists(mini_test):
    shutil.rmtree(mini_test)
  if not os.path.exists(mini_test):
    os.makedirs(mini_test)
  
  files = glob.glob(image_datapath + "/*.jpg")

  files = random.sample(files, 10)
  for file in files:
    basename = os.path.basename(file)
    name     = basename.split(".")[0]
    image_file = os.path.join(orginal_datapath, name + ".png")
 
    shutil.copy2(image_file, mini_test)
    print("=== Copied {} to {}".format(image_file, mini_test))
if __name__ == "__main__":
  try:
    create_mini_test()
  except:
    traceback.print_exc()
