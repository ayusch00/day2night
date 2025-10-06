#!/usr/bin/env bash
cd data/cityscapes
wget --load-cookies cookies.txt "https://www.cityscapes-dataset.com/file-handling/?packageID=1" -O leftImg8bit_trainvaltest.zip
wget --load-cookies cookies.txt "https://www.cityscapes-dataset.com/file-handling/?packageID=3" -O gtFine_trainvaltest.zip
unzip gtFine_trainvaltest.zip -d cityscapes/
unzip leftImg8bit_trainvaltest.zip -d cityscapes/
rm *.zip
