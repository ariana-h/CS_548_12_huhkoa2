#!/usr/bin/python3
import os
DIRNAME="Assign02/Facades/A/train"
files = os.listdir(DIRNAME)
for f in files:
    if '.jpg' in f:
        newname = f.split('_')[0]
        newname = newname + '.jpg'
        os.rename(os.path.join(DIRNAME,f), os.path.join(DIRNAME,newname))
        
DIRNAME="Assign02/Facades/B/train"
files = os.listdir(DIRNAME)
for f in files:
    if '.jpg' in f:
        newname = f.split('_')[0]
        newname = newname + '.jpg'
        os.rename(os.path.join(DIRNAME,f), os.path.join(DIRNAME,newname))