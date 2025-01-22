import os
import re

# This code loosely borrowed from imutils library but restructured to allow
# for following along symbolic links and 4 cases were split out
# to allow for the highest efficiency when iterating a list of directories
# with the fewest checks and string processing possible.
def list_files(basePath, validExts=None, contains=None) :
    if ((contains is not None) and (validExts is not None)) :
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # if the contains string is not none and the filename does not contain
                # the supplied string, then ignore the file
                if filename.find(contains) == -1 : continue

                # determine the file extension of the current file
                ext = filename[filename.rfind("."):].lower()

                # check to see if the file has one of the extensions specified
                if ext.endswith(validExts) :
                    # construct the path to the image and yield it
                    filePath = os.path.join(baseDir, filename)
                    yield filePath

    elif ((contains is None) and (validExts is not None)) :
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # determine the file extension of the current file
                ext = filename[filename.rfind("."):].lower()

                # check to see if the file has one of the extensions specified
                if ext.endswith(validExts) :
                    # construct the path to the image and yield it
                    filePath = os.path.join(baseDir, filename)
                    yield filePath

    elif ((contains is not None) and (validExts is None)) :
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # if the contains string is not none and the filename does not contain
                # the supplied string, then ignore the file
                if filename.find(contains) == -1 : continue

                # construct the path to the image and yield it
                filePath = os.path.join(baseDir, filename)
                yield filePath

    else : # ((contains is None) and (validExts is None))
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # construct the path to the image and yield it
                filePath = os.path.join(baseDir, filename)
                yield filePath


def list_images(basePath,image_types=(".pgm",".ppm",".jpg",".jpeg",".png",".bmp",".tif",".tiff"),contains=None) :
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)

def filter_filelist(filelist,filterstr=None,blocklist=None) :
    if filterstr is not None :
        filelist = [p for p in filelist if re.search(filterstr,p)]
    if blocklist is not None :
        # first convert list to set
        #print(f'First in filelist: {filelist[0]}')
        #print(f'First in blocklist: {blocklist[0]}')
        fileset = set(filelist)
        for b in blocklist : 
            if b in fileset :
                fileset.remove(b)
        filelist = list(fileset)
    filelist.sort()
    return filelist
    

