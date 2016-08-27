# -*- coding: utf-8 -*-

"""
 files.py
 Richard Wen (rwenite@gmail.com)
 
===============================================================
 
 A custom module for managing and manipulating files and 
 directories conveniently on the system.
 
===============================================================
"""


"""
===============================================================
 Modules
===============================================================
"""


import os
import time


from urllib.request import urlparse, urlretrieve 
from zipfile import ZipFile


"""
===============================================================
 Functions
===============================================================
"""


def add_folder(target):
    """
     add_folder: str bool -> Effect
     
    ---------------------------------------------------------------
     
     Creates a folder at the directory [target] if it does not exist.
     
     Required Parameters
     -------------------
     * target: str
             The path to the directory for creation
    
     Effects
     -------
     * Creates a folder at the [target] if it does not exist
     
     Examples
     --------
     * Create a folder if it doesent already exist
         add_folder('path/to/folder')
     
    ---------------------------------------------------------------
    """    
    if not os.path.exists(target):  # Create folder at [target]
        os.makedirs(target)
    else:
        print('Warning: ' + target + ' already exists.')


def date_created(src):
    """
     date_created: str -> str
     
    ---------------------------------------------------------------
     
     Consumes a source directory or file path [src] and returns the
     date that it was created as 'Day Month Number Hours Year'.
     
     Required Parameters
     -------------------
     * src: str
             The source directory folder or file path
    
     Returns
     -------
     * : str
             The creation date of [src]
             
     Examples
     --------
     * Get creation date from folder
         date_created('path/to/folder')
     * Get creation date from file
         date_created('path/to/file')
     
    ---------------------------------------------------------------
    """ 
    if os.path.exists(src):
        return time.ctime(os.path.getctime(src))
    else:
        print('Warning: ' + src + ' does not exist.')


def date_modified(src):
    """
     date_modified: str -> str
     
    ---------------------------------------------------------------
     
     Consumes a source directory or file path [src] and returns the
     date that it was modified as 'Day Month Number Hours Year'.
     
     Required Parameters
     -------------------
     * src: str
             The source directory folder or file path
             
     Returns
     -------
     * : str
             The modification date of [src]
     
     Examples
     --------
     * Get latest modification date from folder
         date_modified('path/to/folder')
     * Get latest modification date from file
         date_modified('path/to/file')
     
    ---------------------------------------------------------------
    """
    if os.path.exists(src):
        return time.ctime(os.path.getmtime(src))
    else:
        print('Warning: ' + src + ' does not exist.')


def dict_dir(src):
    """
     dict_dir: str -> (dictof str)
     
    ---------------------------------------------------------------
     
     Consumes a source directory [src] with files inside and returns
     a dictionary of files with their file paths. The keys will be
     the file names, and the values will be the full file paths.
     
     Required Parameters
     -------------------
     * src: str
             The source directory folder with files inside
    
     Returns
     -------
     * paths_dict: (dictof str)
             A dictionary in which the keys are the file names and the values
             are the full paths to the files relative to [src]
    
     Examples
     --------
     * Get dictionary of contents inside folder
         dict_dir('path/to/folder')
     
    ---------------------------------------------------------------
    """
    
    # (Return_Source) Return [src] if [src] is not a valid directory
    if os.path.isdir(src):
        files = os.listdir(src)
    elif os.path.isfile(src):
        return {os.path.basename(src): src}
    else:
        print('Warning: ' + src + ' is not a directory or a file.')
    
    # (Dictionary) Return the files and full paths at [src]
    paths_dict = {}
    for name in files:
        path = os.path.join(src, name)
        paths_dict[name] = path
    return paths_dict


def dl_file(src, target, overwrite=False):
    """
     dl_file: str str -> str
     
    ---------------------------------------------------------------
     
     Downloads the specified file from a source [src] if it is an
     http or https link to a target directory [target]. Returns the 
     source link [src] if the file is not downloaded, and the 
     target directory or file [target] if it is downloaded or 
     already exists.
     
     If [overwrite] is set to True, then it will download the file
     and overwrite the existing file [target] regardless.
     
     Required Parameters
     -------------------
     * src: str
             The url link with the specified file in it
     * target: str
             The target directory to download the file to
    
     Returns
     -------
     * target: str
             The target directory to download the file to
    
     Effects
     -------
     * Downloads a file from [src] to the directory/file [target]
     
     Examples
     --------
     * Download a file from an online source
         dl_file('http://www.repository.com/file', 'path/to/targetFolder')
     
    ---------------------------------------------------------------
    """    
    if os.path.exists(target) and overwrite == False:  # file exists, skip
        print('Warning: ' + target + ' already exists.')
        return target
    else:
        urlretrieve(src, target)
        print('Progress: ' + src + ' has been downloaded to ' + target)
        return target


def get_bytes(src):
    """
     get_bytes: str -> long
     
    ---------------------------------------------------------------
     
     Calculates the file size in bytes for [src] directory or file.
     Raises an error if the [src] is not a directory or a file.
     
     Required Parameters
     -------------------
     * src: str
             The source directory or file
             
     Returns
     -------
     * size: long
             The size in bytes of [src]
    
     Examples
     --------
     * Get the size of a folder
         get_bytes('path/to/folder')
     * Get the size of a file
         get_bytes('path/to/file')
     
    ---------------------------------------------------------------
    """
    if os.path.isfile(src):
        return int(os.path.getsize(src))
    elif os.path.isdir(src):
        files = list_dir(src, full_paths=True)
        size = 0
        for f in files:
            size += os.path.getsize(f)
        return size
    else:
        raise Exception('Error: ' + src + ' is not a file or directory.')


def list_dir(src, full_paths=False):
    """
     list_dir: str bool -> (listof str)
     
    ---------------------------------------------------------------
     
     Consumes a source directory [src] with files inside and returns
     a list of files or file paths. The user may choose to return
     full paths or just the file names by setting [full_paths].
     
     If the directory is invalid, [src] will be returned instead
     as a full path or file name depending on [full_paths].
     
     Required Parameters
     -------------------
     * src: str
             The source directory folder with files inside
    
     Optional Parameters
     -------------------
     * full_paths: bool
             Whether to return full paths or not
    
     Returns
     -------
     * paths: (listof str)
             The list of file paths
             
     Examples
     --------
     * Get the contents inside a folder
         list_dir('path/to/folder')
     
    ---------------------------------------------------------------
    """
    
    # (Return_Source) Return [src] if [src] is not a valid directory
    if os.path.isdir(src):
        files = os.listdir(src)
    elif os.path.isfile(src):
        if full_paths == True:
            return [src]
        else:
            return [os.path.basename(src)]
    else:
        raise Exception('Error: ' + src + ' is not a file or directory.')
    
    # (Return_Full_Paths) Return the full paths for directory [src]
    if full_paths == True:
        paths = []
        for name in files:
            path = os.path.join(src, name)
            paths.append(path)
        return paths
    else:
        return files


def unzip_file(src, target, clean=False, overwrite=False):
    """
     unzip_file: str str bool -> str AND Effect
     
    ---------------------------------------------------------------
     
     Unzips the specified source file [src] if it is a zip file to a
     target directory or file [target]. Returns the target 
     directory or file [target] if it is unzipped or already exists
     and the source file path [src] if it is not unzipped.
     
     Also removes the [src] file if [clean] is set to True.
     
     Required Parameters
     -------------------
     * src: str
             The zip file path with the specified file in it
     * target: str
             The target directory or file to unzip the file to
    
     Optional Parameters
     -------------------
     * clean: bool
             Whether to delete the zip file or not after unzipping
             
     Returns
     -------
     * target: str
             The target directory or file to unzip the file to
    
     Effects
     -------
     * Unzips the contents from [src] to the directory/file [target]
     
     Examples
     --------
     * Unzip the file to the target location
         unzip_file('path/to/file.zip', 'path/to/targetFolder')
     
    ---------------------------------------------------------------
    """
    if os.path.exists(target) and overwrite == False:
        print('Warning: ' + src + ' already exists.')
    else:
        with ZipFile(src, "r") as unzip:
            unzip.extractall(target)
        print('Progress: ' + src + ' has been unzipped to ' + target)
    if clean == True and os.path.exists(src):
        print('Warning: ' + src + ' has been overwritten.')
        os.remove(src)
    return target


"""
===============================================================
 Classes
===============================================================
"""     


class Fileset(object):    
    def __init__ (self, src, out='', clean=False, overwrite=False):
        """
         Fileset: str str bool bool -> obj
         
        ---------------------------------------------------------------
         
         Setup a fileset object from a [src] complete with references 
         to its [src], files, and full paths. The [src] can be a link
         or a local path, and it can also be in a zipped form.
         
         During initialization, online sources to files will be
         downloaded, and zipped files will be unzipped to the data
         folder under the specified path in [out].
         
         Required Parameters
         -------------------
         * src: str
                 The path to the fileset's source as either
                 an html link, zip file, or local folder that
                 contains a number of files to be managed
         
         Optional Parameters
         -------------------
         * out: str
                 The output directory path to store the file if the
                 [src] is a url or a zip file; does nothing otherwise
         * clean: bool
                 Whether or not to delete leftover zip files or
                 compressed files
         * overwrite: bool
                 Whether or not to overwrite existing files if
                 they already exist
        
         Effects
         -------
         * Downloads files from [src] to [out] if necessary
         * Unzips files from [src] to [out] if necessary
         
         Examples
         --------
         * Reference a local folder fileset
             files.Fileset('path/to/folder')
         * Reference to an online source
             files.Fileset('http://repository.com/data', 'path/to/localFolder')
         * Reference to a zip file
             files.Fileset('path/to/data.zip', 'path/to/localFolder')
         
        ---------------------------------------------------------------
        """
        if not (os.path.isfile(src) or os.path.isdir(src)):
            
            # (Path_Info) Obtain the [src] path info as parts
            src_with_ext = os.path.basename(src)
            src_without_ext = os.path.splitext(src_with_ext)[0]
            add_folder(out)
                
            # (Set_Targets) Set download and unzip targets
            dl_target = os.path.join(out, src_with_ext)
            unzip_target = os.path.join(out, src_without_ext)
            
            # (Download) Download files from [src] if required
            if not os.path.exists(unzip_target) and urlparse(src).scheme in ('http', 'https'):
                src = dl_file(src, dl_target, overwrite=overwrite)
            
            # (Unzip) Unzip files from [src] if required
            src_ext = os.path.splitext(src)[1]
            if src_ext == '.zip':
                src = unzip_file(src, unzip_target, clean=clean, overwrite=overwrite)
        
        # (Attributes) Set attributes for class
        self.src = src
        self.name = os.path.basename(os.path.splitext(src)[0])
        
        
    @property
    def total(self):
        """
         total -> int
         
        ---------------------------------------------------------------
         
         Returns the number of files for the directory or file at
         [self.src].
         
        ---------------------------------------------------------------
        """
        return len(list_dir(self.src))
    
    
    @property
    def created(self):
        """
         created -> str
         
        ---------------------------------------------------------------
         
         Returns the date that the fileset was created as
         'Day Month Number Hours Year'.
         
        ---------------------------------------------------------------
        """       
        return date_created(self.src)
    
    
    @property
    def dictionary(self):
        """
         dictionary -> (dictof str)
         
        ---------------------------------------------------------------
         
         Returns the file names and their full paths as a dictionary.
         The key will be the file names and the full paths will be the
         values.
         
        ---------------------------------------------------------------
        """       
        return dict_dir(self.src)
        
    
    @property
    def files(self):
        """
         files -> (listof str)
         
        ---------------------------------------------------------------
         
         Returns the file names for the directory or file at [self.src].
         The returned full names are in the form of a list.
         
        ---------------------------------------------------------------
        """        
        return list_dir(self.src)
    
    
    @property
    def modified(self):
        """
         modified -> str
         
        ---------------------------------------------------------------
         
         Returns the date that the file in the fileset was modified as
         'Day Month Number Hours Year'.
         
        ---------------------------------------------------------------
        """
        return date_modified(self.src)
    
    
    @property
    def paths(self):
        """
         paths -> (listof str)
         
        ---------------------------------------------------------------
         
         Returns the full paths for the directory or file at [self.src].
         The returned full paths are in the form of a list.
         
        ---------------------------------------------------------------
        """
        return list_dir(self.src, full_paths=True)
   
    
    @property
    def size(self):
        """
         size -> long
         
        ---------------------------------------------------------------
         
         Returns the size in bytes for the fileset.
         
        ---------------------------------------------------------------
        """
        return get_bytes(self.src)
    
    def apply_to_files(self,
                       func,
                       files=None,
                       *args,
                       **kwargs):
        """
         apply_to_files: function (listof str) str any any -> any
         
        ---------------------------------------------------------------
         
         Apply a function to all or select file paths. A list of the
         return values from the given [func] will be returned.
         
         Required Parameters
         -------------------
         * func: function
                 The function must be formatted such that:
                   * the first req arg corresponds to the file path
                   * the second optional arg corresponds to the output file
                     path respectively if [out_dir] is not None
                   * any addition arguments come after the conditions
                     above
                 Format:
                       def func(file_path, *args, **kwargs):
                           pass
        
         Optional Parameters
         -------------------
         * files: (listof str)
                 A select list of file names or full file paths inside
                 the Fileset [self.src] to apply the [func] to.
         * *args: any
                 Any additional arguments that can be passed to [func]
                 in order after [files] and [out_dir]. This is read as
                 a tuple.
         * **kwargs: any
                 Any additional keyword arguments that can be passed to [func]
                 that are not reserved as [files] and [out_dir]
                 This is read as a dict.
                 
         Returns
         -------
         * func_output: (dictof any)
                 A dictionary of returned values if any for each file, where
                 each key is the file name with extension (e.g 'file.txt') and 
                 each value contains the returned data given by [func]
                 
         Effects
         -------
         * Creates files inside [out_dir] if applicable for [func]
         * Creates a folder at [out_dir] if it does not exist
         * Effects are also varied by [func]
         
         Examples
         --------
         * Create a function to read a set of csvs and copy them to another
           directory. show is an additional argument that can be passed.
             import pandas as pd
             def copy_file(file_path, out_dir, show):
                 file_data = pd.read_csv(file_path)
                 out_path = os.path.join(out_dir, os.path.basename(file_path))
                 file_data.to_csv(out_path + ext)
                 if show:
                     print out_path
                 return out_path
         * Apply the copy_file function to all files in a Fileset, and
           print their output paths
             dataset = Fileset('path/to/dataset')
             dataset.apply_to_files(copy_files, out_dir = 'path/to/dir', show = True)

        ---------------------------------------------------------------
        """
        func_output = {}  # dict, keys are file names, vals are the func out
        for file_path in self.paths:
            file_base = os.path.basename(file_path)  # file name with ext
            func_output[file_base] = func(file_path, *args, **kwargs)
        return func_output
        
        