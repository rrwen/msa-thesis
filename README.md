# Geospatial Semantic Pattern Recognition in Volunteered Geographic Data Using the Random forest Algorithm
Richard Wen (rwen@ryerson.ca)  
Masters of Spatial Analysis, Ryerson University, 2016  
Thesis Defended on April 27, 2016  
Supervised by Dr. Claus Rinner
* [PDF](https://github.com/rwenite/msa-thesis/blob/paper/thesis.pdf)
* [Defense Slides](https://rwenite.github.io/msa-thesis)

## Contents
[Abstract](https://github.com/rwenite/msa-thesis#abstract)  
[Code](https://github.com/rwenite/msa-thesis#code)  
* [Dependencies](https://github.com/rwenite/msa-thesis#dependencies)  
* [Windows Installation](https://github.com/rwenite/msa-thesis#windows-installation)  
* [Linux Installation](https://github.com/rwenite/msa-thesis#linux-installation)  
* [Run](https://github.com/rwenite/msa-thesis#run)  
[Information](https://github.com/rwenite/msa-thesis#information)  
* [Defense](https://github.com/rwenite/msa-thesis#defense)  
* [Hardware](https://github.com/rwenite/msa-thesis#hardware)
  
## Abstract
The ubiquitous availability of location technologies has enabled large quantities of Volunteered Geographic Data (VGD) to be produced by users worldwide. VGD has been a cost effective and scalable solution to obtaining unique and freely available geospatial data. However, VGD suffers from reliability issues as user behaviour is often variable. Large quantities make manual assessments of the user generated data inefficient, expensive, and impractical. This research utilized a random forest algorithm based on geospatial semantic variables in order to aid the improvement and understanding of multi-class VGD without ground-truth reference data. An automated Python script of a random forest based procedure was developed. A demonstration of the automated script on OpenStreetMap (OSM) data with user generated tags in Toronto, Ontario, was effective in recognizing patterns in the OSM data with predictive performances of ~71% based on a class weighted metric, and the ability to reveal variable influences and outliers.

## Code
The code was written in [Python 3.5](https://www.python.org/about/) and has been tested for the [Mapzen Toronto data](https://mapzen.com/data/metro-extracts/metro/toronto_canada/) for Windows and Linux operating systems. The code is described in Section 4 of the [PDF](https://github.com/rwenite/msa-thesis/blob/paper/thesis.pdf), which used a tree-optimized random forest model to learn geospatial patterns for the prediction and outlier detection of known spatial object classes (Figure 1).  
![Figure 1](https://github.com/rwenite/msa-thesis/blob/master/methods.png)  
Figure 1. Flowchart of code process  

### Dependencies
* [Anaconda Python 3.5](https://www.continuum.io/downloads/)  
* [GDAL](http://www.gdal.org/)  
* [Fiona](http://toblerity.org/fiona/manual.html)  
* [pyproj](https://github.com/jswhit/pyproj)  
* [Shapely](https://github.com/Toblerity/Shapely)  
* [geopandas](http://geopandas.org/)  
* [joblib](https://pythonhosted.org/joblib/)  
* [seaborn](https://stanford.edu/~mwaskom/software/seaborn/)  
* [treeinterpreter](https://github.com/andosa/treeinterpreter)  
* [tqdm](https://github.com/noamraph/tqdm)  
* [rtree](http://toblerity.org/rtree/)  
  
### Windows Installation
1. Install [Anaconda Python 3.5](https://www.continuum.io/downloads#windows) for windows
2. Download wheel files: [GDAL](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal), [Fiona](http://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona), [pyproj](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyproj), and [shapely](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely) for Python 3.5 (cp35)
3. Uninstall existing OSGeo4W, GDAL, Fiona, pyproj, or shapely libraries
4. Navigate to downloaded wheel files using the console `cd c:/path/to/downloaded_wheels`
5. Install the wheel (.whl) files and libraries using `pip install`  
  
*64-bit Example*
```shell
cd c:/path/to/downloaded_wheels
pip install GDAL-2.0.3-cp35-cp35m-win_amd64.whl
pip install Fiona-1.7.0-cp35-cp35m-win_amd64.whl
pip install pyproj-1.9.5.1-cp35-cp35m-win_amd64.whl
pip install Shapely-1.5.16-cp35-cp35m-win_amd64.whl
pip install geopandas
pip install joblib
pip install seaborn
pip install treeinterpreter
pip install tqdm
conda install -c ioos rtree
```  

*32-bit Example*
```shell
cd c:/path/to/downloaded_wheels
pip install GDAL-2.0.3-cp35-cp35m-win32.whl
pip install Fiona-1.7.0-cp35-cp35m-win32.whl
pip install pyproj-1.9.5.1-cp35-cp35m-win32.whl
pip install Shapely-1.5.16-cp35-cp35m-win32.whl
pip install geopandas
pip install joblib
pip install seaborn
pip install treeinterpreter
pip install tqdm
conda install -c ioos rtree
```  

Thanks to [Geoff Boeing](http://geoffboeing.com/about/) for the [Using geopandas on windows](http://geoffboeing.com/2014/09/using-geopandas-windows/) blog post and [Christoph Gohlke](http://www.lfd.uci.edu/~gohlke/) for the [wheel files](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
  
### Linux Installation
1. Install [Anaconda Python 3.5](https://www.continuum.io/downloads#linux) for linux  
2. Install libraries using `pip install` and `conda install`  
```shell
pip install treeinterpreter
pip install tqdm
conda install -c conda-forge geopandas
conda install joblib
conda install seaborn
conda install -c ioos rtree
``` 

### Run
1. Download [this repository](https://github.com/rwenite/msa-thesis/archive/master.zip)  
2. Unzip the file and navigate to the code folder `cd c:/path/to/msa-thesis-master/py`  
3. Execute the code using `python thesis.py`
```shell
cd c:/path/to/msa-thesis-master/py
python thesis.py config.txt c:/path/to/output_folder
```  
The config file can be used to apply and alter the methods to other datasets.  
Please see Section 4.1 in the [PDF](https://github.com/rwenite/msa-thesis/blob/paper/thesis.pdf) for more details.  
  
*Note: The config.txt file contains the settings used to obtain the results in the thesis. The results may be different if the recent version of the [Mapzen Toronto data](https://mapzen.com/data/metro-extracts/metro/toronto_canada/) is updated.*

# Information

### Defense
* **Date**: April 27, 2016
* **Time**: 2:00 p.m. to 4:00 p.m.
* **Location**: Jorgenson Hall 730, Ryerson University, Toronto, ON
* **Chair**: Dr. Lu Wang
* **Examiner 1**: Dr. Eric Vaz
* **Examiner 2**: Dr. Tony Hernandez
* **Result**: Pass with minor revisions  

### Hardware
Personal machine:
* Windows 8.1 64-bit  
* i7-6700k 4.0 GHz Quad-Core  
* 16 GB DDR4 2133 RAM  
* 256 GB SSD + 512 GB SSD (Read: Up to 540 MB/sec, Write: Up to 520 MB/sec)  
* **Runtime**: ~30-45 minutes  

Virtual machine generously provided by Ryerson [RC4](http://rc4.ryerson.ca/):
* Debian Linux  
* 6-Core CPU  
* 6 GB RAM  
* 66 GB Storage  
* **Runtime**: ~50-60 minutes  
  
