# -*- coding: utf-8 -*-

"""
 logs.py
 Richard Wen (rwenite@gmail.com)
 
===============================================================
 
 A custom module for managing and manipulating logging activity.
 
===============================================================
"""


"""
===============================================================
 Modules
===============================================================
"""


import logging


"""
===============================================================
 Functions
===============================================================
"""


def save_log(target, mode='a'):
    """
     save_log: str -> object
     
    ---------------------------------------------------------------
     
     Saves logging messages to a file specified by [target] and
     returns the logging object.
     
     Required Parameters
     -------------------
     * target: str
             The path to the log file for creation
             
     Returns
     -------
     * file_handler: (objectof logging)
             The logging object resulting from logging.FileHandler
             from the logging module
    
     Effects
     -------
     * Creates a file at [target] for saving log messages
     * Log messages will be appended to [target]
     
    ---------------------------------------------------------------
    """
    logger = logging.getLogger()
    
    # (Config_Handler) Configures the logging handler
    file_handler = logging.FileHandler(filename=target, mode=mode)
    formatter = logging.Formatter('%(asctime)s:%(msecs)s,"line %(lineno)s","%(pathname)s",%(levelname)s,"%(message)s"',
                                  datefmt = '"%d-%Y-%m","%H:%M:%S"')
    file_handler.setFormatter(formatter)
    
    # (Set_Logger) Configures the logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger
    
    