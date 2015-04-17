"""
Receptive fields
================

A subpackage that provides a bunch of receptive field estimation methods
"""

from .. import database # Ensures that all the modules have been loaded in their new locations *first*.
import sys
sys.modules['lnpy.database.database'] = database
