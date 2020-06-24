name = 'stlabutils'
# __all__ = ["metagen", "readdata", "S11fit", "newfile"]

# Loads utils with shorter names
# from stlabutils.X.Y import Z becomes stlabutils.Z

# File creation, callable as stlabutils.newfile(...)
from .newfile import *

# Metagen creation, callable as stlabutils.metagen.fromlimits(...)
from .metagen import *

# File reading, callable as stlabutils.readdata.readQUCS(...)
from .readdata import *

# File writing, callable as stlabutils.savetxt(...)
from .writematrix import *

from .stlabdict import *

# Fitting routines, callable as stlabutils.S11fit(...)
from .S11fit import *

# Autoplotter, callable as stlabutils.autoplot(...)
from .autoplotter import *

# git id, callable as stlabutils.get_gitid(...)
from .getgitid import *
