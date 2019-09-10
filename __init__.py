# __all__ = ["metagen", "readdata", "S11fit", "newfile"]

# Loads utils with shorter names
# from stlabutils.X.Y import Z becomes stlabutils.Z

# File creation, callable as stlabutils.newfile(...)
from stlabutils.utils.newfile import *

# Metagen creation, callable as stlabutils.metagen.fromlimits(...)
from stlabutils.utils.metagen import *

# File reading, callable as stlabutils.readdata.readQUCS(...)
from stlabutils.utils.readdata import *

# File writing, callable as stlabutils.savetxt(...)
from stlabutils.utils.writematrix import *

from stlabutils.utils.stlabdict import *

# Fitting routines, callable as stlabutils.S11fit(...)
from stlabutils.utils.S11fit import *

# Autoplotter, callable as stlabutils.autoplot(...)
from stlabutils.utils.autoplotter import *

# git id, callable as stlabutils.get_gitid(...)
from stlabutils.utils.getgitid import *
