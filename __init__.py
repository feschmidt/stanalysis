# __all__ = ["metagen", "readdata", "S11fit", "newfile"]

# Loads utils with shorter names
# File creation, callable as stlabutils.newfile(...)
from stlabutils.utils.newfile import newfile

# Metagen creation, callable as stlabutils.metagen.fromlimits(...)
import stlabutils.utils.metagen as metagen

# File reading, callable as stlabutils.readdata.readQUCS(...)
import stlabutils.utils.readdata as readdata

# File writing, callable as stlabutils.savetxt(...)
from stlabutils.utils.writematrix import writematrix as savetxt
from stlabutils.utils.writematrix import writedict as savedict
from stlabutils.utils.writematrix import writedictarray as savedictarray
from stlabutils.utils.writematrix import writeparams as writeparams
from stlabutils.utils.writematrix import writeparnames as writeparnames
from stlabutils.utils.writematrix import params_to_str as params_to_str
from stlabutils.utils.writematrix import writeline as writeline
from stlabutils.utils.writematrix import writeframe as saveframe
from stlabutils.utils.writematrix import writeframearray as saveframearray
from stlabutils.utils.stlabdict import stlabdict
from stlabutils.utils.stlabdict import stlabmtx
from stlabutils.utils.stlabdict import framearr_to_mtx

# Fitting routines, callable as stlabutils.S11fit(...)
from stlabutils.utils.S11fit import fit as S11fit
from stlabutils.utils.S11fit import S11full as S11func
from stlabutils.utils.S11fit import backmodel as S11back
from stlabutils.utils.S11fit import S11theo as S11theo

# Autoplotter, callable as stlabutils.autoplot(...)
from stlabutils.utils.autoplotter import autoplot

# git id, callable as stlabutils.get_gitid(...)
from stlabutils.utils.getgitid import get_gitid
